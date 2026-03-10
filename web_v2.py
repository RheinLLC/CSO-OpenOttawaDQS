import csv
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype

from metadata import fetch_metadata_xml_from_dataset_url
from xml_to_jsonl import convert_xml_dir_to_jsonl
from jsonl_cleaning import clean_jsonl

WEIGHTS = {
    "freshness": 0.35,
    "metadata": 0.35,
    "accessibility": 0.15,
    "completeness": 0.10,
    "usability": 0.05,
}

PARSE_OK_RATE = 0.60
STALE_YEARS = 2
STOP_WORDS = {"id", "uid", "key", "value", "val", "x", "y", "col", "column", "unnamed"}
IMPORTANT_COL_HINTS = ["date", "year", "count", "total", "value", "rate", "amount", "type", "category", "class"]

try:
    _ON_BAD_LINES_SUPPORTED = "on_bad_lines" in pd.read_csv.__code__.co_varnames  # type: ignore[attr-defined]
except Exception:
    _ON_BAD_LINES_SUPPORTED = False


def issue(metric: str, detail: str, severity: str) -> Dict[str, str]:
    return {"metric": metric, "detail": detail, "severity": severity}


def likert_from_score(score_0_100: float) -> int:
    if score_0_100 >= 90:
        return 5
    if score_0_100 >= 80:
        return 4
    if score_0_100 >= 70:
        return 3
    if score_0_100 >= 60:
        return 2
    return 1


def grade_from_score(score_0_100: float) -> str:
    if score_0_100 >= 80:
        return "A"
    if score_0_100 >= 60:
        return "B"
    return "C"


def compute_total_score(scores: Dict[str, float]) -> float:
    return float(np.clip(sum(WEIGHTS[k] * scores[k] for k in WEIGHTS), 0, 100))


def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    orig = list(df.columns)
    new = []
    for col in orig:
        clean = str(col).replace("\ufeff", "").strip()
        clean = re.sub(r"\s+", " ", clean)
        new.append(clean)
    out = df.copy()
    out.columns = new
    changes = {o: n for o, n in zip(orig, new) if o != n}
    return out, changes


def is_vestigial(col: str) -> bool:
    return bool(re.match(r"^(objectid|fid|gid|shape|shape__area|shape__length|x|y)$", str(col).strip().lower()))


def is_important_col(col: str) -> bool:
    c = str(col).strip().lower()
    if is_vestigial(c):
        return False
    return any(hint in c for hint in IMPORTANT_COL_HINTS)


def is_comma_number_text(series: pd.Series) -> bool:
    if series.dtype != object:
        return False
    sample = series.dropna().astype(str).head(80)
    patt = r"^\s*-?\d{1,3}(,\d{3})+(\.\d+)?\s*$"
    return any(re.search(patt, x) for x in sample)


def to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def is_date_like(series: pd.Series) -> bool:
    if series.dtype != object:
        return False
    sample = series.dropna().astype(str).head(80).str.strip()
    patterns = [
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?",
        r"\d{1,2}[-/]\d{1,2}[-/]\d{4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?",
        r"\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}",
        r"[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}",
    ]
    return any(re.search(patt, x) for patt in patterns for x in sample)


def parse_datetime_safe(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        s2 = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    else:
        s2 = series
    dt = pd.to_datetime(s2, format="%d-%b-%y", errors="coerce")
    if dt.notna().sum() >= max(1, int(0.6 * s2.notna().sum())):
        return dt
    return pd.to_datetime(s2, errors="coerce")


def _read_csv_attempt(path: str, **kwargs) -> pd.DataFrame:
    defaults = dict(sep=",", engine="c")
    defaults.update(kwargs)
    if str(defaults.get("engine", "c")).lower() == "python":
        defaults.pop("low_memory", None)
    else:
        defaults.setdefault("low_memory", False)
    if _ON_BAD_LINES_SUPPORTED and "on_bad_lines" not in defaults:
        defaults["on_bad_lines"] = "skip"
    else:
        defaults.setdefault("error_bad_lines", False)  # type: ignore
        defaults.setdefault("warn_bad_lines", True)  # type: ignore
    return pd.read_csv(path, **defaults)


def _sniff_delimiter(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(4096)
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        return ","


def _is_probably_xlsx(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def read_csv_flex(path: str) -> pd.DataFrame:
    if _is_probably_xlsx(path):
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.read_excel(path, engine="openpyxl")

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return _read_csv_attempt(path, encoding=enc, engine="c")
        except Exception as exc:
            last_err = exc

    delim = _sniff_delimiter(path)
    for enc in encodings:
        try:
            return _read_csv_attempt(path, encoding=enc, engine="python", sep=delim, quoting=csv.QUOTE_MINIMAL)
        except Exception as exc:
            last_err = exc

    try:
        return pd.read_csv(path, engine="python", sep=delim, encoding="utf-8-sig", encoding_errors="replace")  # type: ignore[arg-type]
    except TypeError:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            return pd.read_csv(f, engine="python", sep=delim)
    except Exception as exc:
        raise last_err or exc


def clean_df(df: pd.DataFrame, issues_list: List[Dict[str, str]]) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    df, changes = standardize_columns(df)
    if changes:
        issues_list.append(issue("usability", "Standardized column names (trimmed/collapsed spaces).", "low"))

    for col in df.columns:
        if is_comma_number_text(df[col]):
            converted = to_numeric_clean(df[col])
            non_null = int(df[col].notna().sum())
            ok = int(converted.notna().sum())
            if non_null > 0 and ok / non_null >= PARSE_OK_RATE:
                df[col] = converted
                issues_list.append(issue("usability", f"Converted comma-number text to numeric in '{col}'.", "low"))

    for col in df.columns:
        if is_date_like(df[col]):
            dt = parse_datetime_safe(df[col])
            non_null = int(df[col].notna().sum())
            ok = int(dt.notna().sum())
            if non_null > 0 and ok / non_null >= PARSE_OK_RATE:
                df[col] = dt
                issues_list.append(issue("usability", f"Parsed datetime in '{col}'.", "low"))

    if "Year" not in df.columns:
        dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df["Year"] = df[dt_cols[0]].dt.year.astype("Int64")
            issues_list.append(issue("usability", f"Derived Year from '{dt_cols[0]}'.", "low"))
        elif "session" in df.columns:
            yrs = df["session"].astype(str).str.extract(r"(\d{4})")[0]
            yrs = pd.to_numeric(yrs, errors="coerce")
            if yrs.notna().sum() > 0:
                df["Year"] = yrs.astype("Int64")
                issues_list.append(issue("usability", "Derived Year from 'session'.", "low"))
    else:
        if not pd.api.types.is_numeric_dtype(df["Year"]):
            yr = pd.to_numeric(df["Year"], errors="coerce")
            non_null = int(df["Year"].notna().sum())
            if non_null == 0 or yr.notna().sum() >= max(1, int(PARSE_OK_RATE * non_null)):
                df["Year"] = yr.astype("Int64")
                issues_list.append(issue("usability", "Converted Year to numeric.", "low"))

    return df, issues_list


def score_completeness(df: pd.DataFrame, issues_list: List[Dict[str, str]]) -> Tuple[float, List[Dict[str, str]]]:
    score = 100.0
    weighted_missing = []
    for col in df.columns:
        miss = float(df[col].isna().mean()) if len(df) else 0.0
        if miss <= 0:
            continue
        if is_vestigial(col):
            weight = 0.25
        elif is_important_col(col):
            weight = 1.25
        else:
            weight = 1.0
        weighted_missing.append(weight * miss)
        if miss >= 0.50 and not is_vestigial(col):
            issues_list.append(issue("completeness", f"Column '{col}' has {miss:.0%} missing values.", "high"))

    if weighted_missing:
        score -= 60.0 * float(np.mean(weighted_missing))

    if "Year" in df.columns:
        years = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int)
        if len(years) >= 2:
            expected = set(range(int(years.min()), int(years.max()) + 1))
            missing_years = sorted(expected - set(years))
            if missing_years:
                score -= 25.0
                issues_list.append(issue("completeness", f"Missing entire year(s): {missing_years}.", "high"))

    return float(np.clip(score, 0, 100)), issues_list


def score_freshness(
    df: pd.DataFrame,
    issues_list: List[Dict[str, str]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[Dict[str, str]]]:
    now = datetime.now()
    if metadata:
        freq = metadata.get("update_frequency")
        last_updated = metadata.get("last_updated")
        if freq == "one-off":
            return 100.0, issues_list
        if isinstance(last_updated, datetime):
            age_days = (now - last_updated).days
            allowed_lag = {
                "daily": 2,
                "weekly": 10,
                "monthly": 45,
                "annual": 400,
            }.get(str(freq).lower() if freq else "annual", 400)
            if age_days <= allowed_lag:
                return 100.0, issues_list
            score = max(0.0, 100.0 - (age_days - allowed_lag) * 0.15)
            issues_list.append(
                issue(
                    "freshness",
                    f"Dataset is {age_days} days old (expected {freq or 'annual'} refresh).",
                    "high",
                )
            )
            return float(np.clip(score, 0, 100)), issues_list

    current_year = now.year
    if "Year" in df.columns:
        years = pd.to_numeric(df["Year"], errors="coerce").dropna()
        if not years.empty:
            ymax = int(years.max())
            age = current_year - ymax
            if age <= STALE_YEARS:
                return 100.0, issues_list
            score = max(0.0, 100.0 - 15.0 * (age - STALE_YEARS))
            issues_list.append(issue("freshness", f"Latest year {ymax} is {age} years behind {current_year}.", "medium"))
            return float(np.clip(score, 0, 100)), issues_list

    dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    if dt_cols:
        col = dt_cols[0]
        dt = df[col].dropna()
        if not dt.empty:
            latest = pd.to_datetime(dt).max()
            age_years = current_year - int(latest.year)
            if age_years <= STALE_YEARS:
                return 100.0, issues_list
            score = max(0.0, 100.0 - 15.0 * (age_years - STALE_YEARS))
            issues_list.append(issue("freshness", f"Latest date in '{col}' is {latest.date()} (age≈{age_years} years).", "medium"))
            return float(np.clip(score, 0, 100)), issues_list

    issues_list.append(issue("freshness", "No schedule metadata and no usable Year/date found; freshness uncertain.", "medium"))
    return 40.0, issues_list


def meaningful_name_ratio(columns) -> float:
    good = 0
    for col in columns:
        c0 = str(col).strip().lower()
        if c0.startswith("unnamed"):
            continue
        tokens = re.split(r"[^a-z0-9]+", c0)
        tokens = [token for token in tokens if token and token not in STOP_WORDS]
        if any(len(token) >= 3 for token in tokens):
            good += 1
    return good / max(1, len(columns))


def score_usability(df: pd.DataFrame, issues_list: List[Dict[str, str]]) -> Tuple[float, List[Dict[str, str]]]:
    score = 100.0
    mratio = meaningful_name_ratio(df.columns)
    if mratio < 0.20:
        issues_list.append(issue("usability", f"Low meaningful column name ratio: {mratio:.0%}.", "high"))
        score -= 50
    elif mratio < 0.50:
        issues_list.append(issue("usability", f"Moderate column name meaningfulness: {mratio:.0%}.", "medium"))
        score -= 20

    const_cols = []
    for col in df.columns:
        nn = df[col].dropna()
        if len(nn) == 0:
            if not is_vestigial(col):
                const_cols.append(col)
        elif nn.nunique() == 1 and not is_vestigial(col):
            const_cols.append(col)
    if const_cols:
        issues_list.append(issue("usability", f"{len(const_cols)} constant/empty columns may reduce usability.", "medium"))
        score -= min(25, 3 * len(const_cols))

    num_text_cols = [col for col in df.columns if is_comma_number_text(df[col])]
    if num_text_cols:
        issues_list.append(
            issue(
                "usability",
                f"Numeric-like text columns require preprocessing: {num_text_cols[:6]}{'...' if len(num_text_cols) > 6 else ''}",
                "medium",
            )
        )
        score -= 10

    obj_cols = [col for col in df.columns if df[col].dtype == object]
    risk_cols = []
    for col in obj_cols[:50]:
        s = df[col].dropna().astype(str).head(200)
        if s.empty:
            continue
        if s.str.match(r"^\s*\d{1,2}[-/]\d{1,2}\s*$").any() or s.str.match(r"^\s*\d{1,2}[-/]\d{1,2}[-/]\d{2}\s*$").any():
            risk_cols.append(col)
    if risk_cols:
        issues_list.append(
            issue(
                "usability",
                f"Excel auto-date conversion risk in fields: {risk_cols[:6]}{'...' if len(risk_cols) > 6 else ''}",
                "low",
            )
        )
        score -= 5

    cr_cols = []
    for col in obj_cols[:50]:
        s = df[col].dropna().astype(str).head(200)
        if not s.empty and s.str.contains(r"\r|\n").any():
            cr_cols.append(col)
    if cr_cols:
        issues_list.append(
            issue(
                "usability",
                f"Carriage returns/newlines found in text fields: {cr_cols[:6]}{'...' if len(cr_cols) > 6 else ''}",
                "low",
            )
        )
        score -= 3

    return float(np.clip(score, 0, 100)), issues_list


def score_accessibility(csv_path: str, df: pd.DataFrame, issues_list: List[Dict[str, str]]) -> Tuple[float, List[Dict[str, str]]]:
    del csv_path
    score = 100.0
    if df.shape[1] > 120:
        issues_list.append(issue("accessibility", f"Very high column count ({df.shape[1]}) may reduce accessibility in spreadsheets.", "medium"))
        score -= 15
    elif df.shape[1] > 80:
        issues_list.append(issue("accessibility", f"High column count ({df.shape[1]}) may reduce accessibility in spreadsheets.", "low"))
        score -= 8

    obj_cols = [col for col in df.columns if df[col].dtype == object]
    if obj_cols:
        sample_lens = []
        for col in obj_cols[:10]:
            s = df[col].dropna().astype(str).head(200)
            if not s.empty:
                sample_lens.append(float(s.map(len).mean()))
        if sample_lens and float(np.mean(sample_lens)) > 200:
            issues_list.append(issue("accessibility", "Long free-text fields may limit some users/tools (large cells).", "low"))
            score -= 5
    return float(np.clip(score, 0, 100)), issues_list


def _read_clean_jsonl_record(clean_jsonl_path: str) -> Optional[Dict[str, Any]]:
    with open(clean_jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _parse_update_frequency(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    if re.search(r"update\s*frequency\s*:\s*as\s*needed|as-needed|one[-\s]?off|sporadic", t):
        return "one-off"
    if re.search(r"update\s*frequency\s*:\s*daily|\bdaily\b", t):
        return "daily"
    if re.search(r"update\s*frequency\s*:\s*weekly|\bweekly\b", t):
        return "weekly"
    if re.search(r"update\s*frequency\s*:\s*monthly|\bmonthly\b", t):
        return "monthly"
    if re.search(r"update\s*frequency\s*:\s*annual|yearly|annually", t):
        return "annual"
    return None


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, dict):
        value = value.get("#text")
    if not isinstance(value, str):
        return None
    value = value.replace("Z", "")
    for fmt in [None, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            if fmt is None:
                return datetime.fromisoformat(value)
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    return None


def portal_record_to_metadata(record: Dict[str, Any], dataset_url: str) -> Dict[str, Any]:
    di = record.get("dataIdInfo", {}) if isinstance(record, dict) else {}
    title = (((di.get("idCitation") or {}).get("resTitle") or {}).get("#text")) if isinstance(di, dict) else None
    id_abs = (di.get("idAbs") or {}).get("#text") if isinstance(di.get("idAbs"), dict) else ""
    id_purp = (di.get("idPurp") or {}).get("#text") if isinstance(di.get("idPurp"), dict) else ""

    keywords: List[str] = []
    search_keys = di.get("searchKeys")
    if isinstance(search_keys, dict):
        keyword_node = search_keys.get("keyword")
        if isinstance(keyword_node, list):
            for kw in keyword_node:
                if isinstance(kw, dict) and kw.get("#text"):
                    keywords.append(str(kw["#text"]))
        elif isinstance(keyword_node, dict) and keyword_node.get("#text"):
            keywords.append(str(keyword_node["#text"]))

    date_obj = (((di.get("idCitation") or {}).get("date") or {})) if isinstance(di, dict) else {}
    create_dt = _parse_dt(date_obj.get("createDate"))
    revise_dt = _parse_dt(date_obj.get("reviseDate"))
    free_text = "\n".join([str(id_abs or ""), str(id_purp or ""), " ".join(keywords)]).strip()

    return {
        "title": title,
        "description_text": free_text or None,
        "description_excerpt": str(id_purp or "").strip() or None,
        "update_frequency": _parse_update_frequency(free_text),
        "last_updated": revise_dt or create_dt,
        "source": "portal",
        "dataset_url": dataset_url,
        "raw_record": record,
    }


def build_portal_metadata_from_url(dataset_url: str, workspace: str) -> Dict[str, Any]:
    xml_dir = os.path.join(workspace, "xml")
    raw_jsonl = os.path.join(workspace, "dataIdInfo_only.jsonl")
    clean_jsonl_path = os.path.join(workspace, "dataIdInfo_only_clean.jsonl")
    skip_list = os.path.join(workspace, "skipped_parse_error.txt")

    download_result = fetch_metadata_xml_from_dataset_url(dataset_url, xml_dir)
    convert_result = convert_xml_dir_to_jsonl(xml_dir, raw_jsonl, skip_list)
    clean_result = clean_jsonl(raw_jsonl, clean_jsonl_path)
    record = _read_clean_jsonl_record(clean_jsonl_path)
    if not record:
        raise ValueError("Metadata XML was downloaded, but no usable dataIdInfo record could be extracted.")

    meta = portal_record_to_metadata(record, dataset_url)
    meta["pipeline"] = {
        "download": download_result,
        "convert": convert_result,
        "clean": clean_result,
        "raw_jsonl": raw_jsonl,
        "clean_jsonl": clean_jsonl_path,
    }
    return meta


def score_metadata(
    csv_path: str,
    df: pd.DataFrame,
    issues_list: List[Dict[str, str]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[Dict[str, str]]]:
    del csv_path
    score = 100.0
    has_portal_desc = bool(metadata and metadata.get("source") == "portal" and metadata.get("description_text"))
    if not has_portal_desc:
        issues_list.append(issue("metadata", "Portal metadata description was not available.", "high"))
        score -= 30

    description_text = str(metadata.get("description_text")) if metadata and metadata.get("description_text") else None
    if description_text:
        desc = description_text.lower()
        missing_desc = [col for col in df.columns if (str(col).lower() not in desc) and not is_vestigial(col)]
        if missing_desc:
            score -= min(40, 3 * len(missing_desc))
            issues_list.append(issue("metadata", f"{len(missing_desc)} fields not described in metadata/description.", "medium"))
        if len(desc.strip()) < 80:
            score -= 10
            issues_list.append(issue("metadata", "Metadata description is very short; may be insufficient.", "medium"))
    else:
        score -= 30
        issues_list.append(issue("metadata", "No description text found in extracted portal metadata.", "high"))

    unnamed = [col for col in df.columns if str(col).strip().lower().startswith("unnamed")]
    if unnamed:
        issues_list.append(issue("metadata", f"Unnamed columns detected: {unnamed[:4]}{'...' if len(unnamed) > 4 else ''}.", "high"))
        score -= 20
    return float(np.clip(score, 0, 100)), issues_list


def _bucket(score: float) -> str:
    if score >= 80:
        return "good"
    if score >= 60:
        return "ok"
    return "low"


DIMENSION_ADVICE = {
    "completeness_score": {
        "do": ["Use for high-level aggregation and trend analysis", "Use for exploratory analysis (EDA)"],
        "avoid": ["Avoid record-level or individual decision-making", "Avoid claiming exact totals without coverage checks"],
    },
    "freshness_score": {
        "do": ["Use for historical context and long-term trends"],
        "avoid": ["Avoid real-time or current-state decision making"],
    },
    "metadata_score": {
        "do": ["Use internally by experienced analysts"],
        "avoid": ["Avoid public-facing dashboards without added documentation"],
    },
    "usability_score": {
        "do": ["Use after preprocessing and cleaning"],
        "avoid": ["Avoid rapid analysis without validation"],
    },
    "accessibility_score": {
        "do": ["Use in programmatic workflows (Python/R)"],
        "avoid": ["Avoid expecting easy spreadsheet-based use"],
    },
}


def generate_usage_guidance(row: pd.Series) -> pd.Series:
    do, avoid = [], []
    total = row["total_score_0_100"]
    if total >= 80:
        do.append("Suitable for dashboards and general analysis")
    elif total >= 60:
        do.append("Suitable for analysis with clear limitations")
        avoid.append("Avoid precise lookups or operational decisions")
    else:
        do.append("Suitable for prototypes, education, and context only")
        avoid.append("Avoid decision-making or public reporting")

    for dim in DIMENSION_ADVICE:
        score = row[dim]
        if _bucket(score) == "low":
            do.extend(DIMENSION_ADVICE[dim]["do"])
            avoid.extend(DIMENSION_ADVICE[dim]["avoid"])

    do = list(dict.fromkeys(do))
    avoid = list(dict.fromkeys(avoid))
    return pd.Series({"Recommended_Use": " | ".join(do), "Not_Recommended_For": " | ".join(avoid)})


def run_assessment(uploaded_path: str, dataset_url: str) -> Dict[str, Any]:
    workspace = tempfile.mkdtemp(prefix="ottawa_dqs_")
    metadata = build_portal_metadata_from_url(dataset_url, workspace)
    df = read_csv_flex(uploaded_path)
    issues_list: List[Dict[str, str]] = []
    df_clean, issues_list = clean_df(df, issues_list)

    usability, issues_list = score_usability(df_clean, issues_list)
    completeness, issues_list = score_completeness(df_clean, issues_list)
    freshness, issues_list = score_freshness(df_clean, issues_list, metadata=metadata)
    metadata_score, issues_list = score_metadata(uploaded_path, df_clean, issues_list, metadata=metadata)
    accessibility, issues_list = score_accessibility(uploaded_path, df_clean, issues_list)

    scores = {
        "usability": usability,
        "metadata": metadata_score,
        "freshness": freshness,
        "completeness": completeness,
        "accessibility": accessibility,
    }
    total = compute_total_score(scores)

    score_row = {
        "Report Identification": Path(uploaded_path).name,
        "usability_score": round(usability, 2),
        "metadata_score": round(metadata_score, 2),
        "freshness_score": round(freshness, 2),
        "completeness_score": round(completeness, 2),
        "accessibility_score": round(accessibility, 2),
        "total_score_0_100": round(total, 2),
        "Likert Score": likert_from_score(total),
        "Grade": grade_from_score(total),
        "rows": int(len(df_clean)),
        "cols": int(df_clean.shape[1]),
    }
    score_df = pd.DataFrame([score_row])
    guidance = score_df.apply(generate_usage_guidance, axis=1)
    score_df = pd.concat([score_df, guidance], axis=1)

    issues_df = pd.DataFrame(issues_list)
    if not issues_df.empty:
        issues_df = issues_df[["metric", "severity", "detail"]]

    return {
        "df_raw": df,
        "df_clean": df_clean,
        "metadata": metadata,
        "scores": scores,
        "score_df": score_df,
        "issues_df": issues_df,
        "total": total,
        "workspace": workspace,
    }


st.set_page_config(page_title="Open Ottawa DQS", layout="wide")
st.title("Open Ottawa DQS")
st.caption("Single-file assessment with runtime Open Ottawa metadata retrieval.")

dataset_url = st.text_input(
    "Open Ottawa dataset URL",
    placeholder="https://open.ottawa.ca/datasets/...",
    help="Required. Used to retrieve ArcGIS metadata.xml and derive portal metadata.",
)
uploaded = st.file_uploader("Upload a CSV or Excel export", type=["csv", "txt", "xlsx"])

if not dataset_url or uploaded is None:
    st.info("Provide the Open Ottawa dataset URL and upload one dataset file to run the assessment.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
    tmp.write(uploaded.getbuffer())
    upload_path = tmp.name

try:
    with st.spinner("Fetching portal metadata, extracting XML fields, cleaning metadata, and scoring dataset..."):
        result = run_assessment(upload_path, dataset_url)
except Exception as exc:
    st.error(str(exc))
    st.stop()
finally:
    try:
        os.unlink(upload_path)
    except OSError:
        pass

score_df = result["score_df"]
issues_df = result["issues_df"]
metadata = result["metadata"]

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Preview")
    st.dataframe(result["df_raw"].head(20), use_container_width=True)
    st.divider()
    st.subheader("Extracted Portal Metadata")
    meta_view = {
        "title": metadata.get("title"),
        "update_frequency": metadata.get("update_frequency"),
        "last_updated": metadata.get("last_updated").isoformat()
        if isinstance(metadata.get("last_updated"), datetime)
        else None,
        "dataset_url": metadata.get("dataset_url"),
    }
    st.json(meta_view)
    if metadata.get("description_excerpt"):
        st.markdown("**Description excerpt**")
        st.write(str(metadata["description_excerpt"])[:1200])


with right:
    st.subheader("Quality Score")
    st.metric("Total Score (0–100)", f"{result['total']:.2f}")
    st.write(
        {
            "Likert Score (1–5)": int(score_df.iloc[0]["Likert Score"]),
            "Grade": score_df.iloc[0]["Grade"],
        }
    )
    st.json({f"{k}_score": round(v, 2) for k, v in result["scores"].items()})

    st.divider()
    st.subheader("Usage Guidance")
    st.markdown("**Recommended_Use**")
    st.write(score_df.iloc[0]["Recommended_Use"] or "—")
    st.markdown("**Not_Recommended_For**")
    st.write(score_df.iloc[0]["Not_Recommended_For"] or "—")

st.divider()
st.subheader("DQS Issue Log")
if issues_df.empty:
    st.info("No issues detected by the current rule-based checks.")
else:
    st.dataframe(issues_df, use_container_width=True, hide_index=True)