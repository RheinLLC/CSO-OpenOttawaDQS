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
SIDECAR_EXTS = [".md", ".txt", ".json", ".yml", ".yaml", ".pdf", ".docx"]

# 与 v5 对齐
METADATA_JSONL_PATH = "dataIdInfo_only_clean.jsonl"
METADATA_CATALOG_JSON_PATH = "ottawa_catalog.json"

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
    # 保留 web_v2 原本 A/B/C 映射
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


# =========================
# Sidecar metadata helpers
# =========================
def find_sidecar_metadata_files(csv_path: str) -> List[str]:
    base = os.path.splitext(csv_path)[0]
    found = []
    for ext in SIDECAR_EXTS:
        cand = base + ext
        if os.path.exists(cand):
            found.append(cand)
    return found


def _read_text_sidecar(path: str, max_chars: int = 8000) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".md", ".txt", ".yml", ".yaml", ".json"]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(max_chars)
        except Exception:
            try:
                with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                    return f.read(max_chars)
            except Exception:
                return None
    return None


def infer_metadata_from_sidecars(sidecars: List[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    texts = []
    for p in sidecars:
        t = _read_text_sidecar(p)
        if t:
            texts.append(t)
    if not texts:
        return meta

    combined = "\n".join(texts)
    meta["description_text"] = combined

    freq_map = {
        "daily": r"\bdaily\b",
        "weekly": r"\bweekly\b",
        "monthly": r"\bmonthly\b",
        "annual": r"\bannual(?:ly)?\b|\byearly\b",
        "one-off": r"\bone[-\s]?off\b|\bas needed\b|\bsporadic\b",
    }
    for k, patt in freq_map.items():
        if re.search(patt, combined, flags=re.I):
            meta["update_frequency"] = k
            break

    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", combined)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            meta["last_updated"] = datetime(y, mo, d)
        except Exception:
            pass

    return meta


# =========================
# Portal/Catalog metadata helpers
# =========================
def _norm_name(s: str) -> str:
    s = (s or "").lower()
    s = os.path.splitext(s)[0]
    s = re.sub(r"__.*$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set(s: str) -> set:
    return {t for t in _norm_name(s).split(" ") if t and t not in STOP_WORDS}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_update_frequency(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    if re.search(r"update\s*frequency\s*:\s*as\s*needed|as-needed", t):
        return "one-off"
    if re.search(r"update\s*frequency\s*:\s*not\s*applicable|one[-\s]?time", t):
        return "one-off"
    if re.search(r"update\s*frequency\s*:\s*daily", t):
        return "daily"
    if re.search(r"update\s*frequency\s*:\s*weekly", t):
        return "weekly"
    if re.search(r"update\s*frequency\s*:\s*monthly", t):
        return "monthly"
    if re.search(r"update\s*frequency\s*:\s*annual|yearly", t):
        return "annual"
    return None


def _parse_first_date(text: str) -> Optional[datetime]:
    if not text:
        return None

    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d)
        except Exception:
            pass

    m2 = re.search(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(20\d{2})\b",
        text,
        flags=re.I,
    )
    if m2:
        mon_s, d_s, y_s = m2.groups()
        mon_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        mo = mon_map[mon_s[:3].lower()]
        y = int(y_s)
        d = int(d_s)
        try:
            return datetime(y, mo, d)
        except Exception:
            pass
    return None


def load_portal_metadata_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            di = rec.get("dataIdInfo", {}) if isinstance(rec, dict) else {}
            title = (((di.get("idCitation") or {}).get("resTitle") or {}).get("#text")) if isinstance(di, dict) else None
            if not title:
                continue

            id_abs = (di.get("idAbs") or {}).get("#text") if isinstance(di.get("idAbs"), dict) else ""
            id_purp = (di.get("idPurp") or {}).get("#text") if isinstance(di.get("idPurp"), dict) else ""
            keywords = []
            sk = di.get("searchKeys")
            if isinstance(sk, dict) and isinstance(sk.get("keyword"), list):
                for kw in sk["keyword"]:
                    if isinstance(kw, dict) and kw.get("#text"):
                        keywords.append(str(kw["#text"]))

            create_dt = None
            revise_dt = None
            for k in ["createDate", "reviseDate"]:
                v = di.get(k)
                if isinstance(v, dict) and v.get("#text"):
                    try:
                        s = str(v["#text"]).replace("Z", "")
                        if k == "createDate":
                            create_dt = datetime.fromisoformat(s)
                        else:
                            revise_dt = datetime.fromisoformat(s)
                    except Exception:
                        pass

            free_text = "\n".join([str(id_abs or ""), str(id_purp or "")])
            if create_dt is None:
                create_dt = _parse_first_date(free_text)
            last_updated = revise_dt or create_dt
            freq = _parse_update_frequency(free_text)
            desc_text = " ".join([str(id_abs or ""), str(id_purp or ""), " ".join(keywords)]).strip()

            out[_norm_name(title)] = {
                "title": title,
                "description_text": desc_text if desc_text else None,
                "update_frequency": freq,
                "last_updated": last_updated,
                "source": "portal",
            }
    return out


def match_portal_metadata(csv_filename: str, portal_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not portal_map:
        return None
    base = _norm_name(csv_filename)
    if base in portal_map:
        return portal_map[base]

    base_tokens = _token_set(csv_filename)
    best = None
    best_sim = 0.0
    for k, meta in portal_map.items():
        sim = _jaccard(base_tokens, _token_set(k))
        if sim > best_sim:
            best_sim = sim
            best = meta
    if best and best_sim >= 0.55:
        return best
    return None


def _strip_html(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    s = re.sub(r"<br\s*/?>", " ", str(text), flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&nbsp;", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _parse_iso_datetime(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None


def _distribution_access_urls(distributions: Any) -> List[str]:
    urls: List[str] = []
    if not isinstance(distributions, list):
        return urls
    for dist in distributions:
        if not isinstance(dist, dict):
            continue
        for k in ["accessURL", "downloadURL"]:
            u = dist.get(k)
            if u:
                urls.append(str(u))
    return urls


def _distribution_csv_url(distributions: Any) -> Optional[str]:
    if not isinstance(distributions, list):
        return None
    for dist in distributions:
        if not isinstance(dist, dict):
            continue
        fmt = str(dist.get("format") or "").strip().lower()
        media = str(dist.get("mediaType") or "").strip().lower()
        title = str(dist.get("title") or "").strip().lower()
        if fmt == "csv" or media == "text/csv" or title == "csv":
            return str(dist.get("accessURL") or dist.get("downloadURL") or "") or None
    return None


def load_catalog_metadata_json(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}

    datasets = obj.get("dataset", []) if isinstance(obj, dict) else []
    if not isinstance(datasets, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for rec in datasets:
        if not isinstance(rec, dict):
            continue

        title = rec.get("title")
        if not title:
            continue

        description = _strip_html(rec.get("description"))
        keywords = rec.get("keyword") if isinstance(rec.get("keyword"), list) else []
        keyword_text = " ".join([str(k).strip() for k in keywords if str(k).strip()])

        issued_dt = _parse_iso_datetime(rec.get("issued"))
        modified_dt = _parse_iso_datetime(rec.get("modified"))
        last_updated = modified_dt or issued_dt

        free_text = " ".join([str(description or ""), keyword_text, str(rec.get("landingPage") or "")]).strip()
        freq = _parse_update_frequency(free_text)

        out[_norm_name(str(title))] = {
            "title": str(title),
            "description_text": " ".join([x for x in [description, keyword_text] if x]).strip() or None,
            "description": description,
            "update_frequency": freq,
            "last_updated": last_updated,
            "issued": issued_dt,
            "modified": modified_dt,
            "landing_page": rec.get("landingPage"),
            "csv_url": _distribution_csv_url(rec.get("distribution")),
            "distribution_urls": _distribution_access_urls(rec.get("distribution")),
            "source": "catalog",
        }
    return out


def merge_metadata_sources(*meta_dicts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    source_chain: List[str] = []
    for meta in meta_dicts:
        if not meta:
            continue
        src = meta.get("source")
        if src:
            source_chain.append(str(src))
        for k, v in meta.items():
            if v is not None:
                out[k] = v
    if source_chain:
        out["source_chain"] = " > ".join(source_chain)
    return out


# =========================
# Cleaning
# =========================
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


# =========================
# Scoring
# =========================
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
        yrs = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int)
        if len(yrs) >= 2:
            expected = set(range(int(yrs.min()), int(yrs.max()) + 1))
            missing_years = sorted(expected - set(yrs))
            if missing_years:
                score -= 25.0
                issues_list.append(issue("completeness", f"Missing entire year(s): {missing_years}.", "high"))

    return float(np.clip(score, 0, 100)), issues_list


# 与 v5 完全对齐
def score_freshness(df: pd.DataFrame, issues_list: List[Dict[str, str]], metadata: Optional[Dict[str, Any]] = None) -> Tuple[float, List[Dict[str, str]]]:
    now = datetime.now()

    if metadata:
        freq = metadata.get("update_frequency")
        last = metadata.get("last_updated")

        if freq == "one-off":
            return 100.0, issues_list

        if isinstance(last, datetime):
            age_days = (now - last).days
            allowed_lag = {"daily": 2, "weekly": 10, "monthly": 45, "annual": 400}.get(str(freq).lower() if freq else "annual", 400)

            if age_days <= allowed_lag:
                return 100.0, issues_list

            score = max(0.0, 100.0 - (age_days - allowed_lag) * 0.15)
            issues_list.append(issue("freshness", f"Dataset is {age_days} days old (expected {freq or 'annual'} refresh).", "high"))
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
        c = dt_cols[0]
        dt = df[c].dropna()
        if not dt.empty:
            latest = pd.to_datetime(dt).max()
            age_years = current_year - int(latest.year)
            if age_years <= STALE_YEARS:
                return 100.0, issues_list
            score = max(0.0, 100.0 - 15.0 * (age_years - STALE_YEARS))
            issues_list.append(issue("freshness", f"Latest date in '{c}' is {latest.date()} (age≈{age_years} years).", "medium"))
            return float(np.clip(score, 0, 100)), issues_list

    issues_list.append(issue("freshness", "No schedule metadata and no usable Year/date found; freshness uncertain.", "medium"))
    return 40.0, issues_list


def meaningful_name_ratio(columns) -> float:
    good = 0
    for c in columns:
        c0 = str(c).strip().lower()
        if c0.startswith("unnamed"):
            continue
        tokens = re.split(r"[^a-z0-9]+", c0)
        tokens = [t for t in tokens if t and t not in STOP_WORDS]
        if any(len(t) >= 3 for t in tokens):
            good += 1
    return good / max(1, len(columns))


def score_usability(df: pd.DataFrame, issues_list: List[Dict[str, str]]) -> Tuple[float, List[Dict[str, str]]]:
    score = 100.0

    mnr = meaningful_name_ratio(df.columns)
    if mnr < 0.8:
        issues_list.append(issue("usability", f"Some column names are not descriptive enough (meaningful ratio={mnr:.2f}).", "medium"))
        score -= 15

    const_cols = []
    for c in df.columns:
        if is_vestigial(c):
            continue
        if df[c].nunique(dropna=True) <= 1 and df[c].notna().sum() > 0:
            const_cols.append(c)
    if const_cols:
        issues_list.append(issue("usability", f"Constant-value columns may add little analytic value: {const_cols[:6]}{'...' if len(const_cols) > 6 else ''}", "low"))
        score -= min(10, 2 * len(const_cols))

    num_text_cols = [c for c in df.columns if is_comma_number_text(df[c])]
    if num_text_cols:
        issues_list.append(issue("usability", f"Numeric-like text columns require preprocessing: {num_text_cols[:6]}{'...' if len(num_text_cols) > 6 else ''}", "medium"))
        score -= 10

    obj_cols = [c for c in df.columns if df[c].dtype == object]

    risk_cols = []
    for c in obj_cols[:50]:
        s = df[c].dropna().astype(str).head(200)
        if s.empty:
            continue
        if s.str.fullmatch(r"\d{1,2}[-/]\d{1,2}").any():
            risk_cols.append(c)
    if risk_cols:
        issues_list.append(issue("usability", f"Excel auto-date conversion risk in fields: {risk_cols[:6]}{'...' if len(risk_cols) > 6 else ''}", "low"))
        score -= 5

    cr_cols = []
    for c in obj_cols[:50]:
        s = df[c].dropna().astype(str).head(200)
        if s.empty:
            continue
        if s.str.contains(r"\r|\n").any():
            cr_cols.append(c)
    if cr_cols:
        issues_list.append(issue("usability", f"Carriage returns/newlines found in text fields: {cr_cols[:6]}{'...' if len(cr_cols) > 6 else ''}", "low"))
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

    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        sample_lens = []
        for c in obj_cols[:10]:
            s = df[c].dropna().astype(str).head(200)
            if not s.empty:
                sample_lens.append(float(s.map(len).mean()))
        if sample_lens and float(np.mean(sample_lens)) > 200:
            issues_list.append(issue("accessibility", "Long free-text fields may limit some users/tools (large cells).", "low"))
            score -= 5

    return float(np.clip(score, 0, 100)), issues_list


# 与 v5 对齐
def score_metadata(csv_path: str, df: pd.DataFrame, issues_list: List[Dict[str, str]], metadata: Optional[Dict[str, Any]] = None) -> Tuple[float, List[Dict[str, str]]]:
    score = 100.0

    sidecars = find_sidecar_metadata_files(csv_path)

    portal_desc = None
    if metadata:
        portal_desc = (
            metadata.get("description_text")
            or metadata.get("text")
            or metadata.get("description")
            or metadata.get("dataIdInfo", {}).get("idAbs", {}).get("#text")
        )

    has_portal_desc = bool(portal_desc)

    if sidecars:
        issues_list.append(issue("metadata", f"Found sidecar metadata files: {[os.path.basename(x) for x in sidecars]}.", "low"))
    else:
        if not has_portal_desc:
            issues_list.append(issue("metadata", "No sidecar metadata file (e.g., data dictionary/README) found.", "medium"))
            score -= 30

    description_text = None
    if portal_desc:
        description_text = str(portal_desc)
    else:
        for p in sidecars:
            t = _read_text_sidecar(p)
            if t:
                description_text = t
                break

    if description_text:
        desc = description_text.lower()

        missing_desc = [c for c in df.columns if (str(c).lower() not in desc) and not is_vestigial(c)]
        if missing_desc:
            score -= min(40, 3 * len(missing_desc))
            issues_list.append(issue("metadata", f"{len(missing_desc)} fields not described in metadata/description.", "medium"))

        if len(desc.strip()) < 80:
            score -= 10
            issues_list.append(issue("metadata", "Metadata description is very short; may be insufficient.", "medium"))
    else:
        score -= 30
        issues_list.append(issue("metadata", "No description text found (portal description not available locally).", "high"))

    unnamed = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    if unnamed:
        issues_list.append(issue("metadata", f"Unnamed columns detected: {unnamed[:4]}{'...' if len(unnamed) > 4 else ''}.", "high"))
        score -= 20

    return float(np.clip(score, 0, 100)), issues_list


# =========================
# Usage guidance
# =========================
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


# =========================
# Runtime metadata matching
# =========================
def load_runtime_metadata_maps() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    portal_map = load_portal_metadata_jsonl(METADATA_JSONL_PATH)
    catalog_map = load_catalog_metadata_json(METADATA_CATALOG_JSON_PATH)
    return portal_map, catalog_map


def build_metadata_for_uploaded_file(uploaded_path: str, portal_map: Dict[str, Dict[str, Any]], catalog_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    csv_name = Path(uploaded_path).name
    matched_portal = match_portal_metadata(csv_name, portal_map)
    matched_catalog = match_portal_metadata(csv_name, catalog_map)
    sidecar_meta = infer_metadata_from_sidecars(find_sidecar_metadata_files(uploaded_path))
    merged = merge_metadata_sources(sidecar_meta, matched_portal, matched_catalog)
    merged["matched_filename"] = csv_name
    return merged


def run_assessment(uploaded_path: str, portal_map: Dict[str, Dict[str, Any]], catalog_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    metadata = build_metadata_for_uploaded_file(uploaded_path, portal_map, catalog_map)
    df = read_csv_flex(uploaded_path)
    issues_list: List[Dict[str, str]] = []
    df_clean, issues_list = clean_df(df, issues_list)

    usability, issues_list = score_usability(df_clean, issues_list)
    completeness, issues_list = score_completeness(df_clean, issues_list)
    freshness, issues_list = score_freshness(df_clean, issues_list, metadata=metadata or None)
    metadata_score, issues_list = score_metadata(uploaded_path, df_clean, issues_list, metadata=metadata or None)
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
    }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Open Ottawa DQS", layout="wide")
st.title("Open Ottawa DQS")
st.caption("Single-file assessment with local dual-metadata matching (JSONL + catalog JSON).")

uploaded = st.file_uploader("Upload a CSV or Excel export", type=["csv", "txt", "xlsx"])

if uploaded is None:
    st.info("Upload one dataset file to run the assessment.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
    tmp.write(uploaded.getbuffer())
    upload_path = tmp.name

try:
    portal_map, catalog_map = load_runtime_metadata_maps()
    with st.spinner("Matching local metadata sources and scoring dataset..."):
        result = run_assessment(upload_path, portal_map, catalog_map)
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
    st.subheader("Matched Metadata")
    meta_view = {
        "title": metadata.get("title"),
        "update_frequency": metadata.get("update_frequency"),
        "last_updated": metadata.get("last_updated").isoformat() if isinstance(metadata.get("last_updated"), datetime) else None,
        "source": metadata.get("source"),
        "source_chain": metadata.get("source_chain"),
        "landing_page": metadata.get("landing_page"),
        "csv_url": metadata.get("csv_url"),
        "matched_filename": metadata.get("matched_filename"),
    }
    st.json(meta_view)
    if metadata.get("description_text"):
        st.markdown("**Description excerpt**")
        st.write(str(metadata["description_text"])[:1200])

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