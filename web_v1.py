import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype

# -------------------------
# DQS-style weights (Toronto-inspired, local-computable)
# -------------------------
WEIGHTS = {
    "freshness": 0.35,
    "metadata": 0.35,  # local proxy (not portal metadata)
    "accessibility": 0.15,
    "completeness": 0.10,
    "usability": 0.05,
}

SEVERE_MISSING_COL = 0.50
STALE_YEARS = 2
STOP_WORDS = {"id", "uid", "key", "value", "val", "x", "y", "col", "column", "unnamed"}

# -------------------------
# Helpers
# -------------------------
def standardize_columns(df: pd.DataFrame):
    orig = list(df.columns)
    new = []
    for c in orig:
        c2 = str(c).replace("\ufeff", "").strip()
        c2 = re.sub(r"\s+", " ", c2)
        new.append(c2)
    out = df.copy()
    out.columns = new
    changes = {o: n for o, n in zip(orig, new) if o != n}
    return out, changes


def is_comma_number_text(s: pd.Series) -> bool:
    if s.dtype != object:
        return False
    sample = s.dropna().astype(str).head(50)
    patt = r"^\s*-?\d{1,3}(,\d{3})+(\.\d+)?\s*$"
    return any(re.search(patt, x) for x in sample)


def to_numeric_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def is_date_like(s: pd.Series) -> bool:
    if s.dtype != object:
        return False
    sample = s.dropna().astype(str).head(30)
    pats = [r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"]
    return any(re.search(p, x) for p in pats for x in sample)


def parse_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def meaningful_name_ratio(columns):
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


def compute_total_score(scores: dict) -> float:
    return float(np.clip(sum(WEIGHTS[k] * scores[k] for k in WEIGHTS), 0, 100))


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


def grade_from_likert(likert: int) -> str:
    # Likert 1 -> C
    # Likert 2–3 -> B
    # Likert 4–5 -> A
    if likert >= 4:
        return "A"
    if likert >= 2:
        return "B"
    return "C"


# -------------------------
# Cleaning (reproducible, deterministic)
# -------------------------
def clean_df(df: pd.DataFrame):
    issues = []

    df, changes = standardize_columns(df)
    if changes:
        issues.append(("usability", "Standardized column names (trim/collapse spaces)."))

    # Convert comma-number text to numeric
    for c in df.columns:
        if is_comma_number_text(df[c]):
            df[c] = to_numeric_clean(df[c])
            issues.append(("usability", f"Converted comma-number text to numeric in '{c}'."))

    # Parse date-like columns
    for c in df.columns:
        if is_date_like(df[c]):
            dt = parse_datetime_safe(df[c])
            non_null = df[c].notna().sum()
            ok = dt.notna().sum()
            if non_null > 0 and ok / non_null >= 0.60:
                df[c] = dt
                issues.append(("usability", f"Parsed datetime in '{c}'."))

    # Add Year if possible
    if "Year" not in df.columns:
        dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df["Year"] = df[dt_cols[0]].dt.year.astype("Int64")
            issues.append(("usability", f"Derived Year from '{dt_cols[0]}'."))

    return df, issues


# -------------------------
# Scoring (dimension scores 0..100)
# -------------------------
def score_dimensions(df: pd.DataFrame):
    issues = []

    # Completeness (missingness + severe columns)
    miss_by_col = df.isna().mean().fillna(0.0)
    avg_missing = float(miss_by_col.mean()) if len(miss_by_col) else 0.0
    severe_cols = miss_by_col[miss_by_col >= SEVERE_MISSING_COL]
    severe_col_ratio = float(len(severe_cols) / max(1, df.shape[1]))
    completeness = 100.0 * (1 - avg_missing) - 40.0 * severe_col_ratio
    completeness = float(np.clip(completeness, 0, 100))
    if len(severe_cols) > 0:
        issues.append(("completeness", f"{len(severe_cols)} columns have >=50% missing."))

    # Freshness
    current_year = datetime.now().year
    freshness = 60.0
    if "Year" in df.columns:
        years = pd.to_numeric(df["Year"], errors="coerce").dropna()
        if not years.empty:
            ymax = int(years.max())
            age = current_year - ymax
            if age <= STALE_YEARS:
                freshness = 100.0
            else:
                freshness = max(0.0, 100.0 - 15.0 * (age - STALE_YEARS))
                issues.append(("freshness", f"Latest year {ymax} is {age} years behind {current_year}."))
    else:
        dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
        if dt_cols:
            latest = pd.to_datetime(df[dt_cols[0]].dropna()).max()
            age = current_year - int(latest.year)
            if age <= STALE_YEARS:
                freshness = 100.0
            else:
                freshness = max(0.0, 100.0 - 15.0 * (age - STALE_YEARS))
                issues.append(("freshness", f"Latest date {latest.date()} in '{dt_cols[0]}' is old."))

    # Usability (names + constant columns)
    usability = 100.0
    mratio = meaningful_name_ratio(df.columns)
    if mratio < 0.20:
        usability -= 50
        issues.append(("usability", f"Low meaningful column names ratio ({mratio:.0%})."))
    elif mratio < 0.50:
        usability -= 20
        issues.append(("usability", f"Moderate meaningful column names ratio ({mratio:.0%})."))

    const_cols = []
    for c in df.columns:
        nn = df[c].dropna()
        if len(nn) == 0 or nn.nunique() == 1:
            const_cols.append(c)
    if const_cols:
        usability -= min(30, 5 * len(const_cols))
        issues.append(("usability", f"{len(const_cols)} constant/empty columns."))

    usability = float(np.clip(usability, 0, 100))

    # Accessibility (local: readable + not huge/wide)
    accessibility = 100.0
    if df.shape[1] > 80:
        accessibility -= 15
        issues.append(("accessibility", f"Very wide table ({df.shape[1]} columns)."))
    accessibility = float(np.clip(accessibility, 0, 100))

    # Metadata (local proxy)
    metadata = 100.0
    meta_like = [
        c
        for c in df.columns
        if re.search(r"source|note|definition|desc|description|unit|method|metadata", str(c), re.I)
    ]
    if not meta_like:
        metadata -= 50
        issues.append(("metadata", "No descriptive/definition/unit columns detected (local proxy)."))
    else:
        metadata -= 30  # still penalize because portal metadata not available locally
    metadata = float(np.clip(metadata, 0, 100))

    scores = {
        "usability": usability,
        "metadata": metadata,
        "freshness": float(freshness),
        "completeness": completeness,
        "accessibility": accessibility,
    }
    total = compute_total_score(scores)
    return scores, total, issues


# -------------------------
# Streamlit UI (Assessment-only; controls removed)
# -------------------------
st.set_page_config(page_title="Open Data DQS", layout="wide")
st.title("Open Data DQS")

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to score it and view detected quality issues.")
    st.stop()

df_raw = pd.read_csv(uploaded)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

with col2:
    st.subheader("Quality Score)")
    df_clean_base, clean_notes = clean_df(df_raw.copy())
    scores_raw, total_raw, issues_raw = score_dimensions(df_clean_base)

    st.metric(
        "Total Score (0–100)",
        f"{total_raw:.2f}",
        help="Weighted sum of 5 DQS dimensions (Toronto-style, local proxy). Scoring is computed after deterministic cleaning.",
    )

    likert = likert_from_score(total_raw)
    st.write({"Likert Score (1–5)": likert, "Grade (A/B/C)": grade_from_likert(likert)})

    st.json({f"{k}_score": round(v, 2) for k, v in scores_raw.items()})

    if clean_notes:
        st.caption("Deterministic cleaning applied before scoring:")
        st.write([f"- {m}: {d}" for m, d in clean_notes])

    st.divider()
    st.subheader("DQS Issue Log")

    if issues_raw:
        # issues_raw: [(metric, detail), ...]
        issues_df = pd.DataFrame(issues_raw, columns=["metric", "detail"])

        # severity
        def _severity(metric: str) -> str:
            metric = (metric or "").lower()
            if metric in {"freshness", "completeness"}:
                return "medium"
            if metric in {"metadata"}:
                return "medium"
            if metric in {"usability", "accessibility"}:
                return "low"
            return "low"

        issues_df["severity"] = issues_df["metric"].map(_severity)

        st.dataframe(
            issues_df[["metric", "severity", "detail"]],
            use_container_width=True,
            hide_index=True,
        )

        # download Issue Log（CSV）
        st.download_button(
            "Download DQS Issue Log (CSV)",
            issues_df.to_csv(index=False).encode("utf-8"),
            file_name="dqs_issue_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No issues detected by current rule-based checks for this dataset.")

