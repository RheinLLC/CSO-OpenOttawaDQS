import html as ihtml
import json
import re
from pathlib import Path
from typing import Any, Dict

IN_JSONL = "dataIdInfo_only.jsonl"
OUT_JSONL = "dataIdInfo_only_clean.jsonl"
TAG_RE = re.compile(r"<[^>]+>")
MAILTO_RE = re.compile(r"mailto:([^'\" >]+)", re.I)


def html_to_text(s: str) -> str:
    if not s:
        return ""
    s = ihtml.unescape(s)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|div|li|ul|ol|h1|h2|h3|h4|h5|h6)>", "\n", s, flags=re.I)
    mailtos = MAILTO_RE.findall(s)
    s = TAG_RE.sub("", s)
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if mailtos:
        uniq = []
        for email in mailtos:
            if email not in uniq:
                uniq.append(email)
        if not any(email in s for email in uniq):
            s = (s + "\n\nContact: " + ", ".join(uniq)).strip()
    return s


def clean_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    data_id_info = rec.get("dataIdInfo", {})
    id_abs = data_id_info.get("idAbs") or {}
    if isinstance(id_abs, dict) and isinstance(id_abs.get("#text"), str):
        id_abs["#text"] = html_to_text(id_abs["#text"])
        data_id_info["idAbs"] = id_abs
    rec["dataIdInfo"] = data_id_info
    return rec


def clean_jsonl(in_jsonl: str = IN_JSONL, out_jsonl: str = OUT_JSONL) -> Dict[str, Any]:
    read_count = 0
    written_count = 0
    cleaned_count = 0

    in_path = Path(in_jsonl)
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            read_count += 1
            record = json.loads(line)
            before = (((record.get("dataIdInfo", {}) or {}).get("idAbs", {}) or {}).get("#text"))
            record = clean_record(record)
            after = (((record.get("dataIdInfo", {}) or {}).get("idAbs", {}) or {}).get("#text"))
            if isinstance(before, str) and isinstance(after, str) and before != after:
                cleaned_count += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_count += 1

    return {
        "read": read_count,
        "written": written_count,
        "cleaned_idAbs": cleaned_count,
        "out": str(out_path),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean HTML-like content inside extracted metadata JSONL.")
    parser.add_argument("--in-jsonl", default=IN_JSONL)
    parser.add_argument("--out-jsonl", default=OUT_JSONL)
    args = parser.parse_args()

    print(clean_jsonl(args.in_jsonl, args.out_jsonl))


if __name__ == "__main__":
    main()
