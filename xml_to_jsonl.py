import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

IN_DIR = "open_ottawa_arcgis_metadata_xml"
OUT_JSONL = "dataIdInfo_only.jsonl"
SKIP_LIST = "skipped_parse_error.txt"

DATAID_RE = re.compile(r"<dataIdInfo\b[^>]*>.*?</dataIdInfo>", re.S)

# Support both old and new Esri date/time styles
# Date examples:
#   20220926
#   2022-09-26
# Time examples:
#   175152
#   17515200
#   17:51:52
#   17:51:52.00
CreaDate_RE = re.compile(r"<CreaDate>\s*([\d-]+)\s*</CreaDate>", re.I)
CreaTime_RE = re.compile(r"<CreaTime>\s*([\d:.]+)\s*</CreaTime>", re.I)
ModDate_RE = re.compile(r"<ModDate>\s*([\d-]+)\s*</ModDate>", re.I)
ModTime_RE = re.compile(r"<ModTime>\s*([\d:.]+)\s*</ModTime>", re.I)


def elem_to_obj(elem: ET.Element) -> Dict[str, Any]:
    """
    Convert XML subtree to a JSON-serializable dict:
      - attributes under "@attrs"
      - text under "#text"
      - repeated child tags become lists
    """
    obj: Dict[str, Any] = {}

    if elem.attrib:
        obj["@attrs"] = dict(elem.attrib)

    text = (elem.text or "").strip()
    if text:
        obj["#text"] = text

    children = list(elem)
    if children:
        grouped: Dict[str, Any] = {}
        for ch in children:
            tag = ch.tag
            ch_obj = elem_to_obj(ch)
            if tag in grouped:
                if isinstance(grouped[tag], list):
                    grouped[tag].append(ch_obj)
                else:
                    grouped[tag] = [grouped[tag], ch_obj]
            else:
                grouped[tag] = ch_obj
        obj.update(grouped)

    return obj


def _normalize_esri_date(date_raw: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """
    Support:
      - YYYYMMDD
      - YYYY-MM-DD
    Return (yyyy, mm, dd)
    """
    if not date_raw:
        return None

    d = date_raw.strip()

    if re.fullmatch(r"\d{8}", d):
        return d[0:4], d[4:6], d[6:8]

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        yyyy, mm, dd = d.split("-")
        return yyyy, mm, dd

    return None


def _normalize_esri_time(time_raw: Optional[str]) -> Tuple[str, str, str]:
    """
    Support:
      - HHMMSS
      - HHMMSS00
      - HH:MM:SS
      - HH:MM:SS.xx
    If invalid or missing, default to 00:00:00
    """
    if not time_raw:
        return "00", "00", "00"

    t = time_raw.strip()

    if re.fullmatch(r"\d{6}(\d{2})?", t):
        t6 = t[:6]
        return t6[0:2], t6[2:4], t6[4:6]

    m = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?", t)
    if m:
        return m.group(1), m.group(2), m.group(3)

    return "00", "00", "00"


def _fmt_esri_datetime(date_raw: Optional[str], time_raw: Optional[str]) -> Optional[str]:
    """
    Convert Esri CreaDate/CreaTime or ModDate/ModTime into:
      YYYY-MM-DDTHH:MM:SS

    Supported combinations:
      1) 20220926 + 17515200
      2) 20220926 + 175152
      3) 2022-09-26 + 17:51:52.00
      4) 2022-09-26 + 17:51:52
    """
    norm_date = _normalize_esri_date(date_raw)
    if not norm_date:
        return None

    yyyy, mm, dd = norm_date
    hh, mi, ss = _normalize_esri_time(time_raw)
    return f"{yyyy}-{mm}-{dd}T{hh}:{mi}:{ss}"


def extract_esri_datetimes_from_root(root: ET.Element) -> Tuple[Optional[str], Optional[str]]:
    """
    Read Esri/CreaDate,CreaTime and Esri/ModDate,ModTime from parsed XML root.
    Returns (create_iso, revise_iso).
    """
    esri = root.find("./Esri")
    if esri is None:
        return None, None

    crea_date = (esri.findtext("./CreaDate") or "").strip() or None
    crea_time = (esri.findtext("./CreaTime") or "").strip() or None
    mod_date = (esri.findtext("./ModDate") or "").strip() or None
    mod_time = (esri.findtext("./ModTime") or "").strip() or None

    create_iso = _fmt_esri_datetime(crea_date, crea_time)
    revise_iso = _fmt_esri_datetime(mod_date, mod_time)
    return create_iso, revise_iso


def extract_esri_datetimes_from_text(xml_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fallback for truncated or partially malformed XML:
    regex extract CreaDate/CreaTime/ModDate/ModTime from raw text.
    """
    crea_date_match = CreaDate_RE.search(xml_text)
    crea_time_match = CreaTime_RE.search(xml_text)
    mod_date_match = ModDate_RE.search(xml_text)
    mod_time_match = ModTime_RE.search(xml_text)

    crea_date = crea_date_match.group(1).strip() if crea_date_match else None
    crea_time = crea_time_match.group(1).strip() if crea_time_match else None
    mod_date = mod_date_match.group(1).strip() if mod_date_match else None
    mod_time = mod_time_match.group(1).strip() if mod_time_match else None

    return _fmt_esri_datetime(crea_date, crea_time), _fmt_esri_datetime(mod_date, mod_time)


def extract_dataidinfo_via_et(xml_path: str) -> Tuple[Optional[ET.Element], Optional[str], Optional[str]]:
    """
    Parse full XML with ElementTree.
    Returns (dataIdInfo_element, create_iso, revise_iso).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data_id = root.find("./dataIdInfo")
    create_iso, revise_iso = extract_esri_datetimes_from_root(root)
    return data_id, create_iso, revise_iso


def extract_dataidinfo_via_regex(xml_path: str) -> Tuple[Optional[ET.Element], Optional[str], Optional[str]]:
    """
    For truncated or malformed XML:
      - salvage dataIdInfo subtree via regex
      - salvage Esri dates via regex
    Returns (dataIdInfo_element, create_iso, revise_iso).
    """
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    m = DATAID_RE.search(txt)
    if not m:
        return None, None, None

    frag = m.group(0)
    wrapped = f"<root>{frag}</root>"
    root = ET.fromstring(wrapped)
    data_id = root.find("./dataIdInfo")

    create_iso, revise_iso = extract_esri_datetimes_from_text(txt)
    return data_id, create_iso, revise_iso


def ensure_date_fields(
    dataid_obj: Dict[str, Any],
    create_iso: Optional[str],
    revise_iso: Optional[str],
) -> Dict[str, Any]:
    """
    Ensure:
      dataIdInfo.idCitation.date.createDate
      dataIdInfo.idCitation.date.reviseDate

    Logic:
      - If idCitation does not exist, create it
      - If date does not exist or is not a dict, replace it with {}
      - If create_iso/revise_iso exists, overwrite those two fields
      - This also fixes cases where the original XML had date: {}
    """
    if not isinstance(dataid_obj, dict):
        return dataid_obj

    idc = dataid_obj.get("idCitation")
    if not isinstance(idc, dict):
        idc = {}
        dataid_obj["idCitation"] = idc

    date_obj = idc.get("date")
    if not isinstance(date_obj, dict):
        date_obj = {}
        idc["date"] = date_obj

    if create_iso:
        date_obj["createDate"] = {"#text": create_iso}

    if revise_iso:
        date_obj["reviseDate"] = {"#text": revise_iso}

    return dataid_obj


def convert_single_xml(xml_path: str) -> Optional[Dict[str, Any]]:
    """
    Convert one XML file into:
    {
      "xml_file": "<basename>.xml",
      "dataIdInfo": {...}
    }
    """
    base = os.path.basename(xml_path)

    data_id_elem: Optional[ET.Element] = None
    create_iso: Optional[str] = None
    revise_iso: Optional[str] = None

    try:
        data_id_elem, create_iso, revise_iso = extract_dataidinfo_via_et(xml_path)
        if data_id_elem is None:
            return None
    except ET.ParseError:
        try:
            data_id_elem, create_iso, revise_iso = extract_dataidinfo_via_regex(xml_path)
            if data_id_elem is None:
                return None
        except Exception:
            return None
    except Exception:
        return None

    dataid_obj = elem_to_obj(data_id_elem)
    dataid_obj = ensure_date_fields(dataid_obj, create_iso, revise_iso)

    return {
        "xml_file": base,
        "dataIdInfo": dataid_obj,
    }


def convert_xml_dir_to_jsonl(
    in_dir: str,
    out_jsonl: str,
    skip_list: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main callable function used by web_v1.py

    Parameters
    ----------
    in_dir : str
        Directory containing .xml files
    out_jsonl : str
        Output JSONL path
    skip_list : Optional[str]
        Optional path to save skipped filenames

    Returns
    -------
    dict
        Summary with written/skipped counts
    """
    files = sorted(glob.glob(os.path.join(in_dir, "*.xml")))
    if not files:
        raise FileNotFoundError(f"No .xml files found under: {in_dir}")

    written = 0
    skipped: List[str] = []

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for p in files:
            rec = convert_single_xml(p)
            if rec is None:
                skipped.append(os.path.basename(p))
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    if skip_list and skipped:
        with open(skip_list, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped))

    return {
        "written": written,
        "skipped": len(skipped),
        "out": out_jsonl,
        "skip_list": skip_list,
    }


def main() -> None:
    result = convert_xml_dir_to_jsonl(IN_DIR, OUT_JSONL, SKIP_LIST)
    print(result)


if __name__ == "__main__":
    main()