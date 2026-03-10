import json
import os
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OOTTAWA = "https://open.ottawa.ca"
AGOL = "https://www.arcgis.com"

TIMEOUT = 60
MAX_RETRIES = 6
SLEEP_SECONDS = 0.25


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "open-ottawa-metadata-fetcher/1.0",
            "Accept": "application/json,text/html,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "identity",
            "Connection": "keep-alive",
        }
    )

    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


session = build_session()


def safe_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z\\-_.() \\[\\]]+", "_", s).strip()[:180]


def looks_like_item_id(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", s or "", re.I))


def get_with_manual_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = TIMEOUT,
    max_attempts: int = MAX_RETRIES,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            return r
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            last_exc = e
            sleep_s = min(8.0, 0.6 * (2 ** (attempt - 1)))
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


def normalize_dataset_url(dataset_url: str) -> str:
    """
    Accept both:
    - https://open.ottawa.ca/datasets/<slug>/about
    - https://open.ottawa.ca/datasets/<slug>
    and normalize to .../about
    """
    dataset_url = dataset_url.strip()
    parsed = urlparse(dataset_url)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid Open Ottawa dataset URL.")

    if "open.ottawa.ca" not in parsed.netloc.lower():
        raise ValueError("The URL must be an Open Ottawa dataset URL.")

    path = parsed.path.rstrip("/")
    if "/datasets/" not in path:
        raise ValueError("The URL does not look like an Open Ottawa dataset page.")

    if not path.endswith("/about"):
        path = f"{path}/about"

    return f"{parsed.scheme}://{parsed.netloc}{path}"


def extract_itemid_from_html(html: str) -> Optional[str]:
    """
    Try multiple patterns from the dataset about page HTML.
    """
    # 1) __NEXT_DATA__
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    if m:
        try:
            data = json.loads(m.group(1))
            blob = json.dumps(data, ensure_ascii=False)

            m2 = re.search(r'"itemId"\s*:\s*"([0-9a-f]{32})"', blob, re.I)
            if m2:
                return m2.group(1)

            m3 = re.search(
                r'arcgis\.com/home/item\.html\?id=([0-9a-f]{32})',
                blob,
                re.I,
            )
            if m3:
                return m3.group(1)

            m4 = re.search(
                r'/sharing/rest/content/items/([0-9a-f]{32})',
                blob,
                re.I,
            )
            if m4:
                return m4.group(1)
        except Exception:
            pass

    # 2) fallback: raw HTML search
    patterns = [
        r'"itemId"\s*:\s*"([0-9a-f]{32})"',
        r'arcgis\.com/home/item\.html\?id=([0-9a-f]{32})',
        r'/sharing/rest/content/items/([0-9a-f]{32})',
        r'"portalItemId"\s*:\s*"([0-9a-f]{32})"',
    ]
    for patt in patterns:
        m2 = re.search(patt, html, re.I)
        if m2:
            return m2.group(1)

    return None


def resolve_arcgis_item_id_from_dataset_url(dataset_url: str) -> str:
    """
    Robustly resolve ArcGIS item id from an Open Ottawa dataset URL,
    even when the URL contains only a dataset slug/name.
    """
    # 1) direct item id in URL, if present
    m = re.search(r"\b([0-9a-f]{32})\b", dataset_url, re.I)
    if m:
        return m.group(1)

    # 2) fetch the /about page and parse item id from page content
    about_url = normalize_dataset_url(dataset_url)
    r = get_with_manual_retry(about_url, timeout=TIMEOUT)
    r.raise_for_status()

    item_id = extract_itemid_from_html(r.text)
    if item_id:
        return item_id

    raise ValueError(
        "Could not resolve an ArcGIS item ID from the provided Open Ottawa dataset URL. "
        "The page was reachable, but no ArcGIS item reference was found in the dataset page HTML."
    )


def download_metadata_xml(item_id: str) -> bytes:
    url = f"{AGOL}/sharing/rest/content/items/{item_id}/info/metadata/metadata.xml"
    r = get_with_manual_retry(url, timeout=TIMEOUT)

    if r.status_code == 404:
        raise ValueError(f"No metadata.xml was found for ArcGIS item {item_id}.")

    r.raise_for_status()

    ct = (r.headers.get("Content-Type") or "").lower()
    if "xml" not in ct and not r.text.lstrip().startswith("<?xml"):
        raise ValueError(f"metadata.xml response is not valid XML for item {item_id}.")

    return r.content


def fetch_metadata_xml_from_dataset_url(dataset_url: str, out_dir: str) -> Dict[str, Any]:
    """
    Main runtime entry used by web_v1.py
    """
    os.makedirs(out_dir, exist_ok=True)

    about_url = normalize_dataset_url(dataset_url)
    item_id = resolve_arcgis_item_id_from_dataset_url(about_url)
    xml_bytes = download_metadata_xml(item_id)

    fname = safe_filename(f"metadata__{item_id}.xml")
    xml_path = os.path.join(out_dir, fname)
    with open(xml_path, "wb") as f:
        f.write(xml_bytes)

    return {
        "dataset_url": dataset_url,
        "about_url": about_url,
        "item_id": item_id,
        "xml_path": xml_path,
    }