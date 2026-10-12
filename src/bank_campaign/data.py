from pathlib import Path
from zipfile import ZipFile
import urllib.request

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ARCHIVE = "bank+marketing.zip"
def ensure_data(root: str) -> Path:
    root = Path(root); target = root / "bank-additional-full.csv"
    if target.exists(): return target
    root.mkdir(parents=True, exist_ok=True); archive = root / ARCHIVE
    if not archive.exists():
        temporary = root / (ARCHIVE + ".partial")
        try: urllib.request.urlretrieve(URL, temporary); temporary.replace(archive)
        finally: temporary.unlink(missing_ok=True)
    with ZipFile(archive) as zipped:
        candidates = [n for n in zipped.namelist() if n.endswith("bank-additional-full.csv") and not n.startswith("/") and ".." not in Path(n).parts]
        if len(candidates) != 1: raise RuntimeError("Expected one bank-additional-full.csv in UCI archive")
        target.write_bytes(zipped.read(candidates[0]))
    return target
