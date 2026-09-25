import json
from pathlib import Path

data = json.loads(Path("cache/zenodo_4010759.json").read_text(encoding="utf-8"))
prefixes = ("Greetings", "Pronouns", "People", "Jobs", "Places", "Days_and_Time")
for f in data["files"]:
    key = f["key"]
    if not key.endswith(".zip"):
        continue
    if not key.startswith(prefixes):
        continue
    p = Path("raw_datasets/INCLUDE") / key
    sz = p.stat().st_size if p.exists() else -1
    status = "OK" if sz == f["size"] else f"HAVE {sz}"
    print(f"{key}: expected {f['size']} {status}")
