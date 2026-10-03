"""Fail fast if installed OASIS is not sourced from the recorded commit."""

import json
from importlib.metadata import distribution
from pathlib import Path

root = Path(__file__).resolve().parents[1]
expected = (root / "OASIS_COMMIT").read_text(encoding="utf-8").strip()
dist = distribution("camel-oasis")
direct_url_entry = next(
    (entry for entry in (dist.files or []) if entry.name == "direct_url.json"),
    None,
)
if direct_url_entry is None:
    raise SystemExit("camel-oasis has no direct_url.json; fixed commit cannot be verified")
payload = json.loads(dist.locate_file(direct_url_entry).read_text(encoding="utf-8"))
actual = payload.get("vcs_info", {}).get("commit_id")
if actual != expected:
    raise SystemExit(f"OASIS commit mismatch: expected {expected}, found {actual}")
print(f"OASIS commit verified: {actual}")
