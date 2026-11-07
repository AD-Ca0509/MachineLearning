"""Offline source-copy integrity and Python syntax checks; does not train."""
from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for item in manifest["files"]:
    path = root / item["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise SystemExit("Hash mismatch: " + item["file"])
scripts = sorted(root.rglob("*.py"))
for path in scripts:
    ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
print(f"Verified {len(manifest['files'])} copied files and {len(scripts)} Python scripts.")
