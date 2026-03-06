#!/usr/bin/env python3
from pathlib import Path
import json

extensions = ["*.txt", "*.md", "*.rs", "*.py", "*.lean", "*.nix", "*.toml", "*.el", "*.sh"]
files = []

print("📂 Scanning spool directories...")
for path in [Path("/mnt/data1/spool"), Path.home() / "DOCS"]:
    if path.exists():
        print(f"  Scanning: {path}")
        for ext in extensions:
            found = list(path.rglob(ext))
            print(f"    {ext}: {len(found)} files")
            files.extend([str(f) for f in found])

print(f"\n✅ Total files: {len(files)}")
print(f"💾 Saving to spool_filelist.json...")

with open("spool_filelist.json", "w") as f:
    json.dump(files, f)

print("Done!")
