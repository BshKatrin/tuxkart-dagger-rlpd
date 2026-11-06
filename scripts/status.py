"""Read status once; no dashboards or background polling required."""

import argparse
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", nargs="?", default="runs/study-polar-v3")
args = parser.parse_args()
root = Path(args.root)
manifest = json.loads((root / "manifest.json").read_text())
print(f"Suite PID {manifest['pid']}: {manifest['state']}")
print(dict(Counter(job["state"] for job in manifest["jobs"])))
for job in manifest["jobs"]:
    if job["state"] in ["running", "failed"]:
        path = Path(job["output"]) / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {}
        print(job["name"], job["state"], "PID", job.get("pid"), status)
