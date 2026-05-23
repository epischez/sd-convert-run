#!/usr/bin/env python3
"""Download a HuggingFace repo snapshot via aria2c (per project policy).

Single-stream `huggingface_hub.snapshot_download` and `hf_transfer` both hit
HF's xet CAS rate limits / route weirdness on multi-GB checkpoints. aria2c
with multi-connection per file is the established reliable path.

Usage:
    python download_via_aria2.py REPO DEST [--ignore PATTERN]... [--revision REV]
"""
import argparse
import fnmatch
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url

ap = argparse.ArgumentParser()
ap.add_argument("repo", help="HF repo id, e.g. jzli/epiCRealism-v3")
ap.add_argument("dest", help="Local destination directory")
ap.add_argument("--ignore", action="append", default=[], metavar="GLOB",
                help="Glob pattern to skip (repeatable)")
ap.add_argument("--revision", default="main")
args = ap.parse_args()

api = HfApi(token=os.environ.get("HF_TOKEN"))
info = api.model_info(args.repo, revision=args.revision)
all_files = [s.rfilename for s in info.siblings]

keep = [f for f in all_files if not any(fnmatch.fnmatch(f, p) for p in args.ignore)]
skipped = len(all_files) - len(keep)
print(f"📋 {args.repo}@{args.revision}: {len(keep)} files to fetch, {skipped} ignored", flush=True)
if not keep:
    sys.exit("❌ Nothing to download after applying --ignore patterns")

dest = Path(args.dest).resolve()
dest.mkdir(parents=True, exist_ok=True)

# aria2c input file format: URL on one line, indented "  out=" directives below.
# https://aria2.github.io/manual/en/html/aria2c.html#input-file
input_lines = []
for f in keep:
    url = hf_hub_url(args.repo, f, revision=args.revision)
    input_lines.append(url)
    input_lines.append(f"  out={f}")
input_file = dest / ".aria2_input.txt"
input_file.write_text("\n".join(input_lines) + "\n")

cmd = [
    "aria2c",
    "-x", "16", "-s", "16", "-k", "1M",
    "--auto-file-renaming=false",
    "--conditional-get=true",
    "--continue=true",
    "--summary-interval=10",
    "--console-log-level=warn",
    "--dir", str(dest),
    "-i", str(input_file),
]
token = os.environ.get("HF_TOKEN")
if token:
    cmd.append(f"--header=Authorization: Bearer {token}")

print(f"🚀 {' '.join(cmd[:8])} ... (input has {len(keep)} URLs)", flush=True)
subprocess.run(cmd, check=True)
input_file.unlink(missing_ok=True)
print(f"✅ {args.repo} → {dest}", flush=True)
