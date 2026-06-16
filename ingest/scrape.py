"""
Scrapes Kubernetes documentation markdown files from the official
kubernetes/website GitHub repo (content/en/docs/).

Why GitHub raw markdown instead of rendered HTML?
- No nav/footer/ad junk to strip
- Headings, code blocks, tables come pre-structured
- File path + heading give us a natural citation unit

Usage:
    python ingest/scrape.py --limit 250
"""
import argparse
import os
import time
import requests
from pathlib import Path

GITHUB_API = "https://api.github.com/repos/kubernetes/website/git/trees/main?recursive=1"
RAW_BASE = "https://raw.githubusercontent.com/kubernetes/website/main"

# Restrict to the most useful, high-signal doc sections.
# (Excludes blog, case-studies, training, contribute -- noisy / low QA value)
INCLUDE_PREFIXES = (
    "content/en/docs/concepts/",
    "content/en/docs/tasks/",
    "content/en/docs/tutorials/",
    "content/en/docs/reference/",
    "content/en/docs/setup/",
)

EXCLUDE_SUFFIXES = ("_index.md",)  # mostly landing pages with little content


def list_doc_files():
    print("Fetching repo tree from GitHub API...")
    resp = requests.get(GITHUB_API, timeout=60)
    resp.raise_for_status()
    tree = resp.json()["tree"]

    files = []
    for entry in tree:
        path = entry["path"]
        if entry["type"] != "blob":
            continue
        if not path.endswith(".md"):
            continue
        if not path.startswith(INCLUDE_PREFIXES):
            continue
        if path.endswith(EXCLUDE_SUFFIXES):
            continue
        files.append(path)
    print(f"Found {len(files)} candidate markdown files.")
    return files


def download(path: str, out_dir: Path, session: requests.Session):
    url = f"{RAW_BASE}/{path}"
    rel = path.replace("content/en/docs/", "")
    out_path = out_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return out_path

    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  WARN: failed {path} ({resp.status_code})")
        return None
    out_path.write_text(resp.text, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--limit", type=int, default=300,
                         help="Max number of files to download")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_doc_files()
    files = files[: args.limit]

    session = requests.Session()
    downloaded = 0
    for i, path in enumerate(files):
        result = download(path, out_dir, session)
        if result:
            downloaded += 1
        if i % 20 == 0:
            print(f"  [{i+1}/{len(files)}] downloaded so far: {downloaded}")
        time.sleep(0.05)  # be polite to raw.githubusercontent.com

    print(f"\nDone. {downloaded} files saved to {out_dir}/")


if __name__ == "__main__":
    main()
