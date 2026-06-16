"""
Chunks Kubernetes docs markdown into citation-friendly sections.

CHUNKING STRATEGY: heading-based (H1/H2/H3 sections), not fixed token windows.

Why:
- Each chunk maps to a real, citable doc section (file path + heading path)
- Preserves semantic coherence -- a section is usually about one concept
- Long sections (>MAX_TOKENS) get split on paragraph boundaries with overlap

Known failure mode (documented in DECISIONS.md):
- Reference pages that are mostly tables (e.g. kubectl flag references) often
  have a single heading followed by one giant table. The table gets chunked
  with little surrounding prose, so retrieval on "what does --flag X do"
  style questions can miss the right chunk because the embedding of a raw
  markdown table row is low-signal.

We tried (v1, abandoned): fixed 512-token sliding window chunks. Splits cut
mid-explanation constantly, citations pointed to "paragraph 3 of some page"
which is useless to a user, and retrieval quality was worse because chunk
boundaries didn't align with concept boundaries.
"""
import json
import re
from pathlib import Path

import tiktoken

MAX_TOKENS = 800
OVERLAP_TOKENS = 100

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def split_into_sections(md_text: str):
    """Split markdown into sections by H1/H2/H3 headings.
    Returns list of dicts: {heading_path, text}
    """
    lines = md_text.splitlines()
    # Strip front matter (--- ... ---)
    if lines and lines[0].strip() == "---":
        end = 1
        while end < len(lines) and lines[end].strip() != "---":
            end += 1
        lines = lines[end + 1:]

    sections = []
    current_heading_stack = []  # list of (level, text)
    current_lines = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text and count_tokens(text) > 20:  # skip near-empty sections
            heading_path = " > ".join(h for _, h in current_heading_stack)
            sections.append({"heading_path": heading_path or "Introduction", "text": text})

    heading_re = re.compile(r"^(#{1,3})\s+(.*)$")
    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            current_heading_stack = [h for h in current_heading_stack if h[0] < level]
            current_heading_stack.append((level, title))
        else:
            current_lines.append(line)
    flush()

    return sections


def split_long_section(text: str):
    """Split an oversized section into overlapping chunks on paragraph
    boundaries, respecting MAX_TOKENS."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        ptoks = count_tokens(para)
        if current_tokens + ptoks > MAX_TOKENS and current:
            chunks.append("\n\n".join(current))
            overlap = []
            otoks = 0
            for p in reversed(current):
                t = count_tokens(p)
                if otoks + t > OVERLAP_TOKENS:
                    break
                overlap.insert(0, p)
                otoks += t
            current = overlap
            current_tokens = otoks
        current.append(para)
        current_tokens += ptoks

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def file_path_to_url(rel_path: str) -> str:
    """Map local relative path back to a kubernetes.io docs URL for citation."""
    url_path = rel_path.replace(".md", "/").replace("_index/", "")
    return f"https://kubernetes.io/docs/{url_path}"


def chunk_file(file_path: Path, root: Path):
    text = file_path.read_text(encoding="utf-8")
    rel_path = str(file_path.relative_to(root))
    sections = split_into_sections(text)

    chunks = []
    for sec in sections:
        if count_tokens(sec["text"]) <= MAX_TOKENS:
            pieces = [sec["text"]]
        else:
            pieces = split_long_section(sec["text"])

        for i, piece in enumerate(pieces):
            suffix = f" (part {i+1})" if len(pieces) > 1 else ""
            chunks.append({
                "file_path": rel_path,
                "heading_path": sec["heading_path"] + suffix,
                "text": piece,
                "url": file_path_to_url(rel_path),
                "n_tokens": count_tokens(piece),
            })
    return chunks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_dir", default="data/raw")
    parser.add_argument("--out", default="data/chunks.jsonl")
    args = parser.parse_args()

    root = Path(args.input_dir)
    files = list(root.rglob("*.md"))
    print(f"Chunking {len(files)} files...")

    all_chunks = []
    for f in files:
        try:
            all_chunks.extend(chunk_file(f, root))
        except Exception as e:
            print(f"  WARN: failed on {f}: {e}")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {out_path}")
    if all_chunks:
        avg = sum(c["n_tokens"] for c in all_chunks) / len(all_chunks)
        print(f"Avg tokens/chunk: {avg:.1f}")


if __name__ == "__main__":
    main()
