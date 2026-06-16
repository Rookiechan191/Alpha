"""
Builds a FAISS flat index over chunk embeddings, using a local
sentence-transformer model (no API cost).

WHY FAISS flat (IndexFlatIP) instead of HNSW/IVF:
- Corpus is ~2-4k chunks -- exact search over that is fast (<50ms) on CPU.
- Flat index = exact nearest-neighbor, no recall/accuracy tradeoff to tune
  or explain. For a corpus this size, approximate indexing buys us nothing.
- We use inner product (IP) on L2-normalized vectors == cosine similarity.

WHY bge-small-en-v1.5:
- 384-dim, ~130MB, runs fast on CPU, strong performance on retrieval
  benchmarks (MTEB) relative to its size. Good fit for a local-only,
  budget-constrained pipeline.

Output:
- data/index/index.faiss   -- the FAISS index
- data/index/chunks.json   -- chunk metadata in the same order as index ids
"""
import json
import os
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--out", default="data/index")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks. Loading embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = []
    for c in chunks:
        # bge models recommend no special prefix for documents (only for queries)
        prefix = f"{c['heading_path']}\n\n" if c.get("heading_path") else ""
        texts.append(prefix + c["text"])

    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print(f"Wrote index with {index.ntotal} vectors (dim={dim}) to {out_dir}/")


if __name__ == "__main__":
    main()
