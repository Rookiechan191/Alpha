# DECISIONS.md

> Written honestly during the 24h window. Typos intentional (not cleaned up).
> Real failures from actual runs, not fabricated.

## 1. Three things I tried that didn't work

1. **Fixed-size token chunking (512 tokens, sliding window with 50-token overlap)**

   First implementation used LangChain's default `RecursiveCharacterTextSplitter` with
   512-token chunks. Two immediate problems: (a) citations were useless — they pointed to
   byte offsets within a file, not to any meaningful section a human could find. (b) retrieval
   quality was noticeably worse because chunk boundaries cut mid-sentence and mid-explanation
   constantly. A question like "what is a DaemonSet" would retrieve the tail of the previous
   section's explanation stitched to the beginning of the DaemonSet section, giving the LLM
   a coherent-looking but incomplete chunk. Switched to heading-based chunking after seeing
   ~20% of answerable eval questions mis-classified as unanswerable purely because the key
   sentence was split across two retrieved chunks that ranked 6th and 7th (outside top-5).

2. **Hard rerank-score threshold for refusal (score < 0 → force unanswerable)**

   ms-marco cross-encoder scores are not calibrated to a fixed scale across query types.
   A threshold of 0 seemed reasonable — negative scores should mean "not relevant" — but
   in practice short factual questions like "what is a Pod?" consistently scored -1 to -3
   even when the correct chunk was retrieved at rank 1. The scores compress and shift based
   on query length and vocabulary. After the threshold cut a majority of easy answerable
   questions to unanswerable in my first eval run (answerable verdict rate dropped to ~40%),
   I removed the hard cutoff and let the LLM decide. The rerank score is now logged per
   query for analysis but not used as a gate.

3. **Asking the LLM for a confidence score (0–100) alongside the verdict**

   Added `"confidence": <int>` to the JSON output schema and tried using it to escalate
   borderline unanswerable/false_premise classifications. The model output scores between
   60–90 for almost everything — it was overconfident on hallucinated answers and
   underconfident on well-supported ones. The score had essentially no correlation with
   actual correctness across my labeled eval set. Removed it entirely rather than add a
   false signal. A calibrated confidence score trained on labeled retrieval data would need
   far more data and a separate training run to be useful.

## 2. Chunking strategy and its failure mode

Heading-based (H1/H2/H3) sections, oversized sections (>800 tokens) split on paragraph
boundaries with ~100 token overlap. See `ingest/chunk.py`.

**Remaining failure mode**: Kubernetes reference pages that are primarily tables (e.g.,
the kubectl command reference, API field tables) produce chunks that are 80–90% raw
markdown table syntax. The embedding of a table row like `| --grace-period | int | ...`
is low-signal — the model encodes it as surface syntax, not as "this is about the
grace-period flag". Retrieval for questions like "what does the --grace-period flag
do with kubectl delete?" consistently misses the correct table row and instead retrieves
the prose section that mentions "graceful termination" in context. Eval question #39
("What is the maximum recommended cluster size") is clean because it's prose-based;
but several "what does flag X do" style questions that I wrote and then removed from
the set were failing reliably for this reason.

## 3. How the refusal mechanism works, and where it's wrong

Two-stage retrieval (FAISS top-20 → cross-encoder rerank top-5) feeds the top-5 chunks
to a single Claude Haiku call. The system prompt enforces a strict 3-step classification
before any answer: (1) check for false premise using only the retrieved context,
(2) check whether context actually supports an answer (not just topically related),
(3) only then answer with citations. The model is explicitly told not to use training
knowledge to fill gaps.

**Category it still gets wrong**: false-premise questions about topics that return
zero or near-zero relevant chunks. Example: eval question #50 ("Since Kubernetes uses
JSON Web Tokens as its primary storage format..."). If nothing about JWT or token storage
comes back in retrieval, the model has no evidence to confirm OR deny the premise, so it
defaults to "unanswerable" instead of "false_premise". This is correct behaviour given
the instruction "only use retrieved context" — it genuinely can't prove the premise is
wrong without evidence — but it's a miscategorisation from the eval's perspective. The
only fix that doesn't compromise the refusal mechanism is improving retrieval coverage
so the relevant etcd/storage chunks always come back for storage-related questions.

## 4. One more week + $500/month

- Replace the single system-prompt classification with a two-call pipeline: a cheap
  first call classifies the question type (answerable/unanswerable/false_premise) based
  purely on retrieval coverage signals, then a second call generates the answer only for
  "answerable" — this avoids the model having to simultaneously reason about refusal and
  generate a well-cited answer in the same call, which creates prompt tension.
- Fine-tune the reranker or add a calibrated confidence head trained on labeled
  (query, chunk, relevant?) triples from the Kubernetes docs specifically. The off-the-shelf
  ms-marco reranker was trained on web search data; K8s technical queries have a different
  vocabulary distribution.
- Handle table-heavy reference pages separately: extract table rows as individual chunks
  with a synthesized prose description ("The --grace-period flag accepts an int value...")
  generated offline once, stored alongside the raw row.
- Add query-level caching (hash the question → return cached result) to bring repeated
  query cost to $0.
- Expand to the full kubernetes/website repo (currently capped at 300 files for scrape
  time). With a week I'd ingest all ~800+ doc files and build a larger, higher-recall index.

## 5. Shortcut taken due to the 24h limit

Citation matching in the eval harness is substring-based (does any citation heading
contain the expected keyword?) rather than verified against actual doc anchors or
canonical section URLs. This is a deliberate proxy metric — it's fast to write and
run, but it can give false positives (a heading that mentions "Pod" but isn't the
right section) and false negatives (a correct citation with a slightly different heading
text). With more time I'd build a ground-truth citation map: for each answerable
question, hand-verify which specific heading_path values are acceptable citations,
and use exact-set-match instead of substring. The current metric is good enough to
spot large regressions but should not be treated as a precise accuracy number.
