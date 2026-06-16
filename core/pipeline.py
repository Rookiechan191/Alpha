"""
Core QA pipeline: retrieve -> classify & answer (single LLM call) -> return
structured result with verdict, answer, citations, cost, latency.

VERDICT TYPES:
  - "answerable"     : answer is supported by retrieved context, returned with citations
  - "unanswerable"   : question is on-topic but the answer is not in the docs
  - "false_premise"  : question contains an assumption contradicted by the docs

REFUSAL MECHANISM (two layers):

1. Pre-filter (cheap, no LLM call): if the top reranker score is below
   RERANK_SCORE_THRESHOLD, we have low confidence anything relevant was
   retrieved. We still pass the (weak) context to the LLM -- ms-marco
   cross-encoder scores can be negative even for genuinely relevant pairs,
   so a hard cutoff was tried and abandoned (see DECISIONS.md). We log the
   score for eval analysis instead.

2. LLM-level (primary mechanism): the model is instructed to:
   a) First check whether the question's premise is contradicted by the
      retrieved context -> false_premise
   b) Then check whether the retrieved context actually supports an answer
      (not just topically related) -> answerable vs unanswerable
   c) Only cite sections it actually used

KNOWN WEAK SPOT (documented further in DECISIONS.md):
False-premise questions about something NOT covered by retrieval at all.
If nothing relevant comes back, the model has no evidence to confirm OR
deny the premise, so it tends to default to "unanswerable" instead of
"false_premise" -- it can't distinguish "this is wrong" from "I have no
idea about this topic."
"""
import json
import os
import time

from anthropic import Anthropic

LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")

# Approximate per-token pricing for Claude Haiku (USD per token).
# Update these if you change LLM_MODEL / pricing changes.
PRICE_PER_INPUT_TOKEN = float(os.environ.get("PRICE_PER_INPUT_TOKEN", 1.0 / 1_000_000))
PRICE_PER_OUTPUT_TOKEN = float(os.environ.get("PRICE_PER_OUTPUT_TOKEN", 5.0 / 1_000_000))

SYSTEM_PROMPT = """You are TruthGate, a question-answering assistant over the official \
Kubernetes documentation. You will be given a user question and several retrieved \
document excerpts, each with a heading path identifying its section.

Follow these steps in order:

STEP 1 - Check the premise. Does the question assume something that is \
factually contradicted by the retrieved excerpts (e.g., it claims Kubernetes \
works in a way that the docs show it does not)? If yes, this is a \
"false_premise" question. Do not answer it; instead briefly explain what the \
incorrect assumption is and what the docs actually say, citing the relevant \
section(s).

STEP 2 - If the premise is fine, check whether the retrieved excerpts \
actually contain enough information to answer the question (not just \
topically related text). If the excerpts do not support a real answer, \
this is "unanswerable" -- respond that the answer is not in the docs. Do \
NOT use outside/training knowledge to fill the gap, even if you know the \
answer.

STEP 3 - If the excerpts do support an answer, this is "answerable". \
Provide a concise, accurate answer using ONLY the retrieved excerpts, and \
list the heading_path values of the excerpts you actually relied on as \
citations.

Respond with ONLY a JSON object, no other text, in this exact format:
{
  "verdict": "answerable" | "unanswerable" | "false_premise",
  "answer": "<your answer, or explanation for false_premise/unanswerable>",
  "citations": ["<heading_path>", ...]
}
"""


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks):
        parts.append(
            f"[Excerpt {i+1}] (heading_path: {c['heading_path']}, file: {c['file_path']})\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


class TruthGate:
    def __init__(self, retriever):
        self.retriever = retriever
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env

    def answer(self, question: str) -> dict:
        t0 = time.time()

        chunks = self.retriever.retrieve(question)
        retrieval_time = time.time() - t0

        if not chunks:
            return {
                "question": question,
                "verdict": "unanswerable",
                "answer": "No relevant documentation was retrieved for this question.",
                "citations": [],
                "retrieved_chunks": [],
                "cost_usd": 0.0,
                "latency_s": round(time.time() - t0, 3),
            }

        context = build_context(chunks)
        user_msg = f"Question: {question}\n\nRetrieved excerpts:\n\n{context}"

        t1 = time.time()
        response = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        llm_time = time.time() - t1

        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            parsed = json.loads(cleaned)
            verdict = parsed.get("verdict", "unanswerable")
            answer = parsed.get("answer", "")
            citations = parsed.get("citations", [])
        except Exception:
            verdict = "unanswerable"
            answer = f"[PARSE ERROR] raw model output: {raw_text[:300]}"
            citations = []

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = input_tokens * PRICE_PER_INPUT_TOKEN + output_tokens * PRICE_PER_OUTPUT_TOKEN

        return {
            "question": question,
            "verdict": verdict,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": [
                {
                    "heading_path": c["heading_path"],
                    "file_path": c["file_path"],
                    "url": c["url"],
                    "retrieval_score": c["retrieval_score"],
                    "rerank_score": c["rerank_score"],
                }
                for c in chunks
            ],
            "cost_usd": round(cost, 6),
            "latency_s": round(time.time() - t0, 3),
            "retrieval_time_s": round(retrieval_time, 3),
            "llm_time_s": round(llm_time, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
