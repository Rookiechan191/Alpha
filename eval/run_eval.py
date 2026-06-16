"""
Eval harness for TruthGate.

Runs eval/questions.json (60 hand-written questions) through the pipeline
and reports:
  - Answer accuracy (answerable questions, judged by citation/keyword overlap
    -- NOT by asking an LLM "is this right?")
  - Refusal precision/recall (unanswerable detection)
  - False-premise detection rate
  - Mean cost per query, p95 latency

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --out eval/results.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from tabulate import tabulate

from core.pipeline import TruthGate
from retrieval.retriever import Retriever


def heading_match(citations, expected_contains):
    """Check if any citation heading contains any of the expected substrings.
    Case-insensitive substring match -- intentionally loose, since exact
    heading text matching is brittle. This is NOT a ground-truth accuracy
    measure on its own; combine with manual spot-checks of result['answer'].
    """
    if not expected_contains:
        return None  # no ground truth to check against
    for c in citations:
        for exp in expected_contains:
            if exp.lower() in c.lower():
                return True
    return False


def run_eval(questions_path, gate, out_path=None):
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        print(f"[{q['id']:>2}/{len(questions)}] ({q['category']}) {q['question'][:70]}")
        try:
            r = gate.answer(q["question"])
        except Exception as e:
            r = {
                "question": q["question"],
                "verdict": "ERROR",
                "answer": str(e),
                "citations": [],
                "cost_usd": 0.0,
                "latency_s": 0.0,
            }
        r["expected_category"] = q["category"]
        r["id"] = q["id"]
        r["citation_match"] = heading_match(r["citations"], q.get("expected_heading_contains", []))
        results.append(r)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def compute_metrics(results):
    metrics = {}

    unans = [r for r in results if r["expected_category"] == "unanswerable"]
    if unans:
        tp = sum(1 for r in unans if r["verdict"] == "unanswerable")
        metrics["unanswerable_recall"] = tp / len(unans)

    if results:
        predicted_unans = sum(1 for r in results if r["verdict"] == "unanswerable")
        tp_for_precision = sum(
            1 for r in results
            if r["verdict"] == "unanswerable" and r["expected_category"] == "unanswerable"
        )
        if predicted_unans:
            metrics["unanswerable_precision"] = tp_for_precision / predicted_unans

    fp_qs = [r for r in results if r["expected_category"] == "false_premise"]
    if fp_qs:
        tp = sum(1 for r in fp_qs if r["verdict"] == "false_premise")
        metrics["false_premise_detection_rate"] = tp / len(fp_qs)

    ans_qs = [r for r in results if r["expected_category"] == "answerable"]
    if ans_qs:
        correct_verdict = sum(1 for r in ans_qs if r["verdict"] == "answerable")
        metrics["answerable_verdict_rate"] = correct_verdict / len(ans_qs)

        checkable = [r for r in ans_qs if r["citation_match"] is not None]
        if checkable:
            citation_hits = sum(1 for r in checkable if r["citation_match"])
            metrics["citation_match_rate"] = citation_hits / len(checkable)
            metrics["citation_match_n"] = len(checkable)

    costs = [r["cost_usd"] for r in results if "cost_usd" in r]
    latencies = [r["latency_s"] for r in results if "latency_s" in r]
    if costs:
        metrics["mean_cost_usd"] = statistics.mean(costs)
    if latencies:
        sorted_lat = sorted(latencies)
        idx = int(0.95 * (len(sorted_lat) - 1))
        metrics["p95_latency_s"] = sorted_lat[idx]
        metrics["mean_latency_s"] = statistics.mean(latencies)

    adv = [r for r in results if r["expected_category"] == "adversarial"]
    if adv:
        broken = sum(1 for r in adv if r["verdict"] == "ERROR")
        metrics["adversarial_errors"] = broken
        metrics["adversarial_total"] = len(adv)

    return metrics


def print_metrics_table(metrics):
    rows = []
    for k, v in metrics.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        rows.append([k, v])
    print("\n" + tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--out", default="eval/results.json")
    args = parser.parse_args()

    print("Loading index and models...")
    retriever = Retriever()
    gate = TruthGate(retriever)

    results = run_eval(args.questions, gate, args.out)
    metrics = compute_metrics(results)
    print_metrics_table(metrics)

    metrics_path = Path(args.out).with_name("metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFull results -> {args.out}")
    print(f"Metrics summary -> {metrics_path}")


if __name__ == "__main__":
    main()
