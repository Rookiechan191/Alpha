"""
CLI entrypoint for TruthGate.

Usage:
    python main.py "How do I create a ConfigMap?"
    python main.py --interactive
"""
import argparse
import json
import sys

from dotenv import load_dotenv

from core.pipeline import TruthGate
from retrieval.retriever import Retriever


def print_result(result: dict):
    print(f"\nVerdict: {result['verdict'].upper()}")
    print(f"Answer: {result['answer']}")
    if result["citations"]:
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  - {c}")
    print(f"\n[cost: ${result['cost_usd']:.5f} | latency: {result['latency_s']}s]")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    args = parser.parse_args()

    print("Loading index and models (this may take a moment)...", file=sys.stderr)
    retriever = Retriever()
    gate = TruthGate(retriever)

    if args.interactive:
        print("TruthGate interactive mode. Ctrl+C to exit.")
        while True:
            try:
                q = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not q.strip():
                continue
            result = gate.answer(q)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_result(result)
    elif args.question:
        result = gate.answer(args.question)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_result(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
