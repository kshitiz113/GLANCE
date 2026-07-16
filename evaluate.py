"""
Evaluation Script — Run All 5 Assignment Queries + Ablation Comparison

Runs the 5 evaluation queries from the assignment specification across
three retrieval modes to demonstrate the progressive improvement:
1. Baseline (FashionCLIP-only)
2. Hybrid (triple-vector + metadata, no VQA)
3. Full Pipeline (hybrid + VQA re-ranking)

This generates the evidence needed for the PDF submission to show that
each architectural component contributes to retrieval quality.

Usage:
    python evaluate.py                  # Full evaluation
    python evaluate.py --top_k 10       # Top-10 results per query
    python evaluate.py --mode full      # Only run full pipeline
    python evaluate.py --save-report    # Save HTML report with images
"""

import os
import sys
import argparse
import logging
import json
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from retriever.run_retriever import FashionRetriever, format_results
from retriever.query_decomposer import QueryDecomposer

logger = logging.getLogger(__name__)

# The 5 evaluation queries from the assignment
EVALUATION_QUERIES = [
    {
        "id": "Q1",
        "type": "Attribute Specific",
        "query": "A person in a bright yellow raincoat.",
    },
    {
        "id": "Q2",
        "type": "Contextual/Place",
        "query": "Professional business attire inside a modern office.",
    },
    {
        "id": "Q3",
        "type": "Complex Semantic",
        "query": "Someone wearing a blue shirt sitting on a park bench.",
    },
    {
        "id": "Q4",
        "type": "Style Inference",
        "query": "Casual weekend outfit for a city walk.",
    },
    {
        "id": "Q5",
        "type": "Compositional",
        "query": "A red tie and a white shirt in a formal setting.",
    },
]


def run_evaluation(
    modes: List[str] = None,
    top_k: int = 5,
    save_report: bool = False,
):
    """
    Run evaluation across all queries and modes.

    Args:
        modes: List of modes to evaluate ['baseline', 'no_vqa', 'full']
        top_k: Number of results per query
        save_report: Whether to save an HTML report
    """
    modes = modes or ["baseline", "no_vqa", "full"]
    decomposer = QueryDecomposer()

    all_results = {}

    for mode in modes:
        print(f"\n{'=' * 70}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'=' * 70}")

        retriever = FashionRetriever(enable_vqa=(mode == "full"))

        mode_results = []

        for eq in EVALUATION_QUERIES:
            query_id = eq["id"]
            query_type = eq["type"]
            query = eq["query"]

            print(f"\n--- {query_id}: {query_type} ---")
            print(f"Query: \"{query}\"")

            # Show query decomposition
            decomposed = decomposer.decompose(query)
            if decomposed["constraints"]:
                print(f"  Constraints: {decomposed['constraints']}")
            if decomposed["environment"]:
                print(f"  Environment: {decomposed['environment']}")
            if decomposed["style"]:
                print(f"  Style: {decomposed['style']}")

            start = time.time()
            results = retriever.retrieve(query=query, top_k=top_k, mode=mode)
            elapsed = time.time() - start

            print(f"  Time: {elapsed:.2f}s")
            print(f"  Results:")
            print(format_results(results, show_details=(mode == "full")))

            mode_results.append({
                "query_id": query_id,
                "query_type": query_type,
                "query": query,
                "mode": mode,
                "time_seconds": elapsed,
                "results": [
                    {
                        "rank": i + 1,
                        "filename": r.get("filename"),
                        "score": r.get("final_score", r.get("score")),
                        "vqa_score": r.get("vqa_score"),
                    }
                    for i, r in enumerate(results)
                ],
            })

        all_results[mode] = mode_results

    # Save results
    results_path = os.path.join(config.INDEX_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate HTML report if requested
    if save_report:
        report_path = os.path.join(config.PROJECT_ROOT, "evaluation_report.html")
        generate_html_report(all_results, report_path, top_k)
        print(f"HTML report saved to {report_path}")

    # Print summary comparison
    print_comparison_summary(all_results)


def print_comparison_summary(all_results: Dict[str, List[Dict]]):
    """Print a comparison table across modes."""
    print(f"\n{'=' * 70}")
    print("  ABLATION COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Query':<8} {'Type':<22} {'Baseline':>10} {'No VQA':>10} {'Full':>10}")
    print("-" * 62)

    for i, eq in enumerate(EVALUATION_QUERIES):
        scores = {}
        for mode in ["baseline", "no_vqa", "full"]:
            if mode in all_results and i < len(all_results[mode]):
                results = all_results[mode][i].get("results", [])
                top_score = results[0]["score"] if results else 0.0
                scores[mode] = top_score

        print(
            f"{eq['id']:<8} {eq['type']:<22} "
            f"{scores.get('baseline', 0):.4f}     "
            f"{scores.get('no_vqa', 0):.4f}     "
            f"{scores.get('full', 0):.4f}"
        )


def generate_html_report(
    all_results: Dict[str, List[Dict]],
    output_path: str,
    top_k: int,
):
    """Generate a visual HTML report with result images."""
    html_parts = ["""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fashion Retrieval — Evaluation Report</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }
            h1 { color: #e94560; text-align: center; }
            h2 { color: #0f3460; background: #16213e; padding: 10px; border-radius: 8px; }
            h3 { color: #e94560; }
            .query-section { margin: 20px 0; padding: 15px; background: #16213e; border-radius: 12px; border-left: 4px solid #e94560; }
            .results-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
            .result-card { background: #0f3460; border-radius: 8px; padding: 8px; text-align: center; width: 180px; }
            .result-card img { width: 160px; height: 200px; object-fit: cover; border-radius: 6px; }
            .score { font-weight: bold; color: #e94560; font-size: 14px; }
            .mode-label { font-size: 12px; color: #a0a0a0; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 10px; border: 1px solid #333; text-align: center; }
            th { background: #0f3460; color: #e94560; }
            .best { color: #00ff88; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🔍 Multimodal Fashion Retrieval — Evaluation Report</h1>
    """]

    for eq in EVALUATION_QUERIES:
        html_parts.append(f"""
        <div class="query-section">
            <h3>{eq['id']}: {eq['type']}</h3>
            <p><strong>Query:</strong> "{eq['query']}"</p>
        """)

        for mode in ["baseline", "no_vqa", "full"]:
            if mode not in all_results:
                continue

            mode_data = next(
                (m for m in all_results[mode] if m["query_id"] == eq["id"]), None
            )
            if not mode_data:
                continue

            mode_labels = {"baseline": "Baseline (CLIP Only)", "no_vqa": "Hybrid (No VQA)", "full": "Full Pipeline"}
            html_parts.append(f"""
            <h4>{mode_labels.get(mode, mode)}</h4>
            <div class="results-grid">
            """)

            for r in mode_data.get("results", [])[:top_k]:
                filename = r.get("filename", "")
                score = r.get("score", 0)
                # Use relative path for images
                img_path = os.path.join(config.DATA_DIR, "test", filename)

                html_parts.append(f"""
                <div class="result-card">
                    <img src="file:///{img_path}" alt="{filename}" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYwIiBoZWlnaHQ9IjIwMCI+PHJlY3Qgd2lkdGg9IjE2MCIgaGVpZ2h0PSIyMDAiIGZpbGw9IiMzMzMiLz48dGV4dCB4PSI4MCIgeT0iMTAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjYWFhIj5JbWFnZTwvdGV4dD48L3N2Zz4='">
                    <div class="score">{score:.4f}</div>
                    <div class="mode-label">{filename[:20]}</div>
                </div>
                """)

            html_parts.append("</div>")

        html_parts.append("</div>")

    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Fashion Retrieval Evaluation")
    parser.add_argument("--top_k", type=int, default=5, help="Results per query")
    parser.add_argument("--mode", type=str, default=None,
                        help="Specific mode to evaluate (baseline/no_vqa/full)")
    parser.add_argument("--save-report", action="store_true",
                        help="Generate HTML report with images")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    modes = [args.mode] if args.mode else ["baseline", "no_vqa", "full"]

    run_evaluation(
        modes=modes,
        top_k=args.top_k,
        save_report=args.save_report,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
