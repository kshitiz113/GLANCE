"""
Retriever CLI — Main Retrieval Interface

Provides a command-line interface for querying the fashion image search engine.
Supports multiple retrieval modes for ablation comparison:

1. Full Pipeline:     Triple-vector + metadata + VQA re-ranking
2. No VQA:            Triple-vector + metadata (without VQA)
3. Baseline CLIP:     FashionCLIP-only search (for comparison)

Usage:
    python retriever/run_retriever.py --query "A red tie and a white shirt in a formal setting" --top_k 5
    python retriever/run_retriever.py --query "Casual weekend outfit" --mode baseline
    python retriever/run_retriever.py --query "Yellow raincoat" --no-vqa
"""

import os
import sys
import argparse
import logging
import json
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from retriever.search_engine import SearchEngine
from retriever.vqa_reranker import VQAReranker
from retriever.query_decomposer import QueryDecomposer

logger = logging.getLogger(__name__)


class FashionRetriever:
    """
    End-to-end fashion image retriever.

    Combines the search engine (triple-vector + metadata) with the VQA
    re-ranker for compositional constraint verification.
    """

    def __init__(self, enable_vqa: bool = True):
        """
        Initialize the retriever.

        Args:
            enable_vqa: Whether to use VQA re-ranking.
                        Set False for faster retrieval without compositional verification.
        """
        self.search_engine = SearchEngine()
        self.decomposer = QueryDecomposer()
        self.enable_vqa = enable_vqa
        self.reranker = VQAReranker() if enable_vqa else None

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        mode: str = "full",
    ) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            mode: Retrieval mode:
                  'full'     — Triple-vector + metadata + VQA (default)
                  'no_vqa'   — Triple-vector + metadata only
                  'baseline' — FashionCLIP-only (for ablation)

        Returns:
            List of result dicts sorted by relevance
        """
        top_k = top_k or config.TOP_K_FINAL

        start_time = time.time()

        # Mode: Baseline CLIP-only
        if mode == "baseline":
            results = self.search_engine.search_baseline_clip(query, top_k=top_k)
            elapsed = time.time() - start_time
            logger.info(f"Baseline search: {len(results)} results in {elapsed:.2f}s")
            return results

        # Step 1: Hybrid search (triple-vector + metadata)
        candidates = self.search_engine.search(
            query, top_k=config.VQA_CANDIDATES
        )

        # Mode: No VQA
        if mode == "no_vqa" or not self.enable_vqa:
            elapsed = time.time() - start_time
            logger.info(f"Hybrid search (no VQA): {len(candidates)} results in {elapsed:.2f}s")
            return candidates[:top_k]

        # Step 2: VQA re-ranking (full pipeline)
        decomposed = self.decomposer.decompose(query)
        vqa_questions = decomposed.get("vqa_questions", [])

        if vqa_questions and self.reranker:
            results = self.reranker.rerank(
                candidates=candidates,
                vqa_questions=vqa_questions,
            )
        else:
            results = candidates

        elapsed = time.time() - start_time
        logger.info(f"Full pipeline: {len(results)} results in {elapsed:.2f}s")

        return results[:top_k]


def format_results(results: List[Dict[str, Any]], show_details: bool = False) -> str:
    """Format results for CLI display."""
    lines = []
    for i, r in enumerate(results, 1):
        score = r.get("final_score", r.get("score", 0.0))
        filename = r.get("filename", "unknown")
        lines.append(f"  #{i}  Score: {score:.4f}  File: {filename}")

        if show_details:
            breakdown = r.get("score_breakdown", {})
            if breakdown:
                lines.append(f"       FashionCLIP: {breakdown.get('fashion_clip', 0):.4f}  "
                             f"SigLIP: {breakdown.get('siglip_image', 0):.4f}  "
                             f"Caption: {breakdown.get('caption', 0):.4f}  "
                             f"AttrBoost: {breakdown.get('attr_boost', 0):.4f}")

            vqa_details = r.get("vqa_details", [])
            if vqa_details:
                lines.append("       VQA Verification:")
                for vd in vqa_details:
                    check = "[PASS]" if vd["similarity"] > 0.5 else "[FAIL]"
                    lines.append(f"         {check} Q: {vd['question']} -> "
                                 f"A: \"{vd['answer']}\" (expected: \"{vd['expected']}\", "
                                 f"sim: {vd['similarity']:.2f})")
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Fashion Image Retriever")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--mode", choices=["full", "no_vqa", "baseline"], default="full",
                        help="Retrieval mode")
    parser.add_argument("--no-vqa", action="store_true", help="Disable VQA re-ranking")
    parser.add_argument("--details", action="store_true", help="Show score breakdown")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mode = "no_vqa" if args.no_vqa else args.mode

    print(f"\n{'=' * 60}")
    print(f"FASHION IMAGE RETRIEVAL")
    print(f"Query: \"{args.query}\"")
    print(f"Mode:  {mode}")
    print(f"Top-K: {args.top_k}")
    print(f"{'=' * 60}\n")

    # Decompose and show query understanding
    decomposer = QueryDecomposer()
    decomposed = decomposer.decompose(args.query)
    print("Query Understanding:")
    print(f"  Garments:    {decomposed['garments']}")
    print(f"  Colors:      {decomposed['colors']}")
    print(f"  Constraints: {decomposed['constraints']}")
    print(f"  Environment: {decomposed['environment']}")
    print(f"  Style:       {decomposed['style']}")
    if decomposed["vqa_questions"]:
        print(f"  VQA Questions:")
        for q in decomposed["vqa_questions"]:
            print(f"    -> {q['question']} (expected: {q['expected']})")
    print()

    # Run retrieval
    retriever = FashionRetriever(enable_vqa=(mode == "full"))
    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        mode=mode,
    )

    # Display results
    print(f"Results ({len(results)} matches):")
    print(format_results(results, show_details=args.details))

    # Save if requested
    if args.save:
        # Strip non-serializable fields
        save_results = []
        for r in results:
            save_r = {
                "filename": r.get("filename"),
                "path": r.get("path"),
                "score": r.get("final_score", r.get("score")),
                "score_breakdown": r.get("score_breakdown"),
                "vqa_details": r.get("vqa_details"),
            }
            save_results.append(save_r)

        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"query": args.query, "mode": mode, "results": save_results}, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
