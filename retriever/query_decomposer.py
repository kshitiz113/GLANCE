"""
Query Decomposer — Natural Language Query → Structured Constraints

Decomposes user queries into structured attribute constraints, mirroring
the attribute extraction done at index time. This consistency between
index-time and query-time decomposition is crucial for accurate matching.

Also auto-generates VQA verification questions for the re-ranking stage.

Example:
    Input:  "A red tie and a white shirt in a formal setting"
    Output: {
        "raw_query": "A red tie and a white shirt in a formal setting",
        "constraints": [
            {"garment": "tie", "color": "red"},
            {"garment": "shirt", "color": "white"}
        ],
        "environment": "formal",
        "style": "formal",
        "vqa_questions": [
            {"question": "What color is the tie?", "expected": "red"},
            {"question": "What color is the shirt?", "expected": "white"},
            {"question": "Is this a formal setting?", "expected": "yes"}
        ]
    }
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indexer.attribute_extractor import AttributeExtractor

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """
    Decomposes natural language queries into structured search constraints.

    Uses the same AttributeExtractor as the indexer to ensure consistency
    between how images are described and how queries are interpreted.
    """

    def __init__(self):
        self.extractor = AttributeExtractor()

    def decompose(self, query: str) -> Dict[str, Any]:
        """
        Full query decomposition pipeline.

        Args:
            query: Natural language search query

        Returns:
            Structured query with constraints, filters, and VQA questions
        """
        # Extract attributes using the same pipeline as indexing
        attrs = self.extractor.extract(query)

        # Build structured constraints from bindings
        constraints = []
        for binding in attrs.get("bindings", []):
            constraints.append({
                "garment": binding.get("garment"),
                "color": binding.get("color"),
            })

        # Add accessories as constraints too
        for acc in attrs.get("accessories", []):
            # Check if already covered by a binding
            if not any(c["garment"] == acc for c in constraints):
                constraints.append({"garment": acc, "color": None})

        # Generate VQA verification questions
        vqa_questions = self._generate_vqa_questions(attrs, constraints)

        return {
            "raw_query": query,
            "garments": attrs.get("garments", []),
            "colors": attrs.get("colors", []),
            "constraints": constraints,
            "environment": attrs.get("environment"),
            "style": attrs.get("style"),
            "accessories": attrs.get("accessories", []),
            "vqa_questions": vqa_questions,
        }

    def _generate_vqa_questions(
        self,
        attrs: Dict[str, Any],
        constraints: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Auto-generate VQA questions from extracted attributes.

        These questions will be asked to BLIP-2 on candidate images
        during the re-ranking stage to verify constraint satisfaction.

        Strategy:
        - For each color+garment binding: "What color is the [garment]?"
        - For each garment without color: "Is there a [garment]?"
        - For environment: "Is this in a [environment]?"
        - For style: "Is this a [style] outfit?"
        """
        questions = []

        # Per-garment color verification (the compositionality killer)
        for constraint in constraints:
            garment = constraint.get("garment")
            color = constraint.get("color")

            if garment and color:
                # Specific: "What color is the tie?" → expected "red"
                questions.append({
                    "question": f"What color is the {garment}?",
                    "expected": color,
                    "type": "color_binding",
                    "weight": 1.0,  # High weight — this is the key differentiator
                })
            elif garment:
                # Existence: "Is there a blazer in the image?"
                questions.append({
                    "question": f"Is there a {garment} in the image?",
                    "expected": "yes",
                    "type": "garment_exists",
                    "weight": 0.7,
                })

        # Environment verification
        env = attrs.get("environment")
        if env:
            questions.append({
                "question": f"Is this photo taken in a {env} setting?",
                "expected": "yes",
                "type": "environment",
                "weight": 0.6,
            })

        # Style verification
        style = attrs.get("style")
        if style:
            questions.append({
                "question": f"Is this a {style} outfit?",
                "expected": "yes",
                "type": "style",
                "weight": 0.5,
            })

        return questions

    def get_metadata_filters(self, decomposed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata filter criteria from decomposed query.

        Used for pre-filtering candidates before vector search,
        which speeds up retrieval and improves precision.

        Returns dict of filter criteria that can be matched against
        image metadata attributes.
        """
        filters = {}

        garments = decomposed.get("garments", [])
        if garments:
            filters["garments"] = garments

        colors = decomposed.get("colors", [])
        if colors:
            filters["colors"] = colors

        env = decomposed.get("environment")
        if env:
            filters["environment"] = env

        style = decomposed.get("style")
        if style:
            filters["style"] = style

        constraints = decomposed.get("constraints", [])
        if constraints:
            filters["bindings"] = constraints

        return filters


if __name__ == "__main__":
    decomposer = QueryDecomposer()

    test_queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
    ]

    for query in test_queries:
        result = decomposer.decompose(query)
        print(f"\nQuery: {query}")
        print(f"  Constraints:  {result['constraints']}")
        print(f"  Environment:  {result['environment']}")
        print(f"  Style:        {result['style']}")
        print(f"  VQA Questions:")
        for q in result["vqa_questions"]:
            print(f"    Q: {q['question']}  Expected: {q['expected']}  (weight: {q['weight']})")
