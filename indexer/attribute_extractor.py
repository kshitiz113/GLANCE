"""
Attribute Extractor — Structured Decomposition with Attribute-Object Binding

Parses BLIP-2 generated captions into structured attribute dictionaries,
crucially preserving the binding between attributes and the objects they
describe. This is the key module that solves CLIP's compositionality problem.

Example:
    Caption: "A woman wearing a red blazer over a white blouse in an office"
    Output:  {
        "garments": ["blazer", "blouse"],
        "colors": ["red", "white"],
        "bindings": [{"garment": "blazer", "color": "red"},
                     {"garment": "blouse", "color": "white"}],
        "environment": "office",
        "style": "formal",
        "accessories": []
    }

Why this matters:
    Without bindings, "red blazer + white blouse" and "white blazer + red blouse"
    produce identical attribute sets {red, white, blazer, blouse}. The binding
    structure preserves which color belongs to which garment, enabling
    compositional query matching that vanilla CLIP cannot do.
"""

import os
import sys
import re
import json
import logging
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class AttributeExtractor:
    """
    Extracts structured fashion attributes from text captions.

    Uses a combination of:
    1. Pattern matching with fashion-specific ontology (from config.py)
    2. Proximity-based attribute-object binding (color nearest to garment)
    3. Keyword lookup for environment and style classification

    This approach is chosen over full NLP dependency parsing for robustness
    with BLIP-2's sometimes noisy caption output, while still achieving
    accurate attribute-object binding for most fashion descriptions.
    """

    def __init__(self):
        # Pre-compile regex patterns for efficiency
        # Sort by length (longest first) to match multi-word terms first
        self._garment_patterns = self._build_patterns(config.GARMENT_TERMS)
        self._color_patterns = self._build_patterns(config.COLOR_TERMS)
        self._accessory_patterns = self._build_patterns(config.ACCESSORY_TERMS)

    @staticmethod
    def _build_patterns(terms: List[str]) -> List[tuple]:
        """Build (compiled_regex, original_term) pairs sorted by length desc."""
        sorted_terms = sorted(terms, key=len, reverse=True)
        patterns = []
        for term in sorted_terms:
            # Word boundary matching, case insensitive
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            patterns.append((pattern, term.lower()))
        return patterns

    def extract_garments(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract garment mentions with their positions in text.

        Returns list of {"term": str, "start": int, "end": int}
        """
        text_lower = text.lower()
        found = []
        used_spans = set()

        for pattern, term in self._garment_patterns:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                # Avoid overlapping matches
                if not any(start < us_end and end > us_start for us_start, us_end in used_spans):
                    found.append({"term": term, "start": start, "end": end})
                    used_spans.add((start, end))

        return sorted(found, key=lambda x: x["start"])

    def extract_colors(self, text: str) -> List[Dict[str, Any]]:
        """Extract color mentions with positions."""
        found = []
        used_spans = set()

        for pattern, term in self._color_patterns:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if not any(start < us_end and end > us_start for us_start, us_end in used_spans):
                    found.append({"term": term, "start": start, "end": end})
                    used_spans.add((start, end))

        return sorted(found, key=lambda x: x["start"])

    def extract_accessories(self, text: str) -> List[str]:
        """Extract accessory mentions (names only, for metadata)."""
        found = set()
        for pattern, term in self._accessory_patterns:
            if pattern.search(text):
                found.add(term)
        return sorted(found)

    def extract_accessories_with_positions(self, text: str) -> List[Dict[str, Any]]:
        """Extract accessory mentions with positions (for binding)."""
        found = []
        used_spans = set()

        for pattern, term in self._accessory_patterns:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if not any(start < us_end and end > us_start for us_start, us_end in used_spans):
                    found.append({"term": term, "start": start, "end": end})
                    used_spans.add((start, end))

        return sorted(found, key=lambda x: x["start"])

    def extract_environment(self, text: str) -> Optional[str]:
        """
        Classify the environment/setting from text.
        Returns the environment category with the most keyword matches.
        """
        text_lower = text.lower()
        scores = {}

        for env_name, keywords in config.ENVIRONMENT_TERMS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[env_name] = score

        if scores:
            return max(scores, key=scores.get)
        return None

    def extract_style(self, text: str) -> Optional[str]:
        """
        Classify the style/formality from text.
        Returns the style category with the most keyword matches.
        """
        text_lower = text.lower()
        scores = {}

        for style_name, keywords in config.STYLE_TERMS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[style_name] = score

        if scores:
            return max(scores, key=scores.get)
        return None

    def bind_attributes(
        self,
        garments: List[Dict[str, Any]],
        colors: List[Dict[str, Any]],
        text: str
    ) -> List[Dict[str, str]]:
        """
        Bind colors to their nearest garment based on text proximity.

        This is the compositionality-solving logic:
        - For each color mention, find the nearest garment in the text
        - "red blazer and white blouse" → [{blazer: red}, {blouse: white}]
        - "white blazer and red blouse" → [{blazer: white}, {blouse: red}]

        Algorithm:
        1. For each garment, find the closest color (within a window)
        2. Prefer colors that appear directly before the garment (adjective position)
        3. Fall back to closest color by character distance

        Args:
            garments: List of garment dicts with 'term', 'start', 'end'
            colors: List of color dicts with 'term', 'start', 'end'
            text: Original text for context

        Returns:
            List of {"garment": str, "color": str} binding dicts
        """
        if not garments or not colors:
            # If we have garments but no colors (or vice versa), return unbound
            return [{"garment": g["term"], "color": None} for g in garments]

        bindings = []
        used_colors = set()

        for garment in garments:
            best_color = None
            best_distance = float("inf")

            for i, color in enumerate(colors):
                if i in used_colors:
                    continue

                # Distance: prefer colors BEFORE the garment (adjective position)
                # "red blazer" — color.end < garment.start
                if color["end"] <= garment["start"]:
                    distance = garment["start"] - color["end"]
                    # Bonus: very close = likely a direct modifier (e.g., "red blazer")
                    if distance < 5:
                        distance = distance * 0.5  # Strong bonus
                else:
                    # Color after garment (less common in English)
                    distance = abs(color["start"] - garment["end"]) + 20  # Penalty

                if distance < best_distance:
                    best_distance = distance
                    best_color = (i, color["term"])

            if best_color and best_distance < 100:  # Max window of 100 chars
                used_colors.add(best_color[0])
                bindings.append({
                    "garment": garment["term"],
                    "color": best_color[1],
                })
            else:
                bindings.append({
                    "garment": garment["term"],
                    "color": None,
                })

        return bindings

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Full extraction pipeline: text → structured attributes with bindings.

        Combines garments AND accessories into a unified item list for
        attribute-object binding. This ensures "red tie" correctly binds
        tie→red even though "tie" is an accessory, not a garment.

        Args:
            text: Caption text (typically BLIP-2 combined caption)

        Returns:
            Structured attribute dictionary with bindings.
        """
        garments = self.extract_garments(text)
        accessories_with_pos = self.extract_accessories_with_positions(text)
        colors = self.extract_colors(text)

        # Merge garments + accessories for unified binding
        # This is critical: "red tie and white shirt" needs tie in the binding list
        all_items = garments + accessories_with_pos
        # Remove duplicates (an item might match both garment and accessory lists)
        seen_spans = set()
        unique_items = []
        for item in sorted(all_items, key=lambda x: x["start"]):
            span = (item["start"], item["end"])
            if span not in seen_spans:
                unique_items.append(item)
                seen_spans.add(span)

        bindings = self.bind_attributes(unique_items, colors, text)

        return {
            "garments": [g["term"] for g in garments],
            "colors": [c["term"] for c in colors],
            "bindings": bindings,
            "environment": self.extract_environment(text),
            "style": self.extract_style(text),
            "accessories": self.extract_accessories(text),
        }

    def extract_from_captions(
        self, captions: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract attributes from all captions in a caption dictionary.

        Args:
            captions: Dict mapping image_id → {general, fashion, style, combined}

        Returns:
            Dict mapping image_id → structured attributes
        """
        attributes = {}
        for image_id, cap_dict in captions.items():
            # Use the combined caption for richest extraction
            combined = cap_dict.get("combined", "")
            attributes[image_id] = self.extract(combined)

        return attributes


def infer_style_from_garments(garments: List[str], accessories: List[str]) -> Optional[str]:
    """
    Infer style when text-based style detection fails.
    Uses garment types as a heuristic.

    Formal indicators: blazer, suit, dress shirt, tie, tuxedo, gown
    Casual indicators: t-shirt, hoodie, jeans, sneakers, shorts
    Sporty indicators: jersey, tracksuit, sports bra, athletic top
    """
    formal_items = {"blazer", "suit", "suit jacket", "dress shirt", "button-down",
                    "button-up", "tuxedo", "gown", "waistcoat", "tie", "bow tie"}
    casual_items = {"t-shirt", "tee", "hoodie", "jeans", "shorts", "sneakers",
                    "sweatshirt", "sweatpants", "joggers", "tank top", "flip-flops"}
    sporty_items = {"jersey", "tracksuit", "sports bra", "athletic top", "running"}

    all_items = set(garments) | set(accessories)

    formal_score = len(all_items & formal_items)
    casual_score = len(all_items & casual_items)
    sporty_score = len(all_items & sporty_items)

    if formal_score > casual_score and formal_score > sporty_score:
        return "formal"
    elif casual_score > formal_score and casual_score > sporty_score:
        return "casual"
    elif sporty_score > 0:
        return "sporty"
    return None


if __name__ == "__main__":
    # Quick test
    extractor = AttributeExtractor()

    test_captions = [
        "A woman wearing a red blazer over a white blouse, standing in a modern office",
        "A man in a bright yellow raincoat walking down a rainy street",
        "Someone in a blue denim shirt and khaki shorts sitting on a park bench",
        "A person wearing a black tuxedo with a red tie at a formal gala event",
        "Casual weekend look: grey hoodie, blue jeans, and white sneakers on a city sidewalk",
    ]

    for caption in test_captions:
        result = extractor.extract(caption)
        print(f"\nCaption: {caption}")
        print(f"  Garments:    {result['garments']}")
        print(f"  Colors:      {result['colors']}")
        print(f"  Bindings:    {result['bindings']}")
        print(f"  Environment: {result['environment']}")
        print(f"  Style:       {result['style']}")
        print(f"  Accessories: {result['accessories']}")
