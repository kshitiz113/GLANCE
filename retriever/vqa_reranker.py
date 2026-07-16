"""
VQA Re-ranker — BLIP-2 Visual Question Answering for Compositional Verification

The key differentiator of this system. After initial retrieval produces
a shortlist of ~50 candidates, this module verifies specific attribute
constraints by asking BLIP-2 targeted questions about each candidate image.

Why this matters (the CLIP compositionality problem):
    Query: "A red tie and a white shirt"
    
    Candidate A: red tie + white shirt  → VQA: "What color is the tie?" → "red" ✓
    Candidate B: white tie + red shirt  → VQA: "What color is the tie?" → "white" ✗
    
    CLIP gives both the SAME score. VQA re-ranking correctly ranks A above B.

How it works:
    1. Query decomposer generates verification questions at query time:
       - "What color is the tie?" (expected: "red")
       - "What color is the shirt?" (expected: "white")
       - "Is this a formal setting?" (expected: "yes")
    2. For each candidate image, ask BLIP-2 each question
    3. Compare answers to expected answers using semantic similarity
    4. Score each candidate by constraint satisfaction
    5. Re-rank: final_score = (1-w)·retrieval_score + w·vqa_score

Performance note:
    VQA runs only on the shortlisted candidates (typically 50), not the
    full corpus, making it O(K) not O(N). At ~10ms per question per image,
    50 candidates × 3 questions = ~1.5s — acceptable for search.
"""

import os
import sys
import re
import logging
from typing import Dict, List, Optional, Any

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class VQAReranker:
    """
    Re-ranks retrieval candidates using BLIP-2 Visual Question Answering.

    For each candidate image, asks auto-generated verification questions
    and scores based on how well the answers match expected values.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.device = device or config.DEVICE
        self.model_name = model_name or (
            config.BLIP2_MODEL if self.device == "cuda" else config.BLIP2_MODEL_SMALL
        )
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self):
        """Lazy-load BLIP/BLIP-2 for VQA."""
        if self._loaded:
            return

        logger.info(f"Loading VQA model: {self.model_name} on {self.device}")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if "blip2" in self.model_name.lower():
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            )
        elif "vqa" in self.model_name.lower():
            from transformers import BlipProcessor, BlipForQuestionAnswering
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(
                self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            )
        else:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            )

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("VQA model loaded")

    def ask_question(self, image: Image.Image, question: str) -> str:
        """
        Ask a question about an image using BLIP-2 VQA.

        Args:
            image: PIL Image
            question: Question string (e.g., "What color is the tie?")

        Returns:
            Answer string from BLIP-2
        """
        self.load_model()

        # Format prompt based on model type
        if "vqa" in self.model.__class__.__name__.lower():
            prompt = question
        else:
            prompt = f"Question: {question} Answer:"

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        inputs = {
            k: v.to(self.device, dtype=self.model.dtype) if hasattr(self.model, "dtype") and v.dtype == torch.float32 else v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=3,
                early_stopping=True,
            )

        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # Clean up repetitive prompt mirroring from non-VQA checkpoints
        if "answer:" in answer.lower() or "answer :" in answer.lower():
            parts = re.split(r"answer\s*:", answer, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer = parts[-1]
        answer = re.sub(r"\b(answer|question)\b.*", "", answer, flags=re.IGNORECASE).strip()
        if not answer:
            answer = "yes"
        return answer.lower()

    def compute_answer_similarity(self, answer: str, expected: str) -> float:
        """
        Compute similarity between VQA answer and expected answer.

        Uses simple string matching with fuzzy tolerance:
        - Exact match: 1.0
        - Contains expected: 0.8
        - Yes/No questions: binary match
        - Partial color match: 0.5

        For production, this could use embedding-based semantic similarity.
        """
        answer = answer.lower().strip()
        expected = expected.lower().strip()

        # Exact match
        if answer == expected:
            return 1.0

        # Expected is contained in answer (e.g., answer="dark red" expected="red")
        if expected in answer:
            return 0.85

        # Answer is contained in expected
        if answer in expected:
            return 0.7

        # Yes/No question handling
        if expected in ("yes", "no"):
            positive_words = {"yes", "yeah", "correct", "true", "right", "indeed", "absolutely"}
            negative_words = {"no", "not", "nope", "false", "wrong", "none", "neither"}

            answer_words = set(answer.split())
            if expected == "yes" and answer_words & positive_words:
                return 1.0
            if expected == "no" and answer_words & negative_words:
                return 1.0
            if expected == "yes" and answer_words & negative_words:
                return 0.0
            if expected == "no" and answer_words & positive_words:
                return 0.0

        # Color family matching (e.g., "crimson" ~ "red")
        color_families = {
            "red": {"red", "crimson", "scarlet", "ruby", "maroon", "burgundy", "cherry"},
            "blue": {"blue", "navy", "azure", "cobalt", "royal blue", "sky blue", "indigo"},
            "green": {"green", "olive", "emerald", "lime", "forest green", "sage", "teal"},
            "yellow": {"yellow", "gold", "mustard", "lemon", "amber"},
            "white": {"white", "cream", "ivory", "off-white", "pearl"},
            "black": {"black", "charcoal", "ebony", "jet black"},
            "pink": {"pink", "magenta", "fuchsia", "rose", "blush"},
            "brown": {"brown", "tan", "beige", "khaki", "camel", "chocolate"},
            "gray": {"gray", "grey", "silver", "slate", "charcoal"},
            "purple": {"purple", "violet", "lavender", "plum", "mauve"},
            "orange": {"orange", "tangerine", "peach", "coral", "amber"},
        }

        for family, members in color_families.items():
            if answer in members and expected in members:
                return 0.7
            if expected == family and answer in members:
                return 0.8
            if answer == family and expected in members:
                return 0.8

        return 0.0

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        vqa_questions: List[Dict[str, str]],
        vqa_weight: float = None,
        max_candidates: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using VQA constraint verification.

        Args:
            candidates: List of candidate dicts from search engine
                        (must have 'path' and 'score' fields)
            vqa_questions: List of {question, expected, type, weight} dicts
                           from query decomposer
            vqa_weight: Weight of VQA score in final ranking
            max_candidates: Max candidates to re-rank (for speed)

        Returns:
            Re-ranked candidate list with VQA scores added
        """
        if not vqa_questions:
            logger.info("No VQA questions generated — skipping re-ranking")
            return candidates

        vqa_weight = vqa_weight if vqa_weight is not None else config.VQA_WEIGHT
        max_candidates = max_candidates or config.VQA_CANDIDATES

        # Only re-rank top candidates (VQA is expensive)
        to_rerank = candidates[:max_candidates]
        rest = candidates[max_candidates:]

        logger.info(f"VQA re-ranking {len(to_rerank)} candidates with {len(vqa_questions)} questions")

        for candidate in to_rerank:
            image_path = candidate.get("path", "")
            if not os.path.exists(image_path):
                candidate["vqa_score"] = 0.0
                candidate["vqa_details"] = []
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {image_path}: {e}")
                candidate["vqa_score"] = 0.0
                candidate["vqa_details"] = []
                continue

            # Ask each verification question
            vqa_details = []
            weighted_score = 0.0
            total_weight = 0.0

            for vq in vqa_questions:
                question = vq["question"]
                expected = vq["expected"]
                q_weight = vq.get("weight", 1.0)

                answer = self.ask_question(image, question)
                similarity = self.compute_answer_similarity(answer, expected)

                vqa_details.append({
                    "question": question,
                    "expected": expected,
                    "answer": answer,
                    "similarity": similarity,
                })

                weighted_score += q_weight * similarity
                total_weight += q_weight

            # Normalize VQA score to [0, 1]
            vqa_score = weighted_score / max(total_weight, 1e-6)

            candidate["vqa_score"] = vqa_score
            candidate["vqa_details"] = vqa_details

            # Compute final re-ranked score
            retrieval_score = candidate.get("score", 0.0)
            candidate["final_score"] = (
                (1 - vqa_weight) * retrieval_score + vqa_weight * vqa_score
            )

        # Sort re-ranked candidates by final score
        to_rerank.sort(key=lambda x: x.get("final_score", x.get("score", 0)), reverse=True)

        if self.device == "cpu" and self._loaded:
            import gc
            self.model = None
            self.processor = None
            self._loaded = False
            gc.collect()

        # Append un-re-ranked candidates at the end
        return to_rerank + rest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("VQA Re-ranker module loaded. Use with retrieval pipeline.")
