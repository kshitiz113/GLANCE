"""
Gradio Demo — Interactive Fashion Image Retrieval UI

Provides a web-based interface for querying the fashion retrieval system.
Features:
- Text input for natural language queries
- Top-K slider
- Mode selector (Full/No VQA/Baseline) for live ablation comparison
- Image grid showing results with scores
- Query decomposition display (shows how the system understands the query)

Usage:
    python demo.py
    # Opens at http://localhost:7860
"""

import os
import sys
import logging
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from retriever.run_retriever import FashionRetriever
from retriever.query_decomposer import QueryDecomposer

logger = logging.getLogger(__name__)

# Initialize components globally (loaded once)
decomposer = QueryDecomposer()
retriever_full = None
retriever_no_vqa = None


def get_retriever(mode: str) -> FashionRetriever:
    """Get or create retriever for the specified mode."""
    global retriever_full, retriever_no_vqa

    if mode == "full":
        if retriever_full is None:
            retriever_full = FashionRetriever(enable_vqa=True)
        return retriever_full
    else:
        if retriever_no_vqa is None:
            retriever_no_vqa = FashionRetriever(enable_vqa=False)
        return retriever_no_vqa


def search(query: str, top_k: int, mode: str):
    """
    Main search function called by Gradio interface.

    Returns:
        - List of (image, label) tuples for the gallery
        - Query decomposition text
        - Score details text
    """
    if not query.strip():
        return [], "Enter a query to search.", ""

    # Decompose query
    decomposed = decomposer.decompose(query)

    decomp_text = "**Query Understanding:**\n"
    decomp_text += f"- **Garments:** {decomposed['garments']}\n"
    decomp_text += f"- **Colors:** {decomposed['colors']}\n"
    decomp_text += f"- **Constraints:** {decomposed['constraints']}\n"
    decomp_text += f"- **Environment:** {decomposed['environment']}\n"
    decomp_text += f"- **Style:** {decomposed['style']}\n"

    if decomposed["vqa_questions"]:
        decomp_text += "\n**VQA Verification Questions:**\n"
        for q in decomposed["vqa_questions"]:
            decomp_text += f"- {q['question']} (expected: *{q['expected']}*)\n"

    # Run retrieval
    retriever = get_retriever(mode)
    actual_mode = mode if mode != "full" else "full"
    if mode == "baseline":
        actual_mode = "baseline"
    elif mode == "hybrid":
        actual_mode = "no_vqa"

    results = retriever.retrieve(query=query, top_k=top_k, mode=actual_mode)

    # Build gallery
    gallery_items = []
    details_lines = []

    for i, r in enumerate(results, 1):
        image_path = r.get("path", "")
        filename = r.get("filename", "unknown")
        score = r.get("final_score", r.get("score", 0.0))

        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                label = f"#{i} | {score:.4f} | {filename[:25]}"
                gallery_items.append((img, label))
            except Exception:
                pass

        # Score breakdown
        breakdown = r.get("score_breakdown", {})
        detail = f"**#{i}** {filename}\n"
        detail += f"  Score: {score:.4f}"
        if breakdown:
            detail += f" (F:{breakdown.get('fashion_clip', 0):.3f} "
            detail += f"S:{breakdown.get('siglip_image', 0):.3f} "
            detail += f"C:{breakdown.get('caption', 0):.3f})"

        vqa_details = r.get("vqa_details", [])
        if vqa_details:
            detail += "\n  VQA: "
            for vd in vqa_details:
                check = "✓" if vd["similarity"] > 0.5 else "✗"
                detail += f"{check} "
            vqa_score = r.get("vqa_score", 0)
            detail += f" (VQA: {vqa_score:.3f})"

        details_lines.append(detail)

    details_text = "\n\n".join(details_lines) if details_lines else "No results found."

    return gallery_items, decomp_text, details_text


def create_demo():
    """Create the Gradio interface."""
    with gr.Blocks(
        title="Fashion Retrieval Engine",
    ) as demo:
        gr.Markdown(
            """
            # 🔍 Multimodal Fashion & Context Retrieval
            ### Attribute-Decomposed Hybrid Search with VQA Compositional Verification
            
            Search through fashion images using natural language. The system understands
            clothing types, colors, environments, and style — going beyond vanilla CLIP
            with compositional attribute verification.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., A red tie and a white shirt in a formal setting",
                    lines=2,
                )
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Top K Results",
                )
                mode_dropdown = gr.Dropdown(
                    choices=["full", "hybrid", "baseline"],
                    value="full",
                    label="Retrieval Mode",
                    info="Full = VQA re-ranking, Hybrid = no VQA, Baseline = CLIP only"
                )
                search_btn = gr.Button("🔍 Search", variant="primary")

        # Example queries
        gr.Examples(
            examples=[
                ["A person in a bright yellow raincoat.", 5, "full"],
                ["Professional business attire inside a modern office.", 5, "full"],
                ["Someone wearing a blue shirt sitting on a park bench.", 5, "full"],
                ["Casual weekend outfit for a city walk.", 5, "full"],
                ["A red tie and a white shirt in a formal setting.", 5, "full"],
            ],
            inputs=[query_input, top_k_slider, mode_dropdown],
        )

        with gr.Row():
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Search Results",
                    columns=5,
                    height=400,
                    object_fit="cover",
                )
            with gr.Column(scale=1):
                decomp_output = gr.Markdown(label="Query Understanding")

        details_output = gr.Markdown(label="Score Details")

        # Wire up the search
        search_btn.click(
            fn=search,
            inputs=[query_input, top_k_slider, mode_dropdown],
            outputs=[gallery, decomp_output, details_output],
        )

        query_input.submit(
            fn=search,
            inputs=[query_input, top_k_slider, mode_dropdown],
            outputs=[gallery, decomp_output, details_output],
        )

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo = create_demo()
    theme = gr.themes.Soft(primary_hue="red", secondary_hue="blue")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            theme=theme,
        )
    except OSError:
        logger.warning("Port 7860 is busy, automatically finding an available port...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=None,
            share=False,
            theme=theme,
        )
