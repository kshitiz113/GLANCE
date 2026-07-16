# Multimodal Fashion & Context Retrieval System

An intelligent fashion image search engine that retrieves images from a database based on natural language queries. Goes **beyond vanilla CLIP** with a 4-stage pipeline that handles compositional queries like *"red tie and white shirt in a formal setting"*.

## Architecture

```
Query: "A red tie and white shirt in a formal setting"
                    │
    ┌───────────────┴───────────────┐
    │     QUERY DECOMPOSER          │
    │  constraints: [{tie:red},     │
    │                {shirt:white}] │
    │  environment: formal          │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │     HYBRID VECTOR SEARCH      │
    │  FashionCLIP ──► FAISS (512d) │
    │  SigLIP-2 img ─► FAISS (768d) │
    │  SigLIP-2 cap ─► FAISS (768d) │
    │  Score Fusion: α·F + β·S + γ·C│
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │     METADATA FILTERING        │
    │  Attribute-object binding     │
    │  boost for matching pairs     │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │     VQA RE-RANKING (BLIP-2)   │
    │  "What color is the tie?" →red│
    │  "What color is shirt?" →white│
    │  Per-constraint verification  │
    └───────────────┬───────────────┘
                    │
              Top-K Results
```

### Why Not Vanilla CLIP?

| Problem | CLIP Limitation | Our Solution |
|:--------|:----------------|:-------------|
| **Compositionality** | "red tie + white shirt" ≈ "white tie + red shirt" (same embedding) | Attribute-object binding + VQA verification |
| **Fashion specificity** | Trained on internet data, not fashion | FashionCLIP backbone (800K fashion products) |
| **Context separation** | Environment and clothing compete in one vector | Separate caption embeddings for semantic matching |

### Key Innovations

1. **Dual-Backbone Embeddings**: FashionCLIP (fashion-domain) + SigLIP-2 (general) capture both fashion nuance and contextual understanding
2. **BLIP-2 Captioning → Structured Attributes**: Multi-prompt captioning with attribute-object binding extraction
3. **Triple-Vector Fusion**: Three FAISS indices fused with configurable weights
4. **VQA Compositional Re-ranking**: Per-attribute constraint verification using BLIP-2 VQA

## Project Structure

```
├── config.py                    # Configuration & fashion ontology
├── setup_data.py                # Dataset extraction
├── requirements.txt             # Dependencies
│
├── indexer/                     # PART A: Indexing Pipeline
│   ├── caption_generator.py     # BLIP-2 multi-prompt captioning
│   ├── attribute_extractor.py   # Structured attribute extraction with bindings
│   ├── embedding_engine.py      # FashionCLIP + SigLIP-2 dual-backbone
│   ├── vector_store.py          # Triple FAISS index + metadata
│   └── run_indexer.py           # Orchestrator
│
├── retriever/                   # PART B: Retrieval Pipeline
│   ├── query_decomposer.py      # Query → structured constraints + VQA questions
│   ├── search_engine.py         # Hybrid search with score fusion
│   ├── vqa_reranker.py          # BLIP-2 VQA constraint verification
│   └── run_retriever.py         # CLI interface
│
├── evaluate.py                  # Run 5 evaluation queries + ablation
└── demo.py                      # Gradio web UI
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place `val_test2020.zip` in the project root, then:

```bash
python setup_data.py
```

This extracts ~3,200 Fashionpedia images to `data/test/`.

### 3. Build the Index (Part A)

```bash
# Full indexing pipeline (requires GPU for BLIP-2, takes ~2-4 hours)
python indexer/run_indexer.py

# Quick test with 50 images
python indexer/run_indexer.py --limit 50

# Resume after interruption (skips already-captioned images)
python indexer/run_indexer.py

# Skip captioning (use existing captions, only recompute embeddings)
python indexer/run_indexer.py --skip-captions
```

### 4. Run Queries (Part B)

```bash
# Full pipeline (with VQA re-ranking)
python retriever/run_retriever.py --query "A red tie and a white shirt in a formal setting" --top_k 5

# Without VQA (faster)
python retriever/run_retriever.py --query "A red tie and a white shirt" --no-vqa

# Baseline CLIP-only (for comparison)
python retriever/run_retriever.py --query "A red tie and a white shirt" --mode baseline

# Show detailed score breakdown
python retriever/run_retriever.py --query "Yellow raincoat" --details
```

### 5. Evaluate

```bash
# Run all 5 evaluation queries across all modes
python evaluate.py

# Generate HTML report with images
python evaluate.py --save-report

# Specific mode only
python evaluate.py --mode full
```

### 6. Interactive Demo

```bash
python demo.py
# Opens at http://localhost:7860
```

## Evaluation Queries

| # | Type | Query |
|:--|:-----|:------|
| Q1 | Attribute Specific | "A person in a bright yellow raincoat." |
| Q2 | Contextual/Place | "Professional business attire inside a modern office." |
| Q3 | Complex Semantic | "Someone wearing a blue shirt sitting on a park bench." |
| Q4 | Style Inference | "Casual weekend outfit for a city walk." |
| Q5 | Compositional | "A red tie and a white shirt in a formal setting." |

## Ablation Study

The system supports three modes for ablation comparison:

| Mode | Components | Best For |
|:-----|:-----------|:---------|
| `baseline` | FashionCLIP only | Simple attribute queries (Q1) |
| `no_vqa` | FashionCLIP + SigLIP-2 + Captions + Metadata | Context + style queries (Q2, Q3, Q4) |
| `full` | All + VQA re-ranking | Compositional queries (Q5) |

## Scalability

| Component | Current (3K images) | At Scale (1M images) |
|:----------|:--------------------|:---------------------|
| FAISS Index | `IndexFlatIP` (exact) | `IndexIVFFlat` / `IndexHNSWFlat` (ANN) |
| Metadata | JSON file | PostgreSQL + Elasticsearch |
| Captioning | On-the-fly BLIP-2 | Pre-computed, batched on GPU cluster |
| VQA Re-ranking | Top-50 candidates | Top-50 (sublinear — only re-ranks shortlist) |

## Models Used

| Model | Purpose | Size |
|:------|:--------|:-----|
| [FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip) | Fashion-domain image/text embeddings | ~400MB |
| [SigLIP-2](https://huggingface.co/google/siglip2-base-patch16-256) | General vision-language embeddings | ~400MB |
| [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) | Captioning + VQA | ~6GB (GPU) |

## License

MIT
