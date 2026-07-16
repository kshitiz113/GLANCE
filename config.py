"""
Configuration for Multimodal Fashion & Context Retrieval System.

Central configuration for all model names, file paths, embedding dimensions,
retrieval hyperparameters, and device selection.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch

# ============================================================================
# Device Configuration
# ============================================================================
def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
INDEX_DIR = os.path.join(PROJECT_ROOT, "index")
CAPTIONS_PATH = os.path.join(INDEX_DIR, "captions.json")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
FAISS_FASHION_INDEX = os.path.join(INDEX_DIR, "fashion_clip.index")
FAISS_SIGLIP_INDEX = os.path.join(INDEX_DIR, "siglip2.index")
FAISS_CAPTION_INDEX = os.path.join(INDEX_DIR, "caption.index")

# ============================================================================
# Model Names (Hugging Face)
# ============================================================================
# FashionCLIP — CLIP fine-tuned on 800K fashion products (Farfetch)
FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"

# SigLIP-2 — Google's improved vision-language model with sigmoid loss
SIGLIP2_MODEL = "google/siglip2-base-patch16-256"

# BLIP-2 — Salesforce VLM for captioning and VQA
# Use opt-2.7b for GPU with >= 6GB VRAM, flan-t5-xl for ~4GB, flan-t5-small for CPU
BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
BLIP2_MODEL_SMALL = "Salesforce/blip-vqa-base"

# ============================================================================
# Embedding Dimensions
# ============================================================================
FASHION_CLIP_DIM = 512
SIGLIP2_DIM = 768

# ============================================================================
# Captioning Configuration
# ============================================================================
CAPTION_PROMPTS = {
    "general": "",  # Open-ended caption
    "fashion": "Describe the clothing items, their colors, patterns, fabrics, and the setting in detail.",
    "style": "What style is this outfit? Describe if it is formal, casual, sporty, streetwear, or another style.",
}
CAPTION_MAX_NEW_TOKENS = 80
CAPTION_BATCH_SIZE = 4  # Adjust based on VRAM

# ============================================================================
# Attribute Extraction — Fashion Ontology
# ============================================================================
GARMENT_TERMS = [
    # Tops
    "shirt", "blouse", "t-shirt", "tee", "top", "tank top", "camisole",
    "sweater", "pullover", "cardigan", "hoodie", "sweatshirt", "turtleneck",
    "polo", "henley", "crop top", "tunic", "vest",
    # Formal tops
    "blazer", "suit jacket", "sport coat", "waistcoat", "button-down",
    "button-up", "dress shirt",
    # Bottoms
    "pants", "trousers", "jeans", "slacks", "chinos", "shorts",
    "skirt", "mini skirt", "midi skirt", "maxi skirt", "leggings",
    "culottes", "palazzo pants", "cargo pants", "joggers", "sweatpants",
    # Dresses & full body
    "dress", "gown", "jumpsuit", "romper", "overalls", "saree", "sari",
    "kimono", "kaftan",
    # Outerwear
    "jacket", "coat", "overcoat", "trench coat", "parka", "windbreaker",
    "raincoat", "poncho", "cape", "bomber jacket", "leather jacket",
    "denim jacket", "puffer jacket", "peacoat", "anorak",
    # Suits
    "suit", "tuxedo",
    # Activewear
    "sports bra", "athletic top", "jersey", "tracksuit",
]

COLOR_TERMS = [
    "red", "blue", "green", "yellow", "orange", "purple", "violet",
    "pink", "black", "white", "gray", "grey", "brown", "beige",
    "navy", "maroon", "burgundy", "teal", "turquoise", "cyan",
    "magenta", "coral", "salmon", "olive", "khaki", "tan",
    "cream", "ivory", "gold", "silver", "bronze",
    "light blue", "dark blue", "royal blue", "sky blue",
    "light green", "dark green", "forest green", "lime green",
    "bright yellow", "mustard", "pastel", "neon",
    "charcoal", "lavender", "indigo", "crimson", "scarlet",
]

ENVIRONMENT_TERMS = {
    "office": ["office", "workplace", "desk", "cubicle", "conference", "meeting room",
               "boardroom", "corporate", "workspace"],
    "street": ["street", "sidewalk", "road", "urban", "city", "downtown", "crosswalk",
               "metropolitan", "avenue", "boulevard"],
    "park": ["park", "garden", "bench", "grass", "tree", "outdoor", "nature",
             "lawn", "greenery", "pathway"],
    "home": ["home", "house", "living room", "bedroom", "kitchen", "couch", "sofa",
             "interior", "apartment", "domestic"],
    "beach": ["beach", "sand", "ocean", "sea", "shore", "coast", "waterfront"],
    "restaurant": ["restaurant", "cafe", "dining", "bar", "bistro"],
    "event": ["event", "party", "gala", "ceremony", "wedding", "red carpet",
              "runway", "fashion show", "stage"],
}

STYLE_TERMS = {
    "formal": ["formal", "professional", "business", "elegant", "sophisticated",
               "dressy", "classy", "polished", "tailored"],
    "casual": ["casual", "relaxed", "laid-back", "everyday", "comfortable",
               "effortless", "weekend", "easy-going"],
    "sporty": ["sporty", "athletic", "activewear", "sportswear", "gym",
               "workout", "fitness", "running"],
    "streetwear": ["streetwear", "street style", "urban", "hip-hop", "trendy",
                   "edgy", "cool", "contemporary"],
    "bohemian": ["bohemian", "boho", "hippie", "free-spirited", "eclectic",
                 "artistic", "flowy"],
    "vintage": ["vintage", "retro", "classic", "old-school", "throwback",
                "antique", "nostalgic"],
    "minimalist": ["minimalist", "minimal", "simple", "clean", "understated",
                   "sleek", "modern"],
}

ACCESSORY_TERMS = [
    "tie", "bow tie", "necktie", "scarf", "shawl", "stole",
    "hat", "cap", "beanie", "beret", "fedora", "sun hat",
    "glasses", "sunglasses", "eyeglasses",
    "watch", "bracelet", "necklace", "earrings", "ring",
    "belt", "suspenders",
    "bag", "purse", "handbag", "backpack", "clutch", "tote",
    "gloves", "mittens",
    "shoes", "boots", "sneakers", "heels", "sandals", "loafers", "flats",
]

# ============================================================================
# Retrieval Hyperparameters
# ============================================================================
# Score fusion weights for triple-vector search
# final = ALPHA * fashion_clip_score + BETA * siglip_image_score + GAMMA * caption_score
ALPHA = 0.30   # FashionCLIP weight (fashion-domain)
BETA = 0.30    # SigLIP-2 image weight (general vision)
GAMMA = 0.40   # Caption embedding weight (semantic/textual)

# Metadata attribute matching boost
ATTR_BOOST = 0.15  # Bonus per matched attribute constraint

# VQA re-ranking
VQA_WEIGHT = 0.35       # Weight of VQA score in final re-ranking
VQA_CANDIDATES = 50     # Number of candidates to re-rank with VQA
TOP_K_RETRIEVAL = 100   # Candidates from vector search before filtering
TOP_K_FINAL = 10        # Final results returned to user

# ============================================================================
# Image Processing
# ============================================================================
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
