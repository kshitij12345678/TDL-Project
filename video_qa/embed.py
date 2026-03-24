"""
embed.py
CLIP-based image and text embeddings (normalized for cosine similarity).
"""

import itertools
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ── Model singleton (loaded once) ────────────────────────────────────────────
_MODEL_NAME = "openai/clip-vit-base-patch32"
_model: Optional[CLIPModel] = None
_processor: Optional[CLIPProcessor] = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _extract_tensor_features(output, *, kind: str) -> torch.Tensor:
    """
    Normalize CLIP output objects to a torch.Tensor.

    Some transformers versions return plain tensors from
    `get_image_features/get_text_features`, while others may return model
    outputs that contain the tensor under fields like `image_embeds`,
    `text_embeds`, or `pooler_output`.
    """
    if torch.is_tensor(output):
        return output

    # Common CLIP output fields first
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if torch.is_tensor(value):
                return value

    raise TypeError(
        f"Unexpected {kind} feature output type: {type(output).__name__}. "
        "Could not find tensor embeddings."
    )


def _load_model():
    global _model, _processor
    if _model is None:
        logger.info(f"Loading CLIP model on {_device} …")
        # use_safetensors=True avoids loading corrupted pytorch_model.bin blobs
        _model = CLIPModel.from_pretrained(_MODEL_NAME, use_safetensors=True).to(_device)
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        _model.eval()
        logger.info("CLIP model loaded.")
    return _model, _processor


def _normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot-product == cosine similarity."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


# ── Public API ────────────────────────────────────────────────────────────────

def embed_images(frame_infos: List[Dict[str, Any]], batch_size: int = 32) -> np.ndarray:
    """
    Produce L2-normalized CLIP image embeddings for a list of frame dicts.

    Args:
        frame_infos:  List of dicts with at least 'frame_path'.
        batch_size:   GPU/CPU batch size.

    Returns:
        np.ndarray of shape (N, 512), float32, L2-normalized.
        N may be less than len(frame_infos) if some frames could not be loaded.

    Raises:
        ValueError: If frame_infos is empty or no frames could be loaded.
    """
    if not frame_infos:
        raise ValueError("frame_infos is empty — nothing to embed.")

    model, processor = _load_model()
    all_embeddings = []

    for i in range(0, len(frame_infos), batch_size):
        batch: List[Dict[str, Any]] = list(itertools.islice(frame_infos, i, i + batch_size))

        # Load images, skipping any that are missing or unreadable
        images = []
        valid_indices = []
        for j, f in enumerate(batch):
            path = f.get("frame_path", "")
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(j)
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                logger.warning(f"Skipping frame '{path}': {e}")

        if not images:
            logger.warning(f"Batch {i // batch_size + 1} had no loadable frames — skipping.")
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(_device)

        with torch.no_grad():
            # Prefer full forward for compatibility across transformers versions.
            # Fallback to get_image_features for older/newer behavior differences.
            try:
                feats = _extract_tensor_features(model(**inputs), kind="image")
            except Exception:
                feats = _extract_tensor_features(model.get_image_features(**inputs), kind="image")

        all_embeddings.append(feats.cpu().float().numpy())
        logger.debug(f"Embedded batch {i // batch_size + 1} ({len(images)} images)")

    if not all_embeddings:
        raise ValueError("No frames could be embedded — all frame files were missing or corrupt.")

    embeddings = np.vstack(all_embeddings)
    return _normalize(embeddings).astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """
    Produce a L2-normalized CLIP text embedding.

    Args:
        text: Natural-language query string.

    Returns:
        np.ndarray of shape (1, 512), float32, L2-normalized.
        The leading dimension is kept so the result can be passed directly
        to faiss.IndexFlatIP.search(), which expects a 2D array.

    Raises:
        ValueError: If text is empty.
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string.")

    model, processor = _load_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(_device)

    with torch.no_grad():
        try:
            feats = _extract_tensor_features(model(**inputs), kind="text")
        except Exception:
            feats = _extract_tensor_features(model.get_text_features(**inputs), kind="text")

    # Shape: (1, 512) — keep 2D so FAISS search() receives the expected shape
    embedding = feats.cpu().float().numpy()
    return _normalize(embedding).astype(np.float32)
