"""Embedding generation module using google/siglip-base-patch16-384.

Generates 768-dimensional image and text embeddings using the SigLIP model
from HuggingFace.
"""

import io
import logging
import time
from typing import List, Optional

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

MODEL_NAME = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768

_model = None
_processor = None
_device = None


def get_model_and_processor():
    """Lazy-load the SigLIP model and processor (singleton)."""
    global _model, _processor, _device
    if _model is None or _processor is None:
        logger.info("Loading SigLIP model: %s", MODEL_NAME)
        start = time.time()

        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")

        logger.info("Using device: %s", _device)

        torch_dtype = torch.float16 if _device.type != "cpu" else torch.float32
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
        ).to(_device)
        _model.eval()

        _processor = AutoProcessor.from_pretrained(MODEL_NAME)

        elapsed = time.time() - start
        logger.info("Model loaded in %.1f seconds", elapsed)

    return _model, _processor


def get_device():
    """Get the current device the model is on."""
    get_model_and_processor()
    return _device


def download_image(image_url: str, max_retries: int = 3) -> Optional[Image.Image]:
    """Download an image from URL and return as PIL Image."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                image_url,
                timeout=30,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                },
            )
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            img = img.convert("RGB")
            return img
        except Exception as e:
            logger.warning(
                "Attempt %d/%d downloading image %s: %s",
                attempt + 1,
                max_retries,
                image_url,
                e,
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    return None


@torch.no_grad()
def generate_image_embedding(image_url: str) -> Optional[List[float]]:
    """Generate a 768-dim image embedding using SigLIP vision encoder."""
    img = download_image(image_url)
    if img is None:
        logger.error("Failed to download image: %s", image_url)
        return None

    model, processor = get_model_and_processor()
    device = get_device()

    try:
        # Process image
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        # Get vision model output and extract pooler output
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        
        if hasattr(vision_outputs, 'pooler_output'):
            image_features = vision_outputs.pooler_output
        else:
            # Fallback: use CLS token from last hidden state
            image_features = vision_outputs.last_hidden_state[:, 0, :]
        
        # L2 normalize
        image_features = F.normalize(image_features, p=2, dim=-1)

        embedding = image_features.squeeze().cpu().tolist()

        if len(embedding) != EMBEDDING_DIM:
            logger.warning(
                "Expected %d-dim embedding, got %d-dim",
                EMBEDDING_DIM,
                len(embedding),
            )

        return embedding
    except Exception as e:
        logger.error("Error generating image embedding: %s", e)
        return None


@torch.no_grad()
def generate_text_embedding(text: str) -> Optional[List[float]]:
    """Generate a 768-dim text embedding using SigLIP text encoder.

    Note: SigLIP's text model has a max position embedding of 64 tokens,
    so long inputs will be truncated.
    """
    if not text or not text.strip():
        return None

    model, processor = get_model_and_processor()
    device = get_device()

    try:
        # Truncate to SigLIP's max position (64 tokens) via the processor
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )

        # Move inputs to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Get text model output and extract pooler output
        text_outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if hasattr(text_outputs, 'pooler_output'):
            text_features = text_outputs.pooler_output
        else:
            # Fallback: use last hidden state pooled
            text_features = text_outputs.last_hidden_state[:, 0, :]

        # L2 normalize
        text_features = F.normalize(text_features, p=2, dim=-1)

        embedding = text_features.squeeze().cpu().tolist()

        if len(embedding) != EMBEDDING_DIM:
            logger.warning(
                "Expected %d-dim embedding, got %d-dim",
                EMBEDDING_DIM,
                len(embedding),
            )

        return embedding
    except Exception as e:
        logger.error("Error generating text embedding: %s", e)
        return None
