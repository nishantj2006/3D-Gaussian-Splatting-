"""Shared CLIP/PCA encoding used by extraction, editing, and generated assets."""

from pathlib import Path
import pickle
import os

import numpy as np
import torch

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def load_pca(path):
    with open(Path(path).expanduser(), "rb") as handle:
        return pickle.load(handle)


def project_and_normalize(features, pca, device="cpu"):
    if isinstance(features, torch.Tensor):
        features = features.detach().float().cpu().numpy()
    projected = pca.transform(np.asarray(features, dtype=np.float32))
    result = torch.from_numpy(np.asarray(projected, dtype=np.float32)).to(device)
    return torch.nn.functional.normalize(result, dim=-1, eps=1e-12)


def load_clip(device="cuda", model_name=DEFAULT_CLIP_MODEL):
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor

    # Prefer offline cache access. On this Jetson, Hugging Face placed the
    # safetensors weights and config in different commit snapshots; support
    # that valid cache layout before falling back to a network fetch.
    try:
        model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
    except (OSError, AttributeError) as local_error:
        cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        repo = cache_home / "hub" / ("models--" + model_name.replace("/", "--")) / "snapshots"
        config_files = sorted(repo.glob("*/config.json"))
        weight_files = sorted(repo.glob("*/model.safetensors"))
        if not config_files or not weight_files:
            try:
                model = CLIPModel.from_pretrained(model_name)
                processor = CLIPProcessor.from_pretrained(model_name)
            except (OSError, AttributeError):
                raise local_error
        else:
            config_dir = config_files[-1].parent
            weights_dir = weight_files[-1].parent
            config = CLIPConfig.from_pretrained(config_dir, local_files_only=True)
            model = CLIPModel.from_pretrained(
                weights_dir, config=config, local_files_only=True, use_safetensors=True
            )
            processor = CLIPProcessor.from_pretrained(config_dir, local_files_only=True)
    return model.to(device).eval(), processor


def encode_text(texts, pca, device="cuda", model=None, processor=None):
    if isinstance(texts, str):
        texts = [texts]
    if model is None or processor is None:
        model, processor = load_clip(device)
    inputs = processor(text=list(texts), return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        features = torch.nn.functional.normalize(model.get_text_features(**inputs), dim=-1)
    return project_and_normalize(features, pca, device=device)


def encode_image(images, pca, device="cuda", model=None, processor=None):
    if not isinstance(images, (list, tuple)):
        images = [images]
    if model is None or processor is None:
        model, processor = load_clip(device)
    inputs = processor(images=list(images), return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        features = torch.nn.functional.normalize(model.get_image_features(**inputs), dim=-1)
    return project_and_normalize(features, pca, device=device)


def blend_embeddings(text_embedding, image_embedding=None, text_weight=0.5):
    if image_embedding is None:
        return torch.nn.functional.normalize(text_embedding, dim=-1, eps=1e-12)
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must be between 0 and 1")
    blended = text_weight * text_embedding + (1.0 - text_weight) * image_embedding
    return torch.nn.functional.normalize(blended, dim=-1, eps=1e-12)
