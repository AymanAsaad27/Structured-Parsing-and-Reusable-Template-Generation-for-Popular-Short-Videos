"""Local visual estimates for an editable prototype, not validated VLM semantics."""

import threading
from pathlib import Path

import numpy as np

ROLES = [
    "hook",
    "product_introduction",
    "texture_demonstration",
    "application_demonstration",
    "result_comparison",
    "presenter_explanation",
    "call_to_action",
    "detail_cutaway",
]
ROLE_NAMES = [
    "Opening hook",
    "Product introduction",
    "Texture detail",
    "Application",
    "Result / comparison",
    "Presenter explanation",
    "Closing / call to action",
    "Detail cutaway",
]
VISUAL_PROMPTS = [
    "a striking close-up beauty portrait showing a face and glowing skin",
    "a beauty product bottle or cosmetic tube displayed clearly to the camera",
    "a macro close-up showing the texture of cosmetic cream or gel on a spatula",
    "a person applying a skincare mask or cosmetic cream to their face",
    "a side by side before and after comparison of a person's skin",
    "a person talking to the camera and explaining a beauty product",
    "a cosmetic product package displayed in a final advertising beauty shot",
    "a close-up of hands, a cosmetic tool or a small beauty product detail",
]
SIZE_NAMES = ["extreme_close_up", "close_up", "medium_shot", "wide_shot", "detail_shot"]
SIZE_PROMPTS = [
    "an extreme close-up of a face, only eyes or lips visible",
    "a close-up portrait showing a person's face and shoulders",
    "a medium shot showing a person from the waist up",
    "a wide shot showing a full person and surroundings",
    "a detail shot of a beauty product, hand or small object",
]
_loaded = {}
_lock = threading.RLock()


def visual_estimates(images, model_dir):
    """Classify supplied PIL images; never downloads a model implicitly."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    directory = Path(model_dir).resolve()
    if not (directory / "model.safetensors").is_file():
        raise FileNotFoundError("Local CLIP weights are missing; visual labels need manual review.")
    with _lock:
        torch.set_num_threads(2)
        key = str(directory)
        if key not in _loaded:
            model = CLIPModel.from_pretrained(
                directory, local_files_only=True, use_safetensors=True
            ).eval()
            processor = CLIPProcessor.from_pretrained(
                directory, local_files_only=True, use_fast=False
            )
            with torch.inference_mode():
                text = model.get_text_features(
                    **processor(
                        text=VISUAL_PROMPTS + SIZE_PROMPTS, return_tensors="pt", padding=True
                    )
                )
                text /= text.norm(dim=-1, keepdim=True)
            _loaded[key] = model, processor, text
        model, processor, text = _loaded[key]
        rows = []
        for start in range(0, len(images), 6):
            batch = images[start : start + 6]
            with torch.inference_mode():
                features = model.get_image_features(**processor(images=batch, return_tensors="pt"))
                features /= features.norm(dim=-1, keepdim=True)
                scores = (features @ text.T * 40).numpy()
            for score in scores:
                role = _softmax(score[: len(ROLES)])
                size = _softmax(score[len(ROLES) :])
                rows.append(
                    {
                        "role_scores": dict(zip(ROLES, role.tolist())),
                        "size_scores": dict(zip(SIZE_NAMES, size.tolist())),
                    }
                )
        return rows


def _softmax(values):
    x = np.exp(values - np.max(values))
    return x / x.sum()
