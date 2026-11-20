"""Adapter for the upstream TransNetV2 PyTorch architecture and local weights."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from .media import envelope, metadata


def parse(
    path,
    source=Path("models/weights/TransNetV2/inference-pytorch/transnetv2_pytorch.py"),
    weights=Path("models/weights/transnetv2-pytorch-weights.pth"),
):
    import av
    import torch

    if not source.is_file() or not weights.is_file():
        raise FileNotFoundError(
            "Install upstream TransNetV2 source and weights; see docs/phase2.md"
        )
    spec = importlib.util.spec_from_file_location("upstream_transnetv2", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    model = upstream.TransNetV2().eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    torch.set_num_threads(2)
    images = []
    times = []
    with av.open(str(path)) as video:
        for f in video.decode(video=0):
            images.append(f.reformat(width=48, height=27, format="rgb24").to_ndarray())
            times.append(float(f.time))
    times = np.asarray(times) - times[0]
    images = np.asarray(images)
    # Match the upstream 100-frame window, retaining its central 50 predictions.
    remainder = (50 - len(images) % 50) % 50
    padded = np.pad(images, ((25, 25 + remainder), (0, 0), (0, 0), (0, 0)), mode="edge")
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(padded) - 99, 50):
            logits, _ = model(torch.from_numpy(padded[start : start + 100][None]))
            predictions.extend(torch.sigmoid(logits[0, 25:75, 0]).numpy().tolist())
    predictions = np.asarray(predictions[: len(images)])
    positive = np.flatnonzero(predictions > 0.5)
    groups = np.split(positive, np.where(np.diff(positive) > 1)[0] + 1) if len(positive) else []
    cuts = sorted({int(g[np.argmax(predictions[g])]) for g in groups if len(g)})
    bounds = [0] + [x for x in cuts if x > 0] + [len(images)]
    duration = metadata(path)[0]["duration_seconds"]
    items = [
        {
            "start": float(times[a]),
            "end": float(times[b]) if b < len(times) else duration,
            "start_frame": a,
            "end_frame_exclusive": b,
            "shot_id": i,
        }
        for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
    ]
    return envelope(
        path,
        "shots",
        "transnetv2-pytorch",
        {
            "threshold": 0.5,
            "window": 100,
            "stride": 50,
            "device": "cpu",
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "weight_distribution": "transnetv2-pytorch 1.0.5 (community packaging)",
            "architecture": "soCzech/TransNetV2 upstream PyTorch implementation",
            "timing": "decoded presentation timestamps",
            "transition_group_policy": "maximum-score frame",
        },
        items,
    )
