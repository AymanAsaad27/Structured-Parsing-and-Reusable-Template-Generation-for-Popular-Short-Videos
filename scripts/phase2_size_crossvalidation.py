"""Supplementary grouped development assessment covering all five shot-size labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor

from beauty_video_templates.evaluation.metrics import classification_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("features", type=Path)
    p.add_argument("classifier_results", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    x, y, groups = [], [], []
    for path in sorted(a.features.glob("V???.npz")):
        with np.load(path) as data:
            x.extend(data["size_features"])
            y.extend(data["size_labels"])
            groups.extend([path.stem] * len(data["size_labels"]))
    x, y, groups = np.asarray(x), np.asarray(y, dtype=np.int64), np.asarray(groups)
    selection = json.loads(a.classifier_results.read_text())["probe_selection"]
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(x, y, groups)
    )
    plan = {
        "scope": "supplementary grouped development cross-validation; previously inspected dataset, not a fresh external test",
        "hyperparameters": selection,
        "hyperparameter_policy": "frozen from the separate validation experiment; no fold/test tuning",
        "folds": [
            {
                "train_ids": sorted(set(groups[train])),
                "test_ids": sorted(set(groups[test])),
                "test_support": np.bincount(y[test], minlength=5).tolist(),
            }
            for train, test in folds
        ],
    }
    plan_file = a.output / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Cross-validation plan changed")
    plan_file.write_text(json.dumps(plan, indent=2))
    predicted = np.full(len(y), -1, dtype=np.int64)
    for index, (train, test) in enumerate(folds):
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Video leakage")
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(
            C=selection["C"], class_weight=selection["class_weight"], max_iter=3000, random_state=42
        ).fit(scaler.transform(x[train]), y[train])
        predicted[test] = model.predict(scaler.transform(x[test]))
        print("Fold", index + 1, flush=True)
    torch.set_num_threads(2)
    clip_path = Path("models/weights/clip-b32")
    clip = CLIPModel.from_pretrained(clip_path, local_files_only=True, use_safetensors=True).eval()
    processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
    descriptions = [
        "an extreme close-up of a face, only eyes or lips visible",
        "a close-up portrait showing a person's face and shoulders",
        "a medium shot showing a person from the waist up",
        "a wide shot showing a full person and surroundings",
        "a detail shot of a beauty product, hand or small object",
    ]
    with torch.inference_mode():
        text = clip.get_text_features(
            **processor(text=descriptions, return_tensors="pt", padding=True)
        ).numpy()
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    zero = (x @ text.T).argmax(1)
    result = {
        "plan": plan,
        "zero_shot": classification_report(y.tolist(), zero.tolist(), list(range(5))),
        "probe_oof": classification_report(y.tolist(), predicted.tolist(), list(range(5))),
    }
    (a.output / "results.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        a.output / "paired_predictions.npz",
        groups=groups,
        reference=y,
        zero_shot=zero,
        probe_oof=predicted,
    )
    print("OOF accuracy", result["probe_oof"]["accuracy"], flush=True)


if __name__ == "__main__":
    main()
