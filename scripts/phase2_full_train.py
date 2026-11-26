"""Validation-selected classifiers on the frozen full-data protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from train_phase2_comparisons import TransitionCNN

from beauty_video_templates.evaluation.metrics import classification_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("features", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    protocol = json.loads((a.features / "protocol.json").read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    split = protocol["split"]
    missing = [k for k in split if not (a.features / (k + ".npz")).exists()]
    if missing:
        raise ValueError(f"Full extraction incomplete: {len(missing)} videos missing")
    parts = {
        name: {"pairs": [], "ty": [], "xf": [], "sy": []}
        for name in ["train", "validation", "test"]
    }
    for key, part in split.items():
        d = np.load(a.features / (key + ".npz"))
        dest = parts[part]
        for target, source in [
            ("pairs", "pairs"),
            ("ty", "transition_labels"),
            ("xf", "size_features"),
            ("sy", "size_labels"),
        ]:
            dest[target].append(d[source])
    for part in parts.values():
        for key in part:
            part[key] = np.concatenate(part[key])
            if key in {"ty", "sy"}:
                part[key] = part[key].astype(np.int64)
    torch.set_num_threads(2)
    train, val, test = [parts[k] for k in ["train", "validation", "test"]]
    tx = torch.tensor(train["pairs"])
    ty = torch.tensor(train["ty"], dtype=torch.long)
    vx = torch.tensor(val["pairs"])
    vy = torch.tensor(val["ty"], dtype=torch.long)
    counts = np.bincount(train["ty"], minlength=4)
    candidates = []
    for weighting in ["sqrt_inverse", "inverse"]:
        torch.manual_seed(42)
        net = TransitionCNN()
        net.layers[-1] = nn.Linear(24, 4)
        weights = (counts.sum() / np.maximum(counts, 1)) ** (
            0.5 if weighting == "sqrt_inverse" else 1
        )
        weight = torch.tensor(weights / weights.mean(), dtype=torch.float32)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.01)
        best = (-1, -float("inf"))
        state = None
        epoch_best = 0
        for epoch in range(100):
            net.train()
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(net(tx), ty, weight=weight)
            loss.backward()
            optimizer.step()
            net.eval()
            with torch.inference_mode():
                logits = net(vx)
                pred = logits.argmax(1).numpy()
                nll = float(nn.functional.cross_entropy(logits, vy))
            score = classification_report(val["ty"].tolist(), pred.tolist(), list(range(4)))[
                "macro_f1"
            ]
            if (score, -nll) > best:
                best = (score, -nll)
                epoch_best = epoch + 1
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        candidates.append(
            {
                "weighting": weighting,
                "validation_macro_f1": best[0],
                "validation_nll": -best[1],
                "epoch": epoch_best,
                "state": state,
            }
        )
        print("Validation candidate:", weighting, best[0], flush=True)
    chosen = max(candidates, key=lambda x: (x["validation_macro_f1"], -x["validation_nll"]))
    selected = TransitionCNN()
    selected.layers[-1] = nn.Linear(24, 4)
    selected.load_state_dict(chosen["state"])
    selected.eval()
    torch.save(
        {"state_dict": chosen["state"], "labels": protocol["transition_labels"]},
        a.output / "transition_four_class.pt",
    )
    scaler = StandardScaler().fit(train["xf"])
    st = scaler.transform(train["xf"])
    sv = scaler.transform(val["xf"])
    probes = []
    for c in [0.001, 0.01, 0.1, 1, 10]:
        for weight in [None, "balanced"]:
            model = LogisticRegression(
                C=c, class_weight=weight, max_iter=3000, random_state=42
            ).fit(st, train["sy"])
            score = classification_report(
                val["sy"].tolist(), model.predict(sv).tolist(), list(range(5))
            )["macro_f1"]
            probes.append((score, c, weight, model))
    best = max(probes, key=lambda row: row[0])
    model = best[3]
    np.savez(
        a.output / "shot_size_probe.npz",
        coef=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
        mean=scaler.mean_,
        scale=scaler.scale_,
    )
    # Test predictions are produced only after both selections are complete.
    with torch.inference_mode():
        pred = selected(torch.tensor(test["pairs"])).argmax(1).numpy()
    legacy = TransitionCNN()
    legacy.load_state_dict(
        torch.load(
            "artifacts/phase2/learned/transition_cnn.pt", weights_only=True, map_location="cpu"
        )
    )
    legacy.eval()
    with torch.inference_mode():
        old_pred = legacy(torch.tensor(test["pairs"])).argmax(1).numpy()
    binary_truth = (test["ty"] >= 2).astype(int)
    binary_new = (pred >= 2).astype(int)
    old_probe = np.load("artifacts/phase2/learned/shot_size_probe.npz")
    old_size = old_probe["classes"][
        (test["xf"] @ old_probe["coef"].T + old_probe["intercept"]).argmax(1)
    ]
    from transformers import CLIPModel, CLIPProcessor

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
        text_features = clip.get_text_features(
            **processor(text=descriptions, return_tensors="pt", padding=True)
        ).numpy()
    text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)
    zero_size = (test["xf"] @ text_features.T).argmax(1)
    new_size = model.predict(scaler.transform(test["xf"]))
    # Keep paired predictions so errors can be traced back to source videos.
    records = []
    transition_offset = size_offset = 0
    for key, part in split.items():
        if part != "test":
            continue
        d = np.load(a.features / (key + ".npz"))
        for i, truth in enumerate(d["transition_labels"]):
            j = transition_offset + i
            records.append(
                {
                    "video_id": key,
                    "module": "transitions",
                    "index": i,
                    "reference": int(truth),
                    "prediction": int(pred[j]),
                    "old_binary_prediction": int(old_pred[j]),
                }
            )
        for i, truth in enumerate(d["size_labels"]):
            j = size_offset + i
            records.append(
                {
                    "video_id": key,
                    "module": "shot_size",
                    "index": i,
                    "reference": int(truth),
                    "prediction": int(new_size[j]),
                    "zero_shot_prediction": int(zero_size[j]),
                    "old_probe_prediction": int(old_size[j]),
                }
            )
        transition_offset += len(d["transition_labels"])
        size_offset += len(d["size_labels"])
    (a.output / "paired_predictions.json").write_text(json.dumps(records, indent=2))
    result = {
        "protocol": protocol,
        "reference_conditioned": True,
        "creator_grouping_verified": False,
        "supports": {
            part: {
                "transitions": np.bincount(d["ty"], minlength=4).tolist(),
                "shot_size": np.bincount(d["sy"], minlength=5).tolist(),
            }
            for part, d in parts.items()
        },
        "transition_candidates": [{k: v for k, v in c.items() if k != "state"} for c in candidates],
        "selected_transition_weighting": chosen["weighting"],
        "transition_four_class": classification_report(
            test["ty"].tolist(), pred.tolist(), list(range(4))
        ),
        "transition_four_class_majority": classification_report(
            test["ty"].tolist(), [int(np.argmax(counts))] * len(test["ty"]), list(range(4))
        ),
        "transition_binary_old": classification_report(
            binary_truth.tolist(), old_pred.tolist(), [0, 1]
        ),
        "transition_binary_new": classification_report(
            binary_truth.tolist(), binary_new.tolist(), [0, 1]
        ),
        "shot_size_old": classification_report(
            test["sy"].tolist(), old_size.tolist(), list(range(5))
        ),
        "shot_size_zero_shot": classification_report(
            test["sy"].tolist(), zero_size.tolist(), list(range(5))
        ),
        "shot_size_new": classification_report(
            test["sy"].tolist(),
            new_size.tolist(),
            list(range(5)),
        ),
        "probe_selection": {"C": best[1], "class_weight": best[2], "validation_macro_f1": best[0]},
    }
    (a.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "transition_four_class",
                    "transition_binary_old",
                    "transition_binary_new",
                    "shot_size_old",
                    "shot_size_new",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
