"""Validation-selected development experiments; the historical pilot test is not fresh."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from train_phase2_comparisons import TransitionCNN

from beauty_video_templates.evaluation.metrics import classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/phase2/learned/training_features_source_frames.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/phase2/improvements/development")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    d = np.load(args.cache)
    torch.set_num_threads(2)

    def mask(groups, lo, hi):
        return np.array([lo <= int(str(g)[1:]) <= hi for g in groups])

    train = mask(d["groups"], 1, 21)
    val = mask(d["groups"], 22, 24)
    test = mask(d["groups"], 25, 30)
    x = torch.tensor(d["pairs"], dtype=torch.float32)
    y = torch.tensor(d["targets"], dtype=torch.long)
    counts = np.bincount(d["targets"][train], minlength=2)
    results = {
        "scope": "development regression only: historical pilot test has already been inspected",
        "selection": "class weighting/regularization selected on validation only; no test-target optimization",
        "transition_train_support": counts.tolist(),
        "fixed_transition_labels": [0, 1],
        "fixed_size_labels": [0, 1, 2, 3, 4],
    }
    candidates = []
    for weighting in ["none", "sqrt_inverse", "inverse"]:
        torch.manual_seed(42)
        net = TransitionCNN()
        optim = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.01)
        weights = (
            np.ones(2)
            if weighting == "none"
            else (counts.sum() / np.maximum(counts, 1))
            ** (0.5 if weighting == "sqrt_inverse" else 1)
        )
        weight = torch.tensor(weights / weights.mean(), dtype=torch.float32)
        best = (-1, float("-inf"))
        state = None
        best_epoch = 0
        for epoch in range(100):
            net.train()
            optim.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(x[train]), y[train], weight=weight)
            loss.backward()
            optim.step()
            net.eval()
            with torch.inference_mode():
                logits = net(x[val])
                pred = logits.argmax(1).numpy()
                nll = float(torch.nn.functional.cross_entropy(logits, y[val]))
            score = classification_report(d["targets"][val].tolist(), pred.tolist(), [0, 1])[
                "macro_f1"
            ]
            if (score, -nll) > best:
                best = (score, -nll)
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                best_epoch = epoch + 1
        candidates.append(
            {
                "weighting": weighting,
                "validation_macro_f1": best[0],
                "validation_nll": -best[1],
                "epoch": best_epoch,
                "state": state,
            }
        )
    chosen = max(candidates, key=lambda c: (c["validation_macro_f1"], -c["validation_nll"]))
    selected = TransitionCNN()
    selected.load_state_dict(chosen["state"])
    selected.eval()
    with torch.inference_mode():
        pred = selected(x[test]).argmax(1).numpy()
    results["transition_candidates"] = [
        {k: v for k, v in c.items() if k != "state"} for c in candidates
    ]
    results["selected_transition_weighting"] = chosen["weighting"]
    results["transition_development_result"] = classification_report(
        d["targets"][test].tolist(), pred.tolist(), [0, 1]
    )
    torch.save(chosen["state"], args.output / "transition_cnn.pt")
    st = mask(d["size_groups"], 1, 21)
    sv = mask(d["size_groups"], 22, 24)
    se = mask(d["size_groups"], 25, 30)
    scaler = StandardScaler().fit(d["features"][st])
    features = scaler.transform(d["features"])
    probes = []
    for c in [0.001, 0.01, 0.1, 1, 10]:
        for weighting in [None, "balanced"]:
            model = LogisticRegression(
                C=c, class_weight=weighting, max_iter=3000, random_state=42
            ).fit(features[st], d["sizes"][st])
            val_pred = model.predict(features[sv])
            score = classification_report(
                d["sizes"][sv].tolist(), val_pred.tolist(), list(range(5))
            )["macro_f1"]
            probes.append((score, c, weighting, model))
    best = max(probes, key=lambda c: c[0])
    model = best[3]
    results["probe_selection"] = {
        "C": best[1],
        "class_weight": best[2],
        "validation_macro_f1": best[0],
    }
    results["probe_development_result"] = classification_report(
        d["sizes"][se].tolist(), model.predict(features[se]).tolist(), list(range(5))
    )
    np.savez(
        args.output / "shot_size_probe.npz",
        coef=model.coef_,
        intercept=model.intercept_,
        classes=model.classes_,
        mean=scaler.mean_,
        scale=scaler.scale_,
    )
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in results.items() if k != "transition_candidates"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
