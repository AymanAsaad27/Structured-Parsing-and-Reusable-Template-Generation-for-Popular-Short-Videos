"""Validation-only decision-threshold selection; the reused test is labelled explicitly."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from beauty_video_templates.evaluation.metrics import classification_report
from beauty_video_templates.parsing.learned import transition_model


def predict(logits, threshold):
    if threshold is None:
        return logits.argmax(1).numpy()
    probability = logits.softmax(1).numpy()
    soft = probability[:, 2:].sum(1) >= threshold
    return np.where(soft, probability[:, 2:].argmax(1) + 2, probability[:, :2].argmax(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("features", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((a.features / "protocol.json").read_text())
    plan = {
        "thresholds": [None, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "selection": "validation fixed-four-class macro F1, then validation accuracy; declared order breaks remaining ties",
        "test_scope": "previously measured 40-video test reused for this development comparison; no new blind-test claim",
    }
    plan_path = a.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Threshold plan changed")
    plan_path.write_text(json.dumps(plan, indent=2))
    model = transition_model(a.checkpoint)

    def load_part(part):
        pairs, labels = [], []
        for key, value in protocol["split"].items():
            if value != part:
                continue
            with np.load(a.features / (key + ".npz")) as data:
                pairs.extend(data["pairs"])
                labels.extend(data["transition_labels"])
        with torch.inference_mode():
            logits = model(torch.tensor(np.asarray(pairs), dtype=torch.float32))
        return logits, np.asarray(labels, dtype=np.int64)

    logits, truth = load_part("validation")
    candidates = [
        {
            "threshold": t,
            "validation": classification_report(
                truth.tolist(), predict(logits, t).tolist(), list(range(4))
            ),
        }
        for t in plan["thresholds"]
    ]
    selected = max(
        candidates, key=lambda c: (c["validation"]["macro_f1"], c["validation"]["accuracy"])
    )
    (a.output / "selected.json").write_text(json.dumps(selected, indent=2))
    raw = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    raw["soft_threshold"] = selected["threshold"]
    torch.save(raw, a.output / "transition_four_class_threshold.pt")
    logits, truth = load_part("test")
    predicted = predict(logits, selected["threshold"])
    result = {
        "plan": plan,
        "selected": selected,
        "candidates": candidates,
        "four_class": classification_report(truth.tolist(), predicted.tolist(), list(range(4))),
        "binary": classification_report(
            (truth >= 2).astype(int).tolist(), (predicted >= 2).astype(int).tolist(), [0, 1]
        ),
    }
    (a.output / "results.json").write_text(json.dumps(result, indent=2))
    print(
        "Selected threshold",
        selected["threshold"],
        "test macro F1",
        result["four_class"]["macro_f1"],
        flush=True,
    )


if __name__ == "__main__":
    main()
