"""Evaluate JSON outputs against independently supplied references."""

import argparse
import json
from pathlib import Path

from .metrics import box_iou, edit_distance, event_f1, word_error_rate


def evaluate(prediction, reference, tolerance=0.07):
    module = prediction["module"]
    if reference["module"] != module or reference["video"]["id"] != prediction["video"]["id"]:
        raise ValueError("Reference must describe the same video and module")
    p, r = prediction["items"], reference["items"]
    if module in {"shots", "beats"}:
        key = "time" if module == "beats" else "start"
        return event_f1(
            [i[key] for i in (p if module == "beats" else p[1:])],
            [i[key] for i in (r if module == "beats" else r[1:])],
            tolerance,
        )
    if module == "asr":
        hypothesis = " ".join(i["text"] for i in p)
        truth = " ".join(i["text"] for i in r)
        return {
            "wer": word_error_rate(hypothesis, truth),
            "normalization": "exact case-sensitive whitespace tokens; normalize externally if needed",
        }
    if module == "ocr":
        # Greedy descending IoU matching, within overlapping visibility intervals.
        candidates = []
        for i, x in enumerate(p):
            for j, y in enumerate(r):
                if min(x["end"], y["end"]) > max(x["start"], y["start"]):
                    overlap = box_iou(x["bbox"], y["bbox"])
                    if overlap > 0:
                        candidates.append((overlap, i, j))
        used_p = set()
        used_r = set()
        pairs = []
        for overlap, i, j in sorted(candidates, reverse=True):
            if i not in used_p and j not in used_r:
                used_p.add(i)
                used_r.add(j)
                pairs.append((overlap, i, j))
        errors = sum(edit_distance(r[j]["text"], p[i]["text"]) for _, i, j in pairs)
        errors += sum(len(y["text"]) for j, y in enumerate(r) if j not in used_r)
        errors += sum(len(x["text"]) for i, x in enumerate(p) if i not in used_p)
        chars = sum(len(y["text"]) for y in r)
        return {
            "matched_boxes": len(pairs),
            "predicted_boxes": len(p),
            "reference_boxes": len(r),
            "mean_matched_iou": sum(t[0] for t in pairs) / len(pairs) if pairs else None,
            "character_error_rate_including_unmatched": errors / chars if chars else None,
            "matching": "greedy descending IoU on temporally overlapping intervals; use aligned frame samples",
        }
    # Reference-conditioned interval labels. Missed candidates reported separately.
    matched = []
    used = set()
    for x in p:
        candidates = [
            (abs(x["start"] - y["start"]), j, y)
            for j, y in enumerate(r)
            if j not in used and abs(x["start"] - y["start"]) <= tolerance
        ]
        if candidates:
            _, j, y = min(candidates, key=lambda item: item[0])
            used.add(j)
            matched.append((y["label"], x["label"]))
    labels = sorted({value for pair in matched for value in pair})
    matrix = [[sum(a == x and b == y for a, b in matched) for y in labels] for x in labels]
    return {
        "matched_accuracy": sum(x == y for x, y in matched) / len(matched) if matched else None,
        "matched": len(matched),
        "unmatched_predictions": len(p) - len(matched),
        "unmatched_references": len(r) - len(matched),
        "labels": labels,
        "confusion_matrix": matrix,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prediction", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument(
        "--tolerance", type=float, default=0.07, help="Seconds; for shots use 2/reference_FPS"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.tolerance < 0:
        p.error("Tolerance must be nonnegative")
    result = evaluate(
        json.loads(args.prediction.read_text(encoding="utf-8")),
        json.loads(args.reference.read_text(encoding="utf-8")),
        args.tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
