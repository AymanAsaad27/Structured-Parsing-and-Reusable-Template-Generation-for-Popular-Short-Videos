"""Evaluate JSON outputs against independently supplied references."""

import argparse
import json
from pathlib import Path

from .metrics import edit_distance, event_f1, match_boxes, word_error_rate


def evaluate(prediction, reference, tolerance=0.07, timebase="seconds"):
    module = prediction["module"]
    if reference["module"] != module or reference["video"]["id"] != prediction["video"]["id"]:
        raise ValueError("Reference must describe the same video and module")
    p, r = prediction["items"], reference["items"]
    if module in {"shots", "beats"}:
        if timebase == "frames" and module != "shots":
            raise ValueError("Frame-index evaluation applies only to shot boundaries")
        key = "time" if module == "beats" else "start_frame" if timebase == "frames" else "start"
        return event_f1(
            [i[key] for i in (p if module == "beats" else p[1:])],
            [i[key] for i in (r if module == "beats" else r[1:])],
            tolerance,
        )
    if module == "asr":
        hypothesis = " ".join(i["text"] for i in p)
        truth = " ".join(i["text"] for i in r)
        return {
            "wer": word_error_rate(hypothesis, truth) if truth.split() else None,
            "reference_words": len(truth.split()),
            "hypothesis_words": len(hypothesis.split()),
            "normalization": "exact case-sensitive whitespace tokens; normalize externally if needed",
        }
    if module == "ocr":
        pairs = match_boxes(p, r)
        used_p = {i for _, i, _ in pairs}
        used_r = {j for _, _, j in pairs}
        errors = sum(edit_distance(r[j]["text"], p[i]["text"]) for _, i, j in pairs)
        errors += sum(len(y["text"]) for j, y in enumerate(r) if j not in used_r)
        errors += sum(len(x["text"]) for i, x in enumerate(p) if i not in used_p)
        chars = sum(len(y["text"]) for y in r)
        cer = errors / chars if chars else None
        iou_sum = sum(t[0] for t in pairs)
        return {
            "matched_boxes": len(pairs),
            "predicted_boxes": len(p),
            "reference_boxes": len(r),
            "box_precision": len(pairs) / len(p) if p else None,
            "box_recall": len(pairs) / len(r) if r else None,
            "box_f1": 2 * len(pairs) / (len(p) + len(r)) if p or r else None,
            "mean_matched_iou": iou_sum / len(pairs) if pairs else None,
            "mean_reference_iou_including_misses": iou_sum / len(r) if r else None,
            "character_error_rate_including_unmatched": cer,
            "text_accuracy_one_minus_cer_clipped": max(0.0, 1 - cer) if cer is not None else None,
            "matching": "maximum-cardinality frame-aligned matching, IoU >=0.5, sample timestamps within 1ms",
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
    p.add_argument("--tolerance", type=float, default=None, help="Default .07 seconds or 2 frames")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timebase", choices=["seconds", "frames"], default="seconds")
    args = p.parse_args()
    if args.tolerance is None:
        args.tolerance = 2 if args.timebase == "frames" else 0.07
    if args.tolerance < 0:
        p.error("Tolerance must be nonnegative")
    result = evaluate(
        json.loads(args.prediction.read_text(encoding="utf-8")),
        json.loads(args.reference.read_text(encoding="utf-8")),
        args.tolerance,
        args.timebase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
