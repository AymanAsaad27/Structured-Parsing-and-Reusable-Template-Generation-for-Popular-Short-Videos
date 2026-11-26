"""Compare unchanged detectors on a predeclared subset of the new held-out videos."""

import argparse
import csv
import json
from pathlib import Path

from beauty_video_templates.evaluation.metrics import event_f1
from beauty_video_templates.parsing.backends import shots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("protocol", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--backends", nargs="+", default=["content", "adaptive", "transnet"])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(a.protocol.read_text())
    ids = sorted(k for k, v in protocol["split"].items() if v == "test")[: a.limit]
    plan = {
        "ids": ids,
        "selection": "first sorted test IDs, fixed before predictions",
        "tolerance_frames": 2,
        "backends": a.backends,
    }
    plan_path = a.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Do not change a frozen comparison plan")
    plan_path.write_text(json.dumps(plan, indent=2))
    rows = list(
        csv.DictReader(
            next(a.root.rglob("200条视频镜头与转场合并表.csv")).open(encoding="utf-8-sig")
        )
    )
    videos = {}
    for path in sorted(a.root.rglob("*.mp4")):
        videos.setdefault(path.stem.split("_")[0], path)
    results = []
    for key in ids:
        reference = sorted(
            int(r["起始帧"]) for r in rows if r["样本编号"] == key and int(r["起始帧"]) > 0
        )
        for backend in a.backends:
            cached = a.output / f"{key}_{backend}.json"
            if not cached.exists():
                prediction = shots(videos[key], backend=backend)
                cached.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
            prediction = json.loads(cached.read_text(encoding="utf-8"))
            points = [i["start_frame"] for i in prediction["items"][1:]]
            result = {"video_id": key, "backend": backend, **event_f1(points, reference, 2)}
            results.append(result)
            print(key, backend, round(result["f1"], 4), flush=True)
            (a.output / "per_video.json").write_text(json.dumps(results, indent=2))
    aggregate = {}
    for backend in a.backends:
        subset = [r for r in results if r["backend"] == backend]
        matched, predicted, reference = [
            sum(r[k] for r in subset) for k in ["matched", "predicted", "reference"]
        ]
        aggregate[backend] = {
            "matched": matched,
            "predicted": predicted,
            "reference": reference,
            "micro_f1": 2 * matched / (predicted + reference),
            "macro_video_f1": sum(r["f1"] for r in subset) / len(subset),
        }
    (a.output / "results.json").write_text(
        json.dumps({"plan": plan, "aggregate": aggregate}, indent=2)
    )


if __name__ == "__main__":
    main()
