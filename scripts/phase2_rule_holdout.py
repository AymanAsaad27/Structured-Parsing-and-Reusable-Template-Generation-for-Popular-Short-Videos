"""Evaluate the unchanged frame/histogram rule on the full classification test set."""

import argparse
import csv
import json
from pathlib import Path

from beauty_video_templates.evaluation.metrics import classification_report
from beauty_video_templates.parsing.backends import transition_features
from beauty_video_templates.parsing.media import metadata, prepare_indexed_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("protocol", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(a.protocol.read_text())
    ids = sorted(k for k, v in protocol["split"].items() if v == "test")
    rows = list(
        csv.DictReader(
            next(a.root.rglob("200条视频镜头与转场合并表.csv")).open(encoding="utf-8-sig")
        )
    )
    videos = {}
    for path in sorted(a.root.rglob("*.mp4")):
        videos.setdefault(path.stem.split("_")[0], path)
    records = []
    for key in ids:
        target = a.output / (key + ".json")
        if not target.exists():
            selected = [r for r in rows if r["样本编号"] == key and r["入镜转场类型"] != "开场直入"]
            indices = [int(r["起始帧"]) for r in selected]
            count = metadata(videos[key])[2]
            pts = prepare_indexed_frames(
                videos[key],
                [min(max(0, i + o), count - 1) for i in indices for o in [-3, -1, 0, 2]],
            )
            features = transition_features(videos[key], [pts[i] for i in indices], indices, pts)
            pred = [0 if x[1] > max(x[0], x[2]) * 1.5 and x[1] > 0.08 else 1 for x in features]
            data = [
                {
                    "video_id": key,
                    "start_frame": i,
                    "reference": 0 if r["入镜转场类型"] in {"硬切", "跳切/动作切"} else 1,
                    "prediction": value,
                }
                for i, r, value in zip(indices, selected, pred)
            ]
            target.write_text(json.dumps(data, indent=2))
        records.extend(json.loads(target.read_text()))
        print(key, flush=True)
    report = {
        "scope": "reference-conditioned; 40 video-disjoint test IDs",
        "parameters": {"offsets": [-3, -1, 0, 2], "ratio": 1.5, "difference": 0.08},
        "binary": classification_report(
            [r["reference"] for r in records], [r["prediction"] for r in records], [0, 1]
        ),
    }
    (a.output / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
