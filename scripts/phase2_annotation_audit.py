"""Audit annotation indexing, taxonomy coverage, and duplicate source identities."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from beauty_video_templates.parsing.media import metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    source = next(a.root.rglob("200条视频镜头与转场合并表.csv"))
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig")))
    files = defaultdict(list)
    for path in sorted(a.root.rglob("*.mp4")):
        files[path.stem.split("_")[0]].append(path)
    report = {
        "annotation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "annotation_rows": len(rows),
        "video_ids": len({r["样本编号"] for r in rows}),
        "transition_labels": dict(Counter(r["入镜转场类型"] for r in rows)),
        "shot_size_labels": dict(Counter(r["景别"] for r in rows)),
        "per_video": [],
        "duplicate_groups": [],
    }
    for key in sorted({r["样本编号"] for r in rows}):
        selected = sorted((r for r in rows if r["样本编号"] == key), key=lambda r: int(r["起始帧"]))
        video, fps, count = metadata(files[key][0])
        discontinuities = [
            (int(x["结束帧"]), int(y["起始帧"]))
            for x, y in zip(selected, selected[1:])
            if int(x["结束帧"]) + 1 != int(y["起始帧"])
        ]
        report["per_video"].append(
            {
                "video_id": key,
                "source_frame_count": count,
                "annotation_end_frame_inclusive": int(selected[-1]["结束帧"]),
                "frame_count_matches": int(selected[-1]["结束帧"]) + 1 == count,
                "starts_at_zero": int(selected[0]["起始帧"]) == 0,
                "discontinuities": discontinuities,
                "source_duration_seconds": video["duration_seconds"],
                "annotated_end_seconds": float(selected[-1]["结束秒"]),
                "seconds_difference": float(selected[-1]["结束秒"]) - video["duration_seconds"],
                "fps_metadata": fps,
            }
        )
        if len(files[key]) > 1:
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files[key]]
            report["duplicate_groups"].append(
                {"video_id": key, "copies": len(hashes), "byte_identical": len(set(hashes)) == 1}
            )
    report["summary"] = {
        "frame_count_mismatches": [
            r["video_id"] for r in report["per_video"] if not r["frame_count_matches"]
        ],
        "noncontiguous_annotations": [
            r["video_id"]
            for r in report["per_video"]
            if r["discontinuities"] or not r["starts_at_zero"]
        ],
        "seconds_mismatches_over_100ms": [
            r["video_id"] for r in report["per_video"] if abs(r["seconds_difference"]) > 0.1
        ],
        "duplicate_extra_files": sum(r["copies"] - 1 for r in report["duplicate_groups"]),
        "all_duplicate_groups_byte_identical": all(
            r["byte_identical"] for r in report["duplicate_groups"]
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
