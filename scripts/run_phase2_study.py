"""Reproducible local sample study; source assets never copied to Git."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from beauty_video_templates.evaluation.metrics import event_f1
from beauty_video_templates.parsing import backends
from beauty_video_templates.parsing.media import frame, metadata, presentation_times


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--heavy-samples", type=int, default=3)
    p.add_argument("--skip-shots", action="store_true")
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    csv_path = next(args.root.rglob("200条视频镜头与转场合并表.csv"))
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        annotations = list(csv.DictReader(f))
    by_id = {}
    for r in annotations:
        by_id.setdefault(r["样本编号"], []).append(r)
    videos = {}
    all_files = list(args.root.rglob("*.mp4"))
    for path in sorted(all_files):
        videos.setdefault(path.stem.split("_")[0], path)
    manifest = []
    for key, path in sorted(videos.items()):
        try:
            video, fps, count = metadata(path)
            manifest.append(
                {
                    "video_id": key,
                    "filename": path.name,
                    "relative_path": str(path.relative_to(args.root)),
                    "duration_seconds": video["duration_seconds"],
                    "fps": fps,
                    "frames": count,
                    "reference_shots": len(by_id.get(key, [])),
                }
            )
        except Exception as exc:
            manifest.append({"video_id": key, "error": str(exc)})
    save(
        out / "inventory.json",
        {
            "mp4_files_including_copies": len(all_files),
            "unique_filename_ids": len(videos),
            "annotation_rows": len(annotations),
            "reference_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "transition_labels": dict(Counter(r["入镜转场类型"] for r in annotations)),
            "shot_size_labels": dict(Counter(r["景别"] for r in annotations)),
            "videos": manifest,
        },
    )
    selected = [key for key in sorted(videos) if key in by_id][: args.limit]
    save(
        out / "study_protocol.json",
        {
            "selected_video_ids": selected,
            "selection": "first sorted IDs; convenience sample, not randomized",
            "reference_status": "provided spreadsheet/CSV; not independently reannotated",
            "boundary_tolerance_frames": 2,
            "excludes_first_and_last_boundary": True,
            "accuracy_claim_scope": "agreement with supplied annotations; not independent ground truth",
            "training": "no training or test tuning in this study",
        },
    )
    results = []
    for key in [] if args.skip_shots else selected:
        path = videos[key]
        reference = [int(r["起始帧"]) for r in by_id[key] if int(float(r["镜头序号"])) > 1]
        if max(int(r["结束帧"]) for r in by_id[key]) + 1 != metadata(path)[2]:
            raise ValueError(f"Reference frame count does not match source: {key}")
        ratios = [
            int(float(r["起始帧"])) / float(r["起始秒"])
            for r in by_id[key]
            if float(r["起始秒"]) > 0
        ]
        reference_fps = sorted(ratios)[len(ratios) // 2] if ratios else metadata(path)[1]
        pts = None
        for backend in ["content", "adaptive"]:
            target = out / "json" / key / f"shots_{backend}.json"
            started = time.perf_counter()
            if target.exists():
                result = json.loads(target.read_text(encoding="utf-8"))
                if result["parameters"].get("timing") != "decoded presentation timestamps":
                    pts = pts or presentation_times(path)[0]
                    duration = result["video"]["duration_seconds"]
                    for item in result["items"]:
                        item["start"] = pts[item["start_frame"]]
                        end = item["end_frame_exclusive"]
                        item["end"] = min(pts[end], duration) if end < len(pts) else duration
                    result["parameters"]["timing"] = "decoded presentation timestamps"
                    save(target, result)
            else:
                result = backends.shots(path, backend=backend)
                save(target, result)
            metrics = event_f1([r["start_frame"] for r in result["items"][1:]], reference, 2)
            results.append(
                {
                    "video_id": key,
                    "module": "shots",
                    "backend": backend,
                    "reference_fps_estimate": reference_fps,
                    "tolerance_seconds": 2 / reference_fps,
                    "evaluation_timebase": "source frame index; verified total frame count",
                    "runtime_or_cache_seconds": time.perf_counter() - started,
                    **metrics,
                }
            )
        save(out / "shot_comparison.json", results)
        print("shots", key, flush=True)
    for key in selected[: args.heavy_samples]:
        path = videos[key]
        shot_result = json.loads(
            (out / "json" / key / "shots_content.json").read_text(encoding="utf-8")
        )
        tasks = [
            ("beats_raw", lambda: backends.beats(path, "raw")),
            ("beats_hpss", lambda: backends.beats(path, "hpss")),
            ("transitions", lambda: backends.transitions(path, shot_result)),
            ("asr_tiny", lambda: backends.asr(path, "tiny", 1)),
            ("asr_tiny_beam5", lambda: backends.asr(path, "tiny", 5)),
            ("ocr", lambda: backends.ocr(path, ("en",), 5)),
            ("shot_size", lambda: backends.shot_size(path, shot_result)),
            ("shot_size_ensemble", lambda: backends.shot_size(path, shot_result, True)),
        ]
        for name, call in tasks:
            target = out / "json" / key / (name + ".json")
            if target.exists():
                continue
            started = time.perf_counter()
            try:
                value = call()
                value["parameters"]["runtime_seconds"] = time.perf_counter() - started
                save(target, value)
                print(name, key, len(value["items"]), flush=True)
            except Exception as exc:
                save(
                    out / "errors" / (key + "_" + name + ".json"),
                    {"error": str(exc), "type": type(exc).__name__},
                )
                print("ERROR", key, name, str(exc), flush=True)
    # Actual frames for visual inspection and failure review.
    from PIL import Image, ImageDraw

    for key in selected[:6]:
        duration = metadata(videos[key])[0]["duration_seconds"]
        canvas = Image.new("RGB", (720, 540), "#101820")
        draw = ImageDraw.Draw(canvas)
        for i, fraction in enumerate([0.05, 0.2, 0.35, 0.5, 0.65, 0.8]):
            t = duration * fraction
            image = Image.fromarray(frame(videos[key], t)[:, :, ::-1])
            image.thumbnail((230, 235))
            x = (i % 3) * 240
            y = (i // 3) * 270
            canvas.paste(image, (x, y + 25))
            draw.text((x + 5, y + 5), f"{key} {t:.2f}s", fill="white")
        target = out / "contact_sheets" / (key + ".jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(target)


if __name__ == "__main__":
    main()
