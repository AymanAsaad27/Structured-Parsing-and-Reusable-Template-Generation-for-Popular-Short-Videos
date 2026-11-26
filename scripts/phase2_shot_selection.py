"""Select shot-detector settings on validation, then evaluate 30 untouched test videos."""

import argparse
import csv
import json
from pathlib import Path

from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video

from beauty_video_templates.evaluation.metrics import event_f1


def predict_many(path, configs):
    # SceneManager supplies the same decoded/downscaled frame to all detectors.
    # Content differences are shared; each detector retains its own temporal state.
    calculator = ContentDetector()
    shared = {"frame": None, "score": None}

    class CachedScore:
        def _calculate_frame_score(self, frame_num, frame_img):
            if shared["frame"] != frame_num:
                shared["score"] = calculator._calculate_frame_score(frame_num, frame_img)
                shared["frame"] = frame_num
            return shared["score"]

        def process_frame(self, frame_num, frame_img):
            self.detected.extend(super().process_frame(frame_num, frame_img))
            return []

    class CachedContent(CachedScore, ContentDetector):
        pass

    class CachedAdaptive(CachedScore, AdaptiveDetector):
        pass

    manager = SceneManager()
    detectors = []
    for c in configs:
        detector = (
            CachedContent(threshold=c["threshold"], min_scene_len=c["min_scene_len"])
            if c["backend"] == "content"
            else CachedAdaptive(
                adaptive_threshold=c["threshold"],
                min_scene_len=c["min_scene_len"],
                min_content_val=c.get("min_content_val", 15),
            )
        )
        detector.detected = []
        manager.add_detector(detector)
        detectors.append(detector)
    manager.detect_scenes(video=open_video(str(path)), show_progress=False)
    return [sorted(set(d.detected) - {0}) for d in detectors]


def aggregate(records):
    m, p, r = [sum(x[k] for x in records) for k in ["matched", "predicted", "reference"]]
    return {
        "matched": m,
        "predicted": p,
        "reference": r,
        "micro_f1": 2 * m / (p + r) if p + r else None,
        "macro_video_f1": sum(x["f1"] for x in records) / len(records),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("protocol", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(a.protocol.read_text())
    validation = sorted(k for k, v in protocol["split"].items() if v == "validation")
    prior_test = sorted(k for k, v in protocol["split"].items() if v == "test")[:10]
    test = sorted(k for k, v in protocol["split"].items() if v == "test" and k not in prior_test)
    configs = [
        {"backend": "content", "threshold": t, "min_scene_len": m}
        for t in [18, 22, 27]
        for m in [1, 3, 6]
    ]
    configs += [
        {"backend": "adaptive", "threshold": t, "min_scene_len": m, "min_content_val": c}
        for t in [1.5, 2, 3]
        for m in [1, 3, 6]
        for c in [10, 15]
    ]
    plan = {
        "validation": validation,
        "test": test,
        "excluded_previously_inspected_shot_test": prior_test,
        "configs": configs,
        "selection": "validation micro F1, first declared config breaks ties",
        "tolerance_frames": 2,
    }
    plan_file = a.output / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Do not alter a frozen selection plan")
    plan_file.write_text(json.dumps(plan, indent=2))
    rows = list(
        csv.DictReader(
            next(a.root.rglob("200条视频镜头与转场合并表.csv")).open(encoding="utf-8-sig")
        )
    )
    videos = {}
    for path in sorted(a.root.rglob("*.mp4")):
        videos.setdefault(path.stem.split("_")[0], path)

    def run(ids, variants, folder):
        scores = [[] for _ in variants]
        folder.mkdir(exist_ok=True)
        for key in ids:
            target = folder / (key + ".json")
            if not target.exists():
                predictions = predict_many(videos[key], variants)
                target.write_text(
                    json.dumps({"configs": variants, "predictions": predictions}, indent=2)
                )
            data = json.loads(target.read_text())
            if data["configs"] != variants:
                raise ValueError("Cached configuration mismatch")
            reference = [
                int(r["起始帧"]) for r in rows if r["样本编号"] == key and int(r["起始帧"]) > 0
            ]
            for i, points in enumerate(data["predictions"]):
                scores[i].append({"video_id": key, **event_f1(points, reference, 2)})
            print(folder.name, key, flush=True)
        return scores

    validation_scores = run(validation, configs, a.output / "validation")
    validation_aggregate = [aggregate(x) for x in validation_scores]
    chosen = max(range(len(configs)), key=lambda i: validation_aggregate[i]["micro_f1"])
    selected = {"config": configs[chosen], "validation": validation_aggregate[chosen]}
    (a.output / "selected.json").write_text(json.dumps(selected, indent=2))
    print("Selected", selected, flush=True)
    variants = [
        {"backend": "content", "threshold": 27, "min_scene_len": 6},
        {"backend": "adaptive", "threshold": 3, "min_scene_len": 6, "min_content_val": 15},
        configs[chosen],
    ]
    test_scores = run(test, variants, a.output / "test")
    result = {
        "plan": plan,
        "selection": selected,
        "validation_candidates": validation_aggregate,
        "test": [
            {"config": c, **aggregate(s), "per_video": s} for c, s in zip(variants, test_scores)
        ],
    }
    (a.output / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
