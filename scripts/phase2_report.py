"""Build measured comparison tables and a standalone local review report."""

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((a.study / "inventory.json").read_text(encoding="utf-8"))
    scores = json.loads((a.study / "shot_comparison.json").read_text(encoding="utf-8"))
    references = list(
        csv.DictReader(
            next(a.root.rglob("200条视频镜头与转场合并表.csv")).open(encoding="utf-8-sig")
        )
    )
    rows = []
    from beauty_video_templates.evaluation.metrics import event_f1

    for target in (a.study / "json").glob("*/shots_transnet.json"):
        result = json.loads(target.read_text(encoding="utf-8"))
        vid = target.parent.name
        truth = [r for r in references if r["样本编号"] == vid and int(r["镜头序号"]) > 1]
        score = event_f1(
            [i["start_frame"] for i in result["items"][1:]],
            [int(r["起始帧"]) for r in truth],
            2,
        )
        scores.append({"video_id": vid, "backend": "transnet", **score})
    for backend in ["content", "adaptive"]:
        subset = [s for s in scores if s["backend"] == backend]
        matched = sum(s["matched"] for s in subset)
        pred = sum(s["predicted"] for s in subset)
        ref = sum(s["reference"] for s in subset)
        precision = matched / pred if pred else 0
        recall = matched / ref if ref else 0
        rows.append(
            {
                "module": "shots",
                "variant": backend,
                "sample_count": len(subset),
                "metric": "micro boundary F1",
                "value": 2 * precision * recall / (precision + recall) if precision + recall else 0,
                "precision": precision,
                "recall": recall,
                "status": "measured supplied-reference agreement",
            }
        )
    labels = {
        "极近景": "extreme_close_up",
        "近景": "close_up",
        "中近景": "medium_shot",
        "中景": "medium_shot",
        "全景": "wide_shot",
        "局部特写": "detail_shot",
        "产品/手部特写": "detail_shot",
    }
    transnet_ids = {s["video_id"] for s in scores if s["backend"] == "transnet"}
    for s in scores:
        if s["video_id"] in transnet_ids:
            rows.append(
                {
                    "module": "shots",
                    "variant": s["backend"] + " paired " + s["video_id"],
                    "sample_count": 1,
                    "metric": "boundary F1 on same comparison video",
                    "value": s["f1"],
                    "precision": s["precision"],
                    "recall": s["recall"],
                    "status": "paired sample, not full-corpus TransNet benchmark",
                }
            )
    learned_path = a.study.parent / "learned" / "learned_comparison.json"
    learned = {}
    if learned_path.exists():
        learned = json.loads(learned_path.read_text(encoding="utf-8"))
        for name in [
            "transition_rules",
            "transition_majority_baseline",
            "transition_cnn",
            "shot_size_linear_probe",
            "shot_size_zero_shot_same_test",
        ]:
            if name in learned:
                value = learned[name]
                rows.append(
                    {
                        "module": "transitions" if name.startswith("transition") else "shot_size",
                        "variant": name,
                        "sample_count": value["n"],
                        "metric": "video-disjoint pilot test accuracy",
                        "value": value["accuracy"],
                        "status": "21/3/6 video split; reference-conditioned; see confusion matrices",
                    }
                )
    detail = []
    size_counts = {}
    transition_counts = Counter()
    for folder in sorted((a.study / "json").iterdir()):
        vid = folder.name
        truth = [r for r in references if r["样本编号"] == vid]
        source_shots = json.loads((folder / "shots_content.json").read_text(encoding="utf-8"))[
            "items"
        ]
        for f in folder.glob("*.json"):
            result = json.loads(f.read_text(encoding="utf-8"))
            detail.append(
                {
                    "video_id": vid,
                    "variant": f.stem,
                    "items": len(result["items"]),
                    "runtime_seconds": result["parameters"].get("runtime_seconds"),
                    "status": "ran successfully",
                }
            )
            if f.stem.startswith("shot_size"):
                for item in result["items"]:
                    detected = next(s for s in source_shots if s["shot_id"] == item["shot_id"])
                    midpoint = (detected["start_frame"] + detected["end_frame_exclusive"] - 1) // 2
                    match = next(
                        (r for r in truth if int(r["起始帧"]) <= midpoint <= int(r["结束帧"])),
                        None,
                    )
                    if match and match["景别"] in labels:
                        counts = size_counts.setdefault(f.stem, [0, 0])
                        counts[0] += item["label"] == labels[match["景别"]]
                        counts[1] += 1
            if f.stem == "transitions":
                for item in result["items"]:
                    boundary = item["boundary_time"]
                    source_frame = min(source_shots, key=lambda s: abs(s["start"] - boundary))[
                        "start_frame"
                    ]
                    matches = [
                        r
                        for r in truth
                        if r["入镜转场序号"] and abs(int(r["起始帧"]) - source_frame) <= 2
                    ]
                    if matches:
                        r = min(matches, key=lambda r: abs(int(r["起始帧"]) - source_frame))
                        actual = (
                            "hard_cut"
                            if r["入镜转场类型"] in {"硬切", "跳切/动作切"}
                            else "non_hard_cut"
                        )
                        transition_counts[(actual, item["label"])] += 1
    for variant, (correct, total) in size_counts.items():
        rows.append(
            {
                "module": "shot_size",
                "variant": variant,
                "sample_count": total,
                "metric": "midpoint-aligned accuracy",
                "value": correct / total,
                "status": "conditional on detected intervals; ambiguous mixed label excluded",
            }
        )
    if transition_counts:
        n = sum(transition_counts.values())
        rows.append(
            {
                "module": "transitions",
                "variant": "rules",
                "sample_count": n,
                "metric": "matched-candidate accuracy",
                "value": sum(v for (x, y), v in transition_counts.items() if x == y) / n,
                "status": "conditional accuracy; unmatched/missed transitions excluded",
            }
        )
    for module, variants in {
        "beats": ["raw", "hpss", "Beat This! small0"],
        "ocr": ["sampled EasyOCR"],
        "asr": ["tiny beam1", "tiny beam5"],
    }.items():
        for variant in variants:
            rows.append(
                {
                    "module": module,
                    "variant": variant,
                    "metric": "accuracy metric",
                    "value": None,
                    "status": "not measurable: no independent reference annotations found",
                }
            )
    fields = [
        "module",
        "variant",
        "sample_count",
        "metric",
        "value",
        "precision",
        "recall",
        "status",
    ]
    with (a.output / "module_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "run_details.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    confusion = [
        {"reference": x, "prediction": y, "count": v} for (x, y), v in transition_counts.items()
    ]
    (a.output / "transition_confusion.json").write_text(
        json.dumps(confusion, indent=2), encoding="utf-8"
    )
    worst = sorted(scores, key=lambda s: s["f1"])[:8]

    def table(records, keys):
        return (
            "<table><tr>"
            + "".join("<th>" + html.escape(k) + "</th>" for k in keys)
            + "</tr>"
            + "".join(
                "<tr>"
                + "".join("<td>" + html.escape(str(r.get(k, ""))) + "</td>" for k in keys)
                + "</tr>"
                for r in records
            )
            + "</table>"
        )

    body = f"""<h1>Phase 2 · Single-modality parsing</h1>
    <p>Local implementation and pilot evaluation. {inventory["unique_filename_ids"]} unique filename IDs;
    {inventory["mp4_files_including_copies"]} files including copies; {inventory["annotation_rows"]} supplied shot records.</p>
    <h2>Measured comparisons</h2>{table(rows, fields)}
    <h2>Interpretation and protocol</h2>
    <p>Shot detection: first 30 sorted video IDs, convenience sample. Compare content and adaptive PySceneDetect without tuning. One-to-one matching uses source frame indices with ±2-frame tolerance. Each video's final annotated frame was verified against its actual frame count. The supplied second timestamps are not reliable for every file: V003 ends at 28.867 seconds in the CSV but 46.867 seconds in the actual media, while its frame indices match. Therefore supplied seconds are not used as boundary ground truth. Output times use decoded presentation timestamps. Scores measure agreement with supplied annotations, whose provenance and inter-annotator reliability have not been independently verified.</p>
    <p>Shot-size labels map to five classes. Mixed “medium/product display” labels are excluded. Reference size is selected by the source-frame midpoint of each detected interval; this is conditional, not end-to-end accuracy. Transition accuracy is conditional on detected boundaries matching a reference within ±2 source frames and is separate from missed-transition recall. Hard cuts and action/jump cuts map to hard_cut; dissolves and black transitions map to non_hard_cut.</p>
    <h2>Failure analysis</h2>
    <p>A late-frame sample in V003 originally failed under OpenCV millisecond seeking. The final sampler uses decoded presentation timestamps, and the affected visual outputs were regenerated. The separate annotation issue (incorrect CSV seconds despite matching source-frame indices) was resolved by frame-index evaluation and frame-indexed training samples, rather than stretching annotations to fit the video.</p>
    <p>Observed in the supplied V001 contact sheet: exposure changes and nearly identical same-person jump cuts. These challenge pixel-difference detectors in different directions. The table below identifies actual low-agreement videos for review; disagreement does not by itself establish which annotation is correct.</p>
    {table(worst, ["video_id", "backend", "precision", "recall", "f1", "predicted", "reference"])}
    <ul><li>Fixed minimum shot length can miss very rapid edits.</li><li>Gradual transitions do not have a single unambiguous hard-cut frame.</li><li>Sampling OCR every five seconds misses short-lived text; returned intervals are sampling bins, not tracked visibility.</li><li>English-only demo OCR does not evaluate Korean or Chinese coverage. The adapter accepts compatible language groups, e.g. ko en or ch_sim en in separate runs.</li><li>Music mixed with narration can produce spurious beats and speech hallucinations. Without independent beat/transcript labels, counts and plausible text are not accuracy evidence.</li><li>CLIP shot-size terminology is prompt-sensitive and may confuse product detail with face framing. Three-frame averaging is a tested variant, not assumed improvement.</li></ul>
    <h2>Learned comparisons</h2><pre>{html.escape(json.dumps(learned, indent=2))}</pre>
    <p>The small binary transition CNN uses two 32×32 frames concatenated as six channels, two convolution layers, global pooling and a linear head. Train for 15 epochs; select by validation macro F1. The shot-size linear probe uses frozen normalized CLIP image embeddings and logistic regression, choosing C from 0.01, 0.1, 1, 10 on validation accuracy. First 30 video IDs split 21/3/6. Each video contributes up to eight evenly spaced reference transitions and four reference shots. No creator/campaign grouping was verified. Binary transition results do not establish the optional four-class target.</p>
    <p><strong>Class-imbalance risk:</strong> The pilot transition test contains 39 hard cuts and only two non-hard cuts. Predicting hard_cut for everything scores 95.12% accuracy. Compare the CNN with that majority baseline and inspect minority recall in the confusion matrix; high accuracy alone is not evidence of useful transition recognition. Shot-size results use only 23 test shots and must not be generalized to the whole dataset.</p>
    <h2>Remaining requirements</h2><p>Independent beat, OCR and speech references are needed before their requested accuracy metrics can be reported. TransNetV2 has only the paired sample coverage shown above. Beat This! output can be compared for event counts and timestamps, not accuracy without reference beats. Neither the small sample nor video-only split establishes broad generalization. No claimed target has been achieved unless measured in the table.</p>
    <h2>Run evidence</h2>{table(detail, ["video_id", "variant", "items", "runtime_seconds", "status"])}
    <h2>Reproduce</h2><pre>python -m pip install -e ".[dev,phase2]"
python scripts/run_phase2_study.py PATH_TO_RESOURCES --output artifacts/phase2/study --limit 30 --heavy-samples 3
python scripts/phase2_report.py artifacts/phase2/study PATH_TO_RESOURCES OUTPUT_FOLDER</pre>
    <h2>Sources</h2><p>Provided 200-video merged CSV (SHA256 {inventory["reference_csv_sha256"]}). Implementation references:
    <a href="https://www.scenedetect.com/docs/latest/api/detectors.html">PySceneDetect</a>,
    <a href="https://github.com/SYSTRAN/faster-whisper">faster-whisper</a>,
    <a href="https://github.com/JaidedAI/EasyOCR">EasyOCR</a>,
    <a href="https://huggingface.co/docs/transformers/model_doc/clip">CLIP</a>.</p>"""
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Phase 2 report</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:48px auto;padding:0 24px;color:#183044}h1,h2{color:#075c63}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{padding:9px;border:1px solid #cddce1;text-align:left}th{background:#e7f3f2}pre{background:#eef3f7;padding:20px;white-space:pre-wrap}</style>'
        + body
        + "</html>"
    )
    (a.output / "Phase_2_Report.html").write_text(document, encoding="utf-8")
    print(a.output / "Phase_2_Report.html")


if __name__ == "__main__":
    main()
