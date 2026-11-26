"""Publish measured aggregates and a concise, editable Phase 2 working report."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--public-report", type=Path, default=Path("reports/phase2/improvements"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    a.public_report.mkdir(parents=True, exist_ok=True)
    classifier = load(a.experiment / "classifiers/results.json")
    shots = load(a.experiment / "shot_selection/results.json")
    comparison = load(a.experiment / "boundaries/results.json")
    audit = load(a.experiment / "annotation_audit.json")
    rules = load(a.experiment / "rules/results.json")
    crossvalidation = load(a.experiment / "size_crossvalidation/results.json")
    measured = []
    for name, result in zip(
        ["content-default", "adaptive-default", "adaptive-validation-selected"], shots["test"]
    ):
        measured.append(
            {
                "module": "shots",
                "variant": name,
                "n": len(result["per_video"]),
                "scope": "30 previously uninspected test videos",
                "metric": "boundary micro F1 +/-2 frames",
                "value": result["micro_f1"],
                "macro_f1": None,
            }
        )
    for name in [
        "transition_four_class_majority",
        "transition_four_class",
        "transition_binary_old",
        "transition_binary_new",
        "shot_size_zero_shot",
        "shot_size_old",
        "shot_size_new",
    ]:
        r = classifier[name]
        measured.append(
            {
                "module": "shot_size" if name.startswith("shot_size") else "transitions",
                "variant": name,
                "n": r["n"],
                "scope": "40 test videos; supplied shot boundaries",
                "metric": "accuracy",
                "value": r["accuracy"],
                "macro_f1": r["macro_f1"],
            }
        )
    measured.append(
        {
            "module": "transitions",
            "variant": "transition_binary_rules",
            "n": rules["binary"]["n"],
            "scope": "40 test videos; supplied shot boundaries",
            "metric": "accuracy",
            "value": rules["binary"]["accuracy"],
            "macro_f1": rules["binary"]["macro_f1"],
        }
    )
    for name in ["zero_shot", "probe_oof"]:
        result = crossvalidation[name]
        measured.append(
            {
                "module": "shot_size",
                "variant": "size_cv_" + name,
                "n": result["n"],
                "scope": "supplementary video-separated five-fold development CV on inspected data",
                "metric": "accuracy",
                "value": result["accuracy"],
                "macro_f1": result["macro_f1"],
            }
        )
    with (a.public_report / "module_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)
    for source, name in [
        (a.experiment / "classifiers/results.json", "classifiers.json"),
        (a.experiment / "shot_selection/results.json", "shot_selection.json"),
        (a.experiment / "boundaries/results.json", "three_detector_comparison.json"),
        (a.experiment / "annotation_audit.json", "annotation_audit.json"),
        (a.experiment / "full200/protocol.json", "protocol.json"),
        (a.experiment / "rules/results.json", "rules.json"),
        (a.experiment / "size_crossvalidation/results.json", "size_crossvalidation.json"),
        (a.experiment / "transition_threshold/results.json", "transition_threshold.json"),
    ]:
        shutil.copyfile(source, a.public_report / name)
    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "models": {},
        "packages": {},
    }
    inference_environment = a.experiment / "inference_environment.json"
    if inference_environment.exists():
        manifest["inference_environment"] = load(inference_environment)
    for name in [
        "torch",
        "transformers",
        "scenedetect",
        "scikit-learn",
        "easyocr",
        "faster-whisper",
        "librosa",
        "beat-this",
        "av",
    ]:
        try:
            manifest["packages"][name] = version(name)
        except PackageNotFoundError:
            manifest["packages"][name] = (
                "not installed in report environment; see inference manifest"
            )
    for path in (a.experiment / "classifiers").glob("*"):
        if path.suffix in {".pt", ".npz"}:
            manifest["models"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (a.public_report / "report_manifest.json").write_text(json.dumps(manifest, indent=2))
    best_shot = shots["test"][2]
    text = [
        "PHASE 2 - IMPROVEMENT RESULTS | 26 November 2026",
        "Working report for manual revision before lab submission.",
        "",
        "Current position",
        "All six parsing adapters are implemented. This run expands supervised training to the 200-video collection, repairs evaluation errors, and measures shot, transition and shot-size performance. Beat, OCR and speech accuracy remain unscored until their reviewed reference files are located.",
        "",
        "Data and experiment design",
        f"The supplied table contains {audit['annotation_rows']} shots in {audit['video_ids']} videos: 3,153 transitions and 3,260 unambiguous shot-size labels. All source frame counts and shot intervals match. There are 80 extra byte-identical video copies; one source per video ID is used. Written end times differ by more than 100 ms in 28 videos, so evaluation uses source frames and decoded timestamps.",
        "The frozen split is 140 training / 20 validation / 40 test videos. The previously inspected V001-V030 are training-only. Creator/campaign grouping is unavailable. Classification uses annotated boundaries and midpoints, so these scores do not include shot-detection errors. Ninety-three mixed 'medium/product presentation' labels are excluded rather than silently relabelled.",
        "The first ten sorted test IDs were used for the three-detector comparison. Shot settings were then selected using only the 20 validation videos; the remaining 30 test videos provide the primary untouched shot-detection comparison. The four-class transition model and shot-size probe use all 40 classification test videos.",
        "Medium and wide labels occur only in the already inspected first 30 videos, leaving zero examples of those classes in the new validation/test sets. A supplementary five-fold video-separated development assessment covers all 3,260 labelled shots with the probe's C=0.01 frozen from validation. It is not a fresh external test.",
        "",
        "Measured results",
    ]
    for r in measured:
        extra = f"; macro F1 {r['macro_f1']:.3f}" if r["macro_f1"] is not None else ""
        text.append(f"- {r['variant']}: {r['metric']} {r['value']:.3f}{extra}; n={r['n']}.")
    text += [
        f"The selected shot configuration is {json.dumps(shots['selection']['config'])}. Its 30-video boundary F1 is {best_shot['micro_f1']:.3f}; the brief's reference target is 0.85.",
        "The validation-selected threshold does not beat default adaptive detection on the 30-video test. Retain adaptive threshold 3, minimum scene length 6 and minimum content score 15 as the current default. The threshold-2 candidate improves recall at a precision cost; it is recorded, not promoted.",
        "The four-class transition CNN reaches 87.1% accuracy versus 85.5% for the majority baseline, but dissolve recall is only 1/26 (3.8%). Validation-selected soft-decision threshold tuning retained ordinary argmax, so it did not fix this failure. The unchanged binary rule has higher macro F1 (0.591) than the collapsed CNN (0.535), despite lower accuracy. Keep both comparisons visible.",
        "Supplementary shot-size CV improves accuracy from 36.4% zero-shot to 61.6% with the probe. Medium-shot recall is 2/67 (3.0%) and wide-shot recall is 0/47; the probe is not yet reliable across the full taxonomy. Full-frame-preserving preprocessing and consistent medium/wide labelling are the next development priorities.",
        "On the separate ten-video comparison, content/adaptive/TransNetV2 boundary F1 is "
        + "/".join(
            f"{comparison['aggregate'][k]['micro_f1']:.3f}"
            for k in ["content", "adaptive", "transnet"]
        )
        + ". TransNetV2 is not the preferred detector for this collection on these measurements.",
        "",
        "What changed",
        "1. Fixed taxonomy-wide macro F1 and added per-class support/recall/confusion matrices. High accuracy can no longer conceal an all-hard-cut classifier.",
        "2. Expanded from a sparse 30-video pilot to all annotated transitions and unambiguous shot-size midpoints. CNN class weighting and probe regularization are selected on validation; the scaler is fitted on training only.",
        "3. Added labelled four-class CNN checkpoints and standardized CLIP probes while preserving legacy inference. Feature extraction is resumable, sharded and writes completed files atomically.",
        "4. Fixed OCR matching across different sampled frames; missed/spurious text contributes to CER, and missed boxes contribute to reference-averaged IoU. Empty speech references no longer produce a misleading perfect WER.",
        "5. Added explicit detector settings, OCR wider-frame/beam-search options, ASR language/decoding options and selectable Beat This! checkpoints.",
        "",
        "Remaining evaluation",
        "Beats: score librosa raw/HPSS and Beat This! against reviewed beat timestamps (70 ms tolerance). OCR: score baseline and enhanced outputs on identical reviewed frames, including boxes/text. Speech: compare exact-token WER for the baseline and beam-5 outputs using reviewed transcripts; document any text normalization. Candidate outputs and configuration changes are not accuracy measurements.",
        "The OCR example uses V003 at 5-second sampling with English/Spanish recognition. Baseline/enhanced outputs contain 33/57 boxes across the sampled video; additional boxes can be correct detections or false positives. ASR examples use the cached tiny Whisper model. The small checkpoint download failed because of network/TLS errors; no small-model result is claimed.",
        "",
        "Findings and next steps",
        "Inspect the confusion matrices before choosing a classifier. The collection has only six black-transition examples, 47 wide shots and 67 medium/medium-close shots. The black class therefore has very little training and test evidence. Add reviewed examples for these classes and check the distinction between face extreme close-ups and product/detail shots.",
        "V056 illustrates missed edits with similar face/background composition and a persistent inset. The displayed frames show why whole-image change can be weak. Inspect the short clip around ambiguous boundaries before changing labels or adding specialized motion logic.",
        "Finish beat/OCR/speech evaluation using the existing reviewed files, then update the module table and failure analysis. Phase 2 should be signed off after those measurements and the lab's review of remaining errors. Phase 3/4 work can then use the chosen detector and trained probes.",
        "",
        "Validation",
        "32 automated tests, repository lint/format checks and package build passed during this improvement run. Private videos, large pretrained weights and credentials remain outside Git. Full experiment configurations, supports and confusion matrices accompany this report.",
    ]
    body = "\n".join(text) + "\n"
    (a.output / "Phase_2_Results.txt").write_text(body, encoding="utf-8")
    headings = {
        "Current position",
        "Data and experiment design",
        "Measured results",
        "What changed",
        "Remaining evaluation",
        "Findings and next steps",
        "Validation",
    }
    markdown = (
        "\n".join(
            "# " + line if i == 0 else "\n## " + line + "\n" if line in headings else line
            for i, line in enumerate(text)
        )
        + "\n"
    )
    (a.public_report / "RESULTS.md").write_text(markdown, encoding="utf-8")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportBody", fontName="Helvetica", fontSize=9.3, leading=13, spaceAfter=7
        )
    )
    styles.add(ParagraphStyle(name="SmallReport", fontName="Helvetica", fontSize=8, leading=10))
    story = []

    def para(value, style="ReportBody"):
        return Paragraph(escape(value), styles[style])

    def table(rows, widths):
        header = ParagraphStyle(
            name="HeaderCell", parent=styles["SmallReport"], textColor=colors.white
        )
        formatted = [
            [
                Paragraph(escape(str(cell)), header if i == 0 else styles["SmallReport"])
                for cell in row
            ]
            for i, row in enumerate(rows)
        ]
        result = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#f0f4f8"), colors.white],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#16324f")),
                ]
            )
        )
        return result

    story += [
        para("Phase 2 | Improvement results", "Title"),
        para("26 November 2026 - Working report for manual revision", "SmallReport"),
        Spacer(1, 12),
        para("Current position", "Heading2"),
        para(text[text.index("Current position") + 1]),
        para("Measured comparisons", "Heading2"),
    ]
    rows = [["Method", "Metric / scope", "Result", "Macro F1"]]
    method_names = {
        "content-default": "PySceneDetect content",
        "adaptive-default": "PySceneDetect adaptive",
        "adaptive-validation-selected": "Adaptive, validation-selected",
        "transition_four_class_majority": "Four-class majority baseline",
        "transition_four_class": "Four-class CNN",
        "transition_binary_old": "Pilot CNN, binary",
        "transition_binary_new": "New CNN, binary collapse",
        "transition_binary_rules": "Frame-difference rules, binary",
        "shot_size_zero_shot": "CLIP zero-shot, test",
        "shot_size_old": "Pilot CLIP probe, test",
        "shot_size_new": "New CLIP probe, test",
        "size_cv_zero_shot": "CLIP zero-shot, grouped CV",
        "size_cv_probe_oof": "CLIP probe, grouped CV",
    }
    for r in measured:
        rows.append(
            [
                method_names[r["variant"]],
                f"{r['metric']} (n={r['n']})",
                f"{r['value']:.3f}",
                "-" if r["macro_f1"] is None else f"{r['macro_f1']:.3f}",
            ]
        )
    story += [
        table(rows, [175, 190, 57, 65]),
        Spacer(1, 10),
        para(
            "Shot scores use 30 test videos and +/-2 source frames. Held-out classification uses 40 videos; supplementary CV uses all 200 inspected videos in separate video folds. Classification uses supplied boundaries. Binary and four-class accuracies measure different tasks."
        ),
        para(
            "Beats, OCR and speech: outputs are available; accuracy scores await the reviewed reference files."
        ),
        para("Data checks", "Heading2"),
        para(text[text.index("Data and experiment design") + 1]),
    ]
    story += [
        PageBreak(),
        para("Class balance and remaining errors", "Title"),
        para(
            "Rows are references; columns are predictions. Transitions use the 40-video test. Shot size uses supplementary video-separated development CV to cover all five labels."
        ),
        para("Transitions: four classes", "Heading2"),
    ]
    for key, labels in [
        ("transition_four_class", ["hard", "jump", "dissolve", "black"]),
        ("shot_size_new", ["extreme close", "close", "medium", "wide", "detail"]),
    ]:
        if key == "shot_size_new":
            story += [Spacer(1, 16), para("Shot size: supplementary grouped CV", "Heading2")]
        result = crossvalidation["probe_oof"] if key == "shot_size_new" else classifier[key]
        rows = [["Reference / predicted", *labels]] + [
            [label, *row] for label, row in zip(labels, result["confusion_matrix"])
        ]
        story.append(table(rows, [137] + [350 / len(labels)] * len(labels)))
        recalls = "; ".join(
            f"{label}: {c['recall']:.0%} (n={c['support']})"
            for label, c in zip(labels, result["per_class"])
        )
        story += [Spacer(1, 6), para("Per-class recall - " + recalls)]
    story += [
        Spacer(1, 10),
        para("Interpretation", "Heading2"),
        para(
            "Class weighting helps the model attend to minority classes; it does not create additional examples. Compare the four-class CNN with the majority baseline as well as overall accuracy. The six black-transition examples are insufficient to establish robust rare-class performance."
        ),
        para(
            "Next annotation priority: medium/wide shots, black transitions and visually subtle same-scene edits. Keep the current mixed-label exclusions visible in future comparisons."
        ),
        para(
            "Dissolve recall is 1/26 on the test set. In grouped size CV, medium recall is 2/67 and wide recall is 0/47. Higher overall accuracy does not resolve these failures; temporal transition context and full-frame size features need further development."
        ),
    ]
    story += [
        PageBreak(),
        para("Input and output example", "Title"),
        para(
            "V056: two annotated boundaries missed by the default adaptive detector. Each row shows source frames two frames before and after the reference boundary, and the nearest predicted cut."
        ),
        Image(str(a.experiment / "failure_examples.png"), width=470, height=470 * 980 / 900),
        para(
            "The first pair retains a similar face/background and a persistent inset. The second pair changes the visible action within a similar portrait composition. These examples motivate clip-level inspection of same-scene edits; they do not establish that the reviewed reference is wrong."
        ),
    ]
    story += [
        PageBreak(),
        para("Completion and reproduction", "Title"),
        para("Completed in this run", "Heading2"),
    ]
    for line in text[text.index("What changed") + 1 : text.index("Remaining evaluation") - 1]:
        if line:
            story.append(para(line))
    story += [
        para("Still required for Phase 2 sign-off", "Heading2"),
        para(text[text.index("Remaining evaluation") + 1]),
        para(
            "Reviewed shot/transition/shot-size tables were used. The remaining files needed are beat timestamps, frame-aligned OCR boxes/text and speech transcripts. No new annotation campaign is assumed."
        ),
        para("Reproduction", "Heading2"),
        para(
            "See docs/phase2_usage.md and the accompanying protocol/configuration JSON. Re-run annotation audit, feature extraction, classifier training, detector comparison and validation selection. Install the optional reporting dependency to regenerate this PDF."
        ),
        para("Checks", "Heading2"),
        para(text[-1]),
        para("Limits", "Heading2"),
        para(
            "Video IDs are separated across splits; creator/campaign separation is not verified. Classification is reference-conditioned. Exact reviewer provenance and inter-annotator agreement were not supplied. The larger ASR checkpoint could not be downloaded, so the new speech examples use tiny-model decoding changes."
        ),
    ]
    pdf = a.output / "Phase_2_Improvement_Report.pdf"

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526475"))
        canvas.drawString(54, 28, "Beauty Video Templates | Phase 2 working results")
        canvas.drawRightString(541, 28, str(doc.page))

    SimpleDocTemplate(
        str(pdf),
        pagesize=(595.28, 841.89),
        rightMargin=54,
        leftMargin=54,
        topMargin=42,
        bottomMargin=44,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(pdf, flush=True)


if __name__ == "__main__":
    main()
