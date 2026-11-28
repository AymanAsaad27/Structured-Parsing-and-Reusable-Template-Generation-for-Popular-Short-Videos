"""Reference video -> editable structural template, using local measured features."""

import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from beauty_video_templates.parsing import backends
from beauty_video_templates.parsing.media import frame, metadata, prepare_frames
from beauty_video_templates.semantics.clip_roles import visual_estimates
from beauty_video_templates.templates.demo_template import caption_for, validate_template


def palette(images):
    pixels = []
    for image in images:
        rgb = np.asarray(image.resize((32, 32)))
        pixels.extend(rgb.reshape(-1, 3))
    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(1, -1, 3))
    quantized = image.quantize(colors=4)
    colours = quantized.getpalette()
    return [
        "#" + "".join(f"{v:02x}" for v in colours[3 * i : 3 * i + 3])
        for _, i in sorted(quantized.getcolors(), reverse=True)
    ]


def analyse_reference(
    path, output_dir, model_dir, *, read_text=True, detect_beats=True, progress=None
):
    progress = progress or (lambda value, message: None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info, _, _ = metadata(path)
    if not 1 <= info["duration_seconds"] <= 90:
        raise ValueError("For this demo, choose a reference between 1 and 90 seconds long.")
    progress(8, "Detecting the reference shots")
    shot_result = backends.shots(path, backend="adaptive")
    raw = shot_result["items"]
    if len(raw) > 80:
        raise ValueError(
            "This reference has more than 80 detected shots. Choose a shorter reference for the demo."
        )
    fps = 24
    end_frame = max(24, round(info["duration_seconds"] * fps))
    bounds = sorted(
        {0, end_frame, *[max(0, min(end_frame, round(s["start"] * fps))) for s in raw[1:]]}
    )
    scenes = [
        {
            "start": a / fps,
            "end": b / fps,
            "source_shot": min(raw, key=lambda s: abs(s["start"] - a / fps)),
        }
        for a, b in zip(bounds, bounds[1:])
    ]
    warnings = []
    progress(20, "Reading representative frames")
    times = [
        min(info["duration_seconds"] - 0.001, s["start"] + (s["end"] - s["start"]) * fraction)
        for s in scenes
        for fraction in [0.25, 0.5, 0.75]
    ]
    prepare_frames(path, times)
    images = [Image.fromarray(frame(path, t)[:, :, ::-1]) for t in times]
    for index in range(len(scenes)):
        preview = images[3 * index + 1].copy()
        preview.thumbnail((240, 400))
        preview.save(output_dir / f"shot_{index + 1:03}.jpg", quality=85)
    progress(32, "Estimating visual shot functions")
    estimates = None
    try:
        estimates = visual_estimates(images, model_dir)
    except (FileNotFoundError, ImportError) as error:
        warnings.append(str(error))
    progress(55, "Estimating subtitle placement")
    ocr_result = None
    if read_text:
        try:
            # Wider OCR decoding is optional; the demo uses the quicker baseline.
            ocr_result = backends.ocr(path, languages=("en",), sample_seconds=5)
        except Exception as error:
            warnings.append(
                "Text analysis unavailable; a default editable caption position is used. "
                + type(error).__name__
            )
    progress(72, "Estimating the music pulse")
    beat_result = None
    if detect_beats:
        try:
            beat_result = backends.beats(path, backend="hpss")
        except Exception as error:
            warnings.append("Beat analysis unavailable. " + type(error).__name__)
    detected_text = ocr_result["items"] if ocr_result else []
    boxes = [
        t["bbox"] for t in detected_text if len(t["text"].strip()) >= 3 and t["confidence"] >= 0.35
    ]
    if boxes:
        center_x = statistics.median((b[0] + b[2]) / 2 for b in boxes)
        center_y = statistics.median((b[1] + b[3]) / 2 for b in boxes)
        left = min(0.2, max(0.04, center_x - 0.43))
        top = min(0.80, max(0.05, center_y - 0.07))
        caption_box = [
            round(left, 3),
            round(top, 3),
            round(min(0.96, left + 0.86), 3),
            round(top + 0.14, 3),
        ]
    else:
        caption_box = [0.07, 0.73, 0.93, 0.87]
    slots = []
    for index, scene in enumerate(scenes):
        role_scores, size_scores = {}, {}
        if estimates:
            for group in ["role_scores", "size_scores"]:
                target = role_scores if group == "role_scores" else size_scores
                for label in estimates[3 * index][group]:
                    target[label] = float(
                        np.mean([x[group][label] for x in estimates[3 * index : 3 * index + 3]])
                    )
        visual_role = max(role_scores, key=role_scores.get) if role_scores else "detail_cutaway"
        role = (
            "hook" if index == 0 else "call_to_action" if index == len(scenes) - 1 else visual_role
        )
        if role in {"hook", "call_to_action"} and index not in {0, len(scenes) - 1}:
            role = "product_introduction" if role == "call_to_action" else "presenter_explanation"
        slots.append(
            {
                "id": f"s{index + 1:03}",
                "start": scene["start"],
                "end": scene["end"],
                "role": role,
                "visual_role": visual_role,
                "role_source": "position prior"
                if index in {0, len(scenes) - 1}
                else "CLIP zero-shot estimate"
                if estimates
                else "manual label required",
                "relative_score": round(role_scores.get(visual_role, 0), 3) if estimates else None,
                "shot_size": max(size_scores, key=size_scores.get) if size_scores else "unknown",
                "caption_box": caption_box.copy(),
                "caption": "",
                "transition": "hard_cut",
                "source_start_frame": scene["source_shot"]["start_frame"],
            }
        )
    durations = [s["end"] - s["start"] for s in slots]
    beat_times = [b["time"] for b in beat_result["items"]] if beat_result else []
    median = statistics.median(durations)
    colours = palette(images[1::3])
    template = {
        "schema_version": "1.0-demo",
        "template_id": "reference_" + Path(path).stem[:60],
        "duration_seconds": end_frame / fps,
        "width": 720,
        "height": 1280,
        "fps": fps,
        "style": {
            "pace": "Fast cuts"
            if median < 1.5
            else "Moderate pace"
            if median < 3
            else "Longer holds",
            "median_shot_seconds": round(median, 3),
            "palette": colours,
            "caption_placement": "OCR estimate" if boxes else "editable default",
            "estimated_beat_times": beat_times,
        },
        "provenance": {
            "timing": "PySceneDetect adaptive threshold 3; original presentation timestamps, quantised to export frames",
            "semantics": "Local CLIP zero-shot visual estimates; opening/closing position priors, not a VLM or validated semantic labels",
            "text": "English EasyOCR placement estimate" if ocr_result else "not analysed",
            "transitions": "hard cuts in this prototype; soft-transition style is not reproduced",
            "review_status": "automatic estimates; editable before export",
        },
        "slots": slots,
    }
    validate_template(template)
    (output_dir / "shots.json").write_text(json.dumps(shot_result, indent=2), encoding="utf-8")
    if ocr_result:
        (output_dir / "ocr.json").write_text(
            json.dumps(ocr_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if beat_result:
        (output_dir / "beats.json").write_text(json.dumps(beat_result, indent=2), encoding="utf-8")
    progress(95, "Preparing the editable template")
    return template, warnings


def analyse_material(path, model_dir, thumbnail):
    path = Path(path)
    still = path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    if still:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        duration = None
    else:
        duration = metadata(path)[0]["duration_seconds"]
        image = Image.fromarray(frame(path, duration * 0.5)[:, :, ::-1])
    preview = image.copy()
    preview.thumbnail((200, 300))
    preview.save(thumbnail, quality=84)
    scores = {}
    try:
        scores = visual_estimates([image], model_dir)[0]
    except (FileNotFoundError, ImportError):
        pass
    return {"kind": "image" if still else "video", "duration_seconds": duration, **scores}


def fill_captions(template, product_name, points):
    for i, slot in enumerate(template["slots"]):
        if not slot.get("caption", "").strip():
            slot["caption"] = caption_for(slot["role"], product_name, points, i)
    return template
