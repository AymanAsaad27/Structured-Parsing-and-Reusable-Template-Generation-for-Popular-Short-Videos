"""Phase 2 adapters. Optional models are loaded only when requested."""

import numpy as np

from .media import audio, envelope, frame, metadata, presentation_times


def shots(path, backend="content", threshold=None, min_scene_len=6):
    if backend == "transnet":
        from .transnet import parse

        return parse(path)
    from scenedetect import AdaptiveDetector, ContentDetector, detect

    if backend not in {"content", "adaptive"}:
        raise ValueError("Shot backend must be content or adaptive")
    threshold = threshold if threshold is not None else (27.0 if backend == "content" else 3.0)
    detector = (
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
        if backend == "content"
        else AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len)
    )
    video, fps, count = metadata(path)
    scenes = detect(str(path), detector, show_progress=False)
    cuts = sorted({int(start.get_frames()) for start, _ in scenes if start.get_frames() > 0})
    bounds = [0] + [c for c in cuts if c < count] + [count]
    times, _ = presentation_times(path)
    duration = video["duration_seconds"]
    items = [
        {
            "start": min(times[a], duration),
            "end": min(times[b], duration) if b < len(times) else duration,
            "start_frame": a,
            "end_frame_exclusive": b,
            "shot_id": i,
        }
        for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
    ]
    return envelope(
        path,
        "shots",
        "pyscenedetect-" + backend,
        {
            "threshold": threshold,
            "min_scene_len_frames": min_scene_len,
            "fps": fps,
            "timing": "decoded presentation timestamps",
        },
        items,
    )


def beats(path, backend="raw"):
    import librosa

    samples, rate = audio(path)
    if backend not in {"raw", "hpss", "beat_this"}:
        raise ValueError("Beat backend must be raw, hpss or beat_this")
    params = {"sample_rate": rate, "hop_length": 512, "preprocessing": backend}
    if samples is None or np.max(np.abs(samples), initial=0) < 1e-6:
        params["audio_status"] = "missing_or_silent"
        return envelope(path, "beats", "librosa-" + backend, params, [])
    if backend == "beat_this":
        import torch
        from beat_this.inference import Audio2Beats

        torch.set_num_threads(2)
        tracker = Audio2Beats(checkpoint_path="small0", device="cpu", dbn=False)
        points, downbeats = tracker(torch.tensor(samples, dtype=torch.float32), rate)
        duration = metadata(path)[0]["duration_seconds"]
        params.update(checkpoint="small0", downbeat_times=[float(t) for t in downbeats])
        return envelope(
            path,
            "beats",
            "beat-this",
            params,
            [{"time": float(t)} for t in points if 0 <= t < duration],
        )
    signal = librosa.effects.percussive(samples) if backend == "hpss" else samples
    tempo, indices = librosa.beat.beat_track(y=signal, sr=rate, hop_length=512)
    params["estimated_tempo_bpm"] = float(np.asarray(tempo).ravel()[0])
    duration = metadata(path)[0]["duration_seconds"]
    times = librosa.frames_to_time(indices, sr=rate, hop_length=512)
    return envelope(
        path,
        "beats",
        "librosa-" + backend,
        params,
        [{"time": float(t)} for t in times if 0 <= t < duration],
    )


_readers = {}


def ocr(path, languages=("en",), sample_seconds=2.0):
    import easyocr

    if sample_seconds <= 0:
        raise ValueError("Sampling interval must be positive")
    key = tuple(languages)
    if key not in _readers:
        _readers[key] = easyocr.Reader(list(languages), gpu=False, verbose=False)
    duration = metadata(path)[0]["duration_seconds"]
    items = []
    for t in np.arange(0, duration, sample_seconds):
        image = frame(path, float(t))
        h, w = image.shape[:2]
        for polygon, text, confidence in _readers[key].readtext(image):
            xs, ys = zip(*polygon)
            items.append(
                {
                    "start": float(t),
                    "end": min(float(t + sample_seconds), duration),
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [
                        max(0, float(min(xs) / w)),
                        max(0, float(min(ys) / h)),
                        min(1, float(max(xs) / w)),
                        min(1, float(max(ys) / h)),
                    ],
                    "sample_time": float(t),
                    "temporal_extent": "sampling-bin-not-tracked",
                }
            )
    return envelope(
        path,
        "ocr",
        "easyocr",
        {"languages": list(languages), "sample_seconds": sample_seconds, "device": "cpu"},
        items,
    )


_speech = {}


def asr(path, model="tiny", beam_size=1):
    from faster_whisper import WhisperModel

    samples, rate = audio(path, 16000)
    params = {
        "model": model,
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": beam_size,
        "vad_filter": True,
    }
    if samples is None or np.max(np.abs(samples), initial=0) < 1e-6:
        params["audio_status"] = "missing_or_silent"
        return envelope(path, "asr", "faster-whisper", params, [])
    if model not in _speech:
        _speech[model] = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=4)
    segments, info = _speech[model].transcribe(
        samples.astype(np.float32), beam_size=beam_size, vad_filter=True, word_timestamps=True
    )
    duration = metadata(path)[0]["duration_seconds"]
    items = []
    for s in segments:
        start, end = max(0, s.start), min(duration, s.end)
        if end > start:
            items.append(
                {
                    "start": start,
                    "end": end,
                    "text": s.text,
                    "words": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "text": w.word,
                            "probability": w.probability,
                        }
                        for w in s.words or []
                    ],
                }
            )
    params.update(language=info.language, language_probability=info.language_probability)
    return envelope(path, "asr", "faster-whisper", params, items)


def transition_features(path, times, frame_indices=None, source_times=None):
    import cv2

    fps = metadata(path)[1]
    duration = metadata(path)[0]["duration_seconds"]
    rows = []
    for index, t in enumerate(times):
        images = [
            frame(
                path,
                source_times[min(max(0, frame_indices[index] + offset), len(source_times) - 1)]
                if source_times is not None
                else min(max(0, t + offset / fps), max(0, duration - 1 / fps)),
                160,
            )
            for offset in (-3, -1, 0, 2)
        ]
        differences = [
            float(np.mean(np.abs(a.astype(float) - b.astype(float))) / 255)
            for a, b in zip(images, images[1:])
        ]
        hist = []
        for image in images:
            h = cv2.calcHist(
                [cv2.cvtColor(image, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256]
            )
            hist.append(cv2.normalize(h, h).flatten())
        distance = float(cv2.compareHist(hist[1], hist[2], cv2.HISTCMP_BHATTACHARYYA))
        rows.append(differences + [distance, float(np.mean(images[2]) / 255)])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def transitions(path, shot_result=None, checkpoint=None):
    shot_result = shot_result or shots(path)
    times = [s["start"] for s in shot_result["items"][1:]]
    indices = [s.get("start_frame") for s in shot_result["items"][1:]]
    source_times = None
    if times and all(i is not None for i in indices):
        from .media import prepare_indexed_frames

        count = metadata(path)[2]
        source_times = prepare_indexed_frames(
            path, [min(max(0, i + offset), count - 1) for i in indices for offset in [-3, -1, 0, 2]]
        )
    features = transition_features(path, times, indices, source_times)
    duration, fps = metadata(path)[:2]
    items = []
    cnn = None
    if checkpoint:
        from .learned import transition_model

        cnn = transition_model(checkpoint)
    for item_index, (t, x) in enumerate(zip(times, features)):
        sharp = x[1] > max(x[0], x[2]) * 1.5 and x[1] > 0.08
        if cnn is not None:
            import cv2
            import torch

            before_time = (
                source_times[max(0, indices[item_index] - 1)]
                if source_times is not None
                else max(0, t - 1 / fps)
            )
            before = cv2.resize(frame(path, before_time), (32, 32))
            after = cv2.resize(frame(path, t), (32, 32))
            pair = np.concatenate([before, after], axis=2).transpose(2, 0, 1) / 255
            with torch.inference_mode():
                sharp = int(cnn(torch.tensor(pair[None], dtype=torch.float32)).argmax(1)) == 0
        items.append(
            {
                "start": max(0, t - 1 / fps),
                "end": min(duration["duration_seconds"], t + 1 / fps),
                "label": "hard_cut" if sharp else "non_hard_cut",
                "boundary_time": t,
                "features": x.tolist(),
            }
        )
    return envelope(
        path,
        "transitions",
        "small-transition-cnn" if checkpoint else "frame-histogram-rules",
        {
            "candidate_source": shot_result["backend"],
            "scope": "candidate-conditioned, not all transitions",
            "sharpness_ratio": 1.5,
            "minimum_difference": 0.08,
            "checkpoint": str(checkpoint) if checkpoint else None,
        },
        items,
    )


_clip = None


def shot_size(path, shot_result=None, ensemble=False, checkpoint=None):
    global _clip
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if _clip is None:
        name = "openai/clip-vit-base-patch32"
        _clip = (
            CLIPModel.from_pretrained(name, use_safetensors=True).eval(),
            CLIPProcessor.from_pretrained(name),
        )
    model, processor = _clip
    labels = ["extreme_close_up", "close_up", "medium_shot", "wide_shot", "detail_shot"]
    descriptions = [
        "an extreme close-up of a face, only eyes or lips visible",
        "a close-up portrait showing a person's face and shoulders",
        "a medium shot showing a person from the waist up",
        "a wide shot showing a full person and surroundings",
        "a detail shot of a beauty product, hand or small object",
    ]
    shot_result = shot_result or shots(path)
    items = []
    torch.set_num_threads(4)
    for s in shot_result["items"]:
        fractions = [0.25, 0.5, 0.75] if ensemble else [0.5]
        images = [
            Image.fromarray(frame(path, s["start"] + (s["end"] - s["start"]) * f)[:, :, ::-1])
            for f in fractions
        ]
        inputs = processor(text=descriptions, images=images, return_tensors="pt", padding=True)
        with torch.inference_mode():
            scores = model(**inputs).logits_per_image.softmax(-1).mean(0).tolist()
            if checkpoint:
                probe = np.load(checkpoint)
                image_inputs = processor(images=images, return_tensors="pt")
                embedding = model.get_image_features(**image_inputs).numpy()
                embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                logits = (embedding @ probe["coef"].T + probe["intercept"]).mean(0)
                if len(probe["classes"]) == 2 and len(logits) == 1:
                    logits = np.array([0.0, logits[0]])
                probability = np.exp(logits - logits.max())
                probability /= probability.sum()
                scores = [0.0] * len(labels)
                for label, prob in zip(probe["classes"], probability):
                    scores[int(label)] = float(prob)
        items.append(
            {
                "start": s["start"],
                "end": s["end"],
                "shot_id": s["shot_id"],
                "label": labels[int(np.argmax(scores))],
                "scores": dict(zip(labels, scores)),
            }
        )
    return envelope(
        path,
        "shot_size",
        "clip-linear-probe" if checkpoint else "clip-zero-shot",
        {
            "model": "openai/clip-vit-base-patch32",
            "model_revision": getattr(model.config, "_commit_hash", None),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "frames_per_shot": 3 if ensemble else 1,
            "prompts": descriptions,
            "score_note": "relative prompt scores, not calibrated probabilities",
        },
        items,
    )
