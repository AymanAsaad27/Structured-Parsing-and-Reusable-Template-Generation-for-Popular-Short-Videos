"""Unified input-video to JSON dispatcher."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description="Run a Phase 2 module locally")
    p.add_argument("module", choices=["shots", "beats", "ocr", "asr", "transitions", "shot_size"])
    p.add_argument("video", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", default=None)
    p.add_argument("--languages", nargs="+", default=["en"])
    p.add_argument("--sample-seconds", type=float, default=2)
    p.add_argument("--model", default="tiny")
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument(
        "--enhanced",
        action="store_true",
        help="OCR: wider frames/beam search; ASR: small model/beam5/no previous-text conditioning",
    )
    p.add_argument(
        "--language", default=None, help="Known ASR language code; omit for automatic detection"
    )
    p.add_argument("--beat-checkpoint", default="small0")
    p.add_argument("--threshold", type=float, help="Shot detector threshold")
    p.add_argument(
        "--min-content-val", type=float, default=15, help="Adaptive detector minimum content score"
    )
    p.add_argument(
        "--min-scene-len", type=int, default=6, help="Minimum shot length in source frames"
    )
    args = p.parse_args(argv)
    from . import backends

    kwargs = {}
    if args.module in {"shots", "beats"} and args.backend:
        kwargs["backend"] = args.backend
    if args.module == "shots":
        kwargs.update(
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            min_content_val=args.min_content_val,
        )
    if args.module == "ocr":
        kwargs.update(
            languages=args.languages, sample_seconds=args.sample_seconds, enhanced=args.enhanced
        )
    if args.module == "asr":
        kwargs.update(
            model="small" if args.enhanced and args.model == "tiny" else args.model,
            beam_size=5 if args.enhanced else 1,
            language=args.language,
            condition_on_previous_text=not args.enhanced,
        )
    if args.module == "beats":
        kwargs["checkpoint"] = args.beat_checkpoint
    if args.module == "shot_size":
        kwargs["ensemble"] = args.ensemble
    if args.checkpoint:
        if args.module not in {"shot_size", "transitions"}:
            p.error("--checkpoint is supported for shot_size and transitions")
        kwargs["checkpoint"] = args.checkpoint
    result = getattr(backends, args.module)(args.video, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
