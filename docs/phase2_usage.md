# Phase 2 implementation and reproduction

All six adapters implement `Backend(**options).parse(Path) -> dict`. The shared JSON envelope is validated before return. Inference runs locally on CPU. First use downloads public model weights; video inputs are not uploaded.

Evaluate independent reference JSON with `bvt-evaluate PREDICTION REFERENCE --tolerance 0.066667 --output artifacts/metrics.json`. References use the same envelope, original video ID, and task-specific items. For shots choose tolerance `2/reference_FPS`; for beats report your chosen seconds tolerance. WER uses exact whitespace tokens unless you normalize externally. OCR matching uses descending IoU among overlapping intervals and must use frame-aligned samples. Unmatched predictions and references must be reviewed, not discarded.

```sh
python -m pip install -e ".[dev,phase2]"
bvt-parse shots VIDEO --backend content --output artifacts/shots.json
bvt-parse beats VIDEO --backend hpss --output artifacts/beats.json
bvt-parse ocr VIDEO --languages en --output artifacts/ocr.json
bvt-parse asr VIDEO --model tiny --output artifacts/asr.json
bvt-parse transitions VIDEO --output artifacts/transitions.json
bvt-parse shot_size VIDEO --ensemble --output artifacts/shot_size.json
```

Shot backends: content, adaptive, transnet. Beat backends: raw, hpss, beat_this. Beat This! additionally requires `pip install -e ".[beat-model]"`. Compatible OCR groups include `ko en` and `ch_sim en` in separate runs; English is the demo default. OCR time intervals represent sampling bins, not tracked visibility.

## Sample study and trained comparisons

```sh
python scripts/run_phase2_study.py PATH_TO_RESOURCES --output artifacts/phase2/study --limit 30 --heavy-samples 3
python scripts/train_phase2_comparisons.py PATH_TO_RESOURCES artifacts/phase2/learned
python scripts/phase2_report.py artifacts/phase2/study PATH_TO_RESOURCES OUTPUT_FOLDER
bvt-parse transitions VIDEO --checkpoint artifacts/phase2/learned/transition_cnn.pt --output artifacts/cnn.json
bvt-parse shot_size VIDEO --checkpoint artifacts/phase2/learned/shot_size_probe.npz --output artifacts/probe.json
```

Training uses 30 IDs split 21/3/6 by video. Sampling is deterministic and limited; creator/campaign grouping has not been verified. Supplied labels are not independent gold annotations. Report conditional classification metrics separately from boundary detection. The CNN is binary, not the optional 4–6 class classifier. Independent transcripts, beat points and OCR boxes are needed for their accuracy metrics.

## TransNetV2

Clone https://github.com/soCzech/TransNetV2 to `models/weights/TransNetV2`. Download `transnetv2-pytorch==1.0.5` as a wheel and extract its `transnetv2_pytorch/transnetv2-pytorch-weights.pth` to `models/weights/transnetv2-pytorch-weights.pth`. Architecture is upstream; weights come from community packaging, a distinct provenance. Keep licenses/revisions/checksums with experiment records. No third-party source or weights are committed.

The adapter uses safe weight-only loading, 100-frame windows, a central 50-frame stride, threshold 0.5, and a maximum-score representative for each contiguous positive transition group. This policy can differ from upstream scene interval conversion.

## Timing and configuration

Shot output times use decoded presentation timestamps. The study verifies annotated total frame counts against each source video and compares boundary frame indices directly with ±2-frame tolerance. Supplied seconds can be wrong for VFR media (V003 is a concrete example) and are not used as boundary ground truth. Training samples use annotated source frame indices and decode those frames' actual PTS. Constant-FPS fields are metadata, not a replacement for PTS.

TOML files in configs/phase2 are planning records; the runner consumes explicit CLI options. Record actual commands and resolved dependencies for every run. Raw media, transcripts, weights and frame previews remain under ignored local artifacts.
