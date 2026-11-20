# Phase 2 implementation plan

The original checklist below is retained as the implementation roadmap. All six module adapters are now implemented; see [current usage and limits](phase2_usage.md). A method listed as a comparison is not necessarily evaluated in every run; use the measured report for actual coverage.

| Module | Baseline | Comparison | Evaluation |
| --- | --- | --- | --- |
| shots | PySceneDetect | TransNetV2 / AutoShot | One-to-one boundary F1, ±2 original frames |
| beats | librosa | Beat This! | Beat F-measure with explicit tolerance |
| ocr | EasyOCR / PaddleOCR | Engine/config comparison | Text accuracy and matched box IoU |
| asr | faster-whisper | Model size/config | WER with explicit multilingual normalization |
| transitions | Frame/histogram rules | Small CNN later | Accuracy, macro F1, confusion matrix |
| shot_size | CLIP / SigLIP prompts | Linear probe | Accuracy and per-class results |

## Suggested order

1. Media metadata: duration, FPS/time base, rotation, audio presence.
2. Shot baseline and boundary matching metric with hand-checked cases.
3. Beat baseline with silence/narration-only handling.
4. OCR sampling/tracking and ASR timestamps.
5. Transition and shot-size baselines.
6. Improved methods, held-out comparisons, and failure analysis.

Each backend implements `ParsingBackend.parse(Path) -> dict` and returns output accepted by `validate_result`. `bvt-parse` is the implemented inference command. Config TOML files record reference defaults; explicit CLI arguments remain the execution interface.

## Definition of done

- Real media produces validated task-specific JSON.
- Model revision, dependencies, config and hardware are recorded.
- Edge cases have meaningful tests.
- Evaluation uses fixed, leakage-aware splits.
- Baseline comparison and failure analysis are recorded.

Store local outputs in artifacts/phase2/<module>/<run-id>/. Version permitted aggregate results in experiments/results.csv. Do not tune on test data.
