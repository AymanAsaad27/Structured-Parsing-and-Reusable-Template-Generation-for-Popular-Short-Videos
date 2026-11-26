# Phase 2 implementation and reproduction

All six adapters implement `Backend(**options).parse(Path) -> dict`. The shared JSON envelope is validated before return. Inference runs locally on CPU. First use downloads public model weights; video inputs are not uploaded.

Evaluate reference JSON with `bvt-evaluate PREDICTION REFERENCE --output artifacts/metrics.json`. References use the same envelope, original video ID, and task-specific items. For shots use `--timebase frames --tolerance 2` with `start_frame` fields. This also handles variable-frame-rate videos and inaccurate written seconds. For beats report the seconds tolerance (default 70 ms). WER uses exact whitespace tokens unless you normalize externally. OCR uses maximum-cardinality matching at IoU >= 0.5 on samples within 1 ms; use the same sampled frames for references and predictions. The report includes missed/spurious text in CER and missed boxes in reference-averaged IoU, alongside conditional matched-box IoU.

```sh
python -m pip install -e ".[dev,phase2]"
bvt-parse shots VIDEO --backend content --output artifacts/shots.json
bvt-parse beats VIDEO --backend hpss --output artifacts/beats.json
bvt-parse ocr VIDEO --languages en --output artifacts/ocr.json
bvt-parse asr VIDEO --model tiny --output artifacts/asr.json
bvt-parse transitions VIDEO --output artifacts/transitions.json
bvt-parse shot_size VIDEO --ensemble --output artifacts/shot_size.json
```

Shot backends: content, adaptive, transnet. Adaptive is the current default (threshold 3, minimum length 6, minimum content score 15); pass `--backend content` to reproduce the content baseline. On the new 30-video test, adaptive/default F1 is 0.869 versus content 0.840; the validation-selected threshold-2 candidate scores 0.858 and is not promoted. Beat backends: raw, hpss, beat_this. Beat This! additionally requires `pip install -e ".[beat-model]"`. Compatible OCR groups include `ko en` and `ch_sim en` in separate runs; English is the demo default. OCR time intervals represent sampling bins, not tracked visibility.

## Sample study and trained comparisons

```sh
python scripts/run_phase2_study.py PATH_TO_RESOURCES --output artifacts/phase2/study --limit 30 --heavy-samples 3
python scripts/train_phase2_comparisons.py PATH_TO_RESOURCES artifacts/phase2/learned
python scripts/phase2_report.py artifacts/phase2/study PATH_TO_RESOURCES OUTPUT_FOLDER
bvt-parse transitions VIDEO --checkpoint artifacts/phase2/learned/transition_cnn.pt --output artifacts/cnn.json
bvt-parse shot_size VIDEO --checkpoint artifacts/phase2/learned/shot_size_probe.npz --output artifacts/probe.json
```

The original pilot uses 30 IDs split 21/3/6 by video and limited deterministic sampling. Its CNN is binary. The user has confirmed annotations were reviewed; reviewer identities and inter-annotator agreement are not recorded in this repository. Keep reference-conditioned classification results separate from end-to-end boundary detection. Creator/campaign grouping has not been verified. Beats, OCR and speech need their corresponding reviewed timestamps, boxes/text and transcripts for accuracy evaluation.

## Expanded experiment

```sh
python scripts/phase2_annotation_audit.py PATH_TO_RESOURCES artifacts/phase2/improvements/annotation_audit.json
python scripts/phase2_full_features.py PATH_TO_RESOURCES artifacts/phase2/improvements/full200
python scripts/phase2_full_train.py artifacts/phase2/improvements/full200 artifacts/phase2/improvements/classifiers
python scripts/phase2_boundary_holdout.py PATH_TO_RESOURCES artifacts/phase2/improvements/full200/protocol.json artifacts/phase2/improvements/boundaries
python scripts/phase2_rule_holdout.py PATH_TO_RESOURCES artifacts/phase2/improvements/full200/protocol.json artifacts/phase2/improvements/rules
python scripts/phase2_shot_selection.py PATH_TO_RESOURCES artifacts/phase2/improvements/full200/protocol.json artifacts/phase2/improvements/shot_selection
python scripts/phase2_size_crossvalidation.py artifacts/phase2/improvements/full200 artifacts/phase2/improvements/classifiers/results.json artifacts/phase2/improvements/size_crossvalidation
python scripts/phase2_transition_threshold.py artifacts/phase2/improvements/full200 artifacts/phase2/improvements/classifiers/transition_four_class.pt artifacts/phase2/improvements/transition_threshold
python scripts/phase2_failure_examples.py PATH_TO_RESOURCES artifacts/phase2/improvements
python -m pip install -e ".[reporting]"
python scripts/phase2_improvement_report.py artifacts/phase2/improvements PATH_TO_DELIVERY_FOLDER
```

Feature extraction is resumable by video. It requires a local `models/weights/clip-b32` directory containing the official CLIP ViT-B/32 processor/config and safetensors weights. The frozen protocol uses 140/20/40 videos; the previously inspected first 30 are restricted to training. Rare-class coverage is allocated before performance measurement. CNN checkpoints now carry their four output labels, and standardized probes carry their train-only scaler. Both legacy binary CNNs and unscaled probes remain supported.

To extract in parallel, run two workers with `--shards 2 --shard 0` and `--shards 2 --shard 1`. Each worker owns separate IDs and writes complete files atomically. Do not start training until both finish. The shot selection experiment excludes the ten test videos already inspected in the detector comparison and uses the remaining 30 for its primary test.

The fresh size test contains no medium/wide examples: those labels occur only in V001-V030. Supplementary five-fold grouped development cross-validation covers all labels with fixed validation-selected probe settings and train-fold-only scaling. It is explicitly not an untouched external test. Medium/wide recall and dissolve recall remain weak; see [expanded results and failure analysis](../reports/phase2/improvements/RESULTS.md). Transition threshold tuning used validation only and retained ordinary argmax; it did not justify a new decision rule.

Use `--checkpoint artifacts/phase2/improvements/classifiers/transition_four_class.pt` for four-class transition inference and `--checkpoint artifacts/phase2/improvements/classifiers/shot_size_probe.npz` for the updated shot-size probe. Classification uses supplied shot boundaries; detection errors must be evaluated separately. Full training compares models on the same held-out frames with a fixed taxonomy, including zero-support classes in macro F1.

`bvt-parse ocr VIDEO --enhanced --languages en es ...` enables wider frames and beam search. `bvt-parse asr VIDEO --enhanced --language en ...` requests the small Whisper model, beam 5 and disables previous-text conditioning. Supply a local model folder through `--model` for offline use. `bvt-parse beats VIDEO --backend beat_this --beat-checkpoint final0 ...` selects a different Beat This! checkpoint. These are candidate settings; enabling them alone does not establish improved accuracy. Choose language groups from the video's annotation, rather than forcing English on every video.

## TransNetV2

Clone https://github.com/soCzech/TransNetV2 to `models/weights/TransNetV2`. Download `transnetv2-pytorch==1.0.5` as a wheel and extract its `transnetv2_pytorch/transnetv2-pytorch-weights.pth` to `models/weights/transnetv2-pytorch-weights.pth`. Architecture is upstream; weights come from community packaging, a distinct provenance. Keep licenses/revisions/checksums with experiment records. No third-party source or weights are committed.

The adapter uses safe weight-only loading, 100-frame windows, a central 50-frame stride, threshold 0.5, and a maximum-score representative for each contiguous positive transition group. This policy can differ from upstream scene interval conversion.

## Timing and configuration

Shot output times use decoded presentation timestamps. The study verifies annotated total frame counts against each source video and compares boundary frame indices directly with ±2-frame tolerance. Supplied seconds can be wrong for VFR media (V003 is a concrete example) and are not used as boundary ground truth. Training samples use annotated source frame indices and decode those frames' actual PTS. Constant-FPS fields are metadata, not a replacement for PTS.

TOML files in configs/phase2 are planning records; the runner consumes explicit CLI options. Record actual commands and resolved dependencies for every run. Raw media, transcripts, weights and frame previews remain under ignored local artifacts.
