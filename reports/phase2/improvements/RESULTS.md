# PHASE 2 - IMPROVEMENT RESULTS | 26 November 2026
Working report for manual revision before lab submission.


## Current position

All six parsing adapters are implemented. This run expands supervised training to the 200-video collection, repairs evaluation errors, and measures shot, transition and shot-size performance. Beat, OCR and speech accuracy remain unscored until their reviewed reference files are located.


## Data and experiment design

The supplied table contains 3353 shots in 200 videos: 3,153 transitions and 3,260 unambiguous shot-size labels. All source frame counts and shot intervals match. There are 80 extra byte-identical video copies; one source per video ID is used. Written end times differ by more than 100 ms in 28 videos, so evaluation uses source frames and decoded timestamps.
The frozen split is 140 training / 20 validation / 40 test videos. The previously inspected V001-V030 are training-only. Creator/campaign grouping is unavailable. Classification uses annotated boundaries and midpoints, so these scores do not include shot-detection errors. Ninety-three mixed 'medium/product presentation' labels are excluded rather than silently relabelled.
The first ten sorted test IDs were used for the three-detector comparison. Shot settings were then selected using only the 20 validation videos; the remaining 30 test videos provide the primary untouched shot-detection comparison. The four-class transition model and shot-size probe use all 40 classification test videos.
Medium and wide labels occur only in the already inspected first 30 videos, leaving zero examples of those classes in the new validation/test sets. A supplementary five-fold video-separated development assessment covers all 3,260 labelled shots with the probe's C=0.01 frozen from validation. It is not a fresh external test.


## Measured results

- content-default: boundary micro F1 +/-2 frames 0.840; n=30.
- adaptive-default: boundary micro F1 +/-2 frames 0.869; n=30.
- adaptive-validation-selected: boundary micro F1 +/-2 frames 0.858; n=30.
- transition_four_class_majority: accuracy 0.855; macro F1 0.230; n=620.
- transition_four_class: accuracy 0.871; macro F1 0.517; n=620.
- transition_binary_old: accuracy 0.956; macro F1 0.489; n=620.
- transition_binary_new: accuracy 0.942; macro F1 0.535; n=620.
- shot_size_zero_shot: accuracy 0.326; macro F1 0.172; n=629.
- shot_size_old: accuracy 0.170; macro F1 0.078; n=629.
- shot_size_new: accuracy 0.598; macro F1 0.301; n=629.
- transition_binary_rules: accuracy 0.802; macro F1 0.591; n=620.
- size_cv_zero_shot: accuracy 0.364; macro F1 0.194; n=3260.
- size_cv_probe_oof: accuracy 0.616; macro F1 0.308; n=3260.
The selected shot configuration is {"backend": "adaptive", "threshold": 2, "min_scene_len": 6, "min_content_val": 15}. Its 30-video boundary F1 is 0.858; the brief's reference target is 0.85.
The validation-selected threshold does not beat default adaptive detection on the 30-video test. Retain adaptive threshold 3, minimum scene length 6 and minimum content score 15 as the current default. The threshold-2 candidate improves recall at a precision cost; it is recorded, not promoted.
The four-class transition CNN reaches 87.1% accuracy versus 85.5% for the majority baseline, but dissolve recall is only 1/26 (3.8%). Validation-selected soft-decision threshold tuning retained ordinary argmax, so it did not fix this failure. The unchanged binary rule has higher macro F1 (0.591) than the collapsed CNN (0.535), despite lower accuracy. Keep both comparisons visible.
Supplementary shot-size CV improves accuracy from 36.4% zero-shot to 61.6% with the probe. Medium-shot recall is 2/67 (3.0%) and wide-shot recall is 0/47; the probe is not yet reliable across the full taxonomy. Full-frame-preserving preprocessing and consistent medium/wide labelling are the next development priorities.
On the separate ten-video comparison, content/adaptive/TransNetV2 boundary F1 is 0.795/0.849/0.682. TransNetV2 is not the preferred detector for this collection on these measurements.


## What changed

1. Fixed taxonomy-wide macro F1 and added per-class support/recall/confusion matrices. High accuracy can no longer conceal an all-hard-cut classifier.
2. Expanded from a sparse 30-video pilot to all annotated transitions and unambiguous shot-size midpoints. CNN class weighting and probe regularization are selected on validation; the scaler is fitted on training only.
3. Added labelled four-class CNN checkpoints and standardized CLIP probes while preserving legacy inference. Feature extraction is resumable, sharded and writes completed files atomically.
4. Fixed OCR matching across different sampled frames; missed/spurious text contributes to CER, and missed boxes contribute to reference-averaged IoU. Empty speech references no longer produce a misleading perfect WER.
5. Added explicit detector settings, OCR wider-frame/beam-search options, ASR language/decoding options and selectable Beat This! checkpoints.


## Remaining evaluation

Beats: score librosa raw/HPSS and Beat This! against reviewed beat timestamps (70 ms tolerance). OCR: score baseline and enhanced outputs on identical reviewed frames, including boxes/text. Speech: compare exact-token WER for the baseline and beam-5 outputs using reviewed transcripts; document any text normalization. Candidate outputs and configuration changes are not accuracy measurements.
The OCR example uses V003 at 5-second sampling with English/Spanish recognition. Baseline/enhanced outputs contain 33/57 boxes across the sampled video; additional boxes can be correct detections or false positives. ASR examples use the cached tiny Whisper model. The small checkpoint download failed because of network/TLS errors; no small-model result is claimed.


## Findings and next steps

Inspect the confusion matrices before choosing a classifier. The collection has only six black-transition examples, 47 wide shots and 67 medium/medium-close shots. The black class therefore has very little training and test evidence. Add reviewed examples for these classes and check the distinction between face extreme close-ups and product/detail shots.
V056 illustrates missed edits with similar face/background composition and a persistent inset. The displayed frames show why whole-image change can be weak. Inspect the short clip around ambiguous boundaries before changing labels or adding specialized motion logic.
Finish beat/OCR/speech evaluation using the existing reviewed files, then update the module table and failure analysis. Phase 2 should be signed off after those measurements and the lab's review of remaining errors. Phase 3/4 work can then use the chosen detector and trained probes.


## Validation

32 automated tests, repository lint/format checks and package build passed during this improvement run. Private videos, large pretrained weights and credentials remain outside Git. Full experiment configurations, supports and confusion matrices accompany this report.
