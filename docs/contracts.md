# Parsing result contract, version 0.1.0

Required fields: schema_version, module, backend, video, parameters, items. All times are seconds on the original video timeline. Intervals use [start, end). Undo sampling and audio offsets before returning timestamps.

The JSON Schema validates the envelope. Python validation additionally rejects non-finite timestamps, zero-length/reversed intervals and timestamps beyond duration. Task payloads are extensible; add task-specific schemas with each real backend.

Proposed payloads:

- shots: interval, shot_id.
- beats: point time, optional confidence/downbeat.
- ocr: interval, text, normalized bbox, language, confidence.
- asr: interval, text, language, optional word timestamps.
- transitions: interval, label, confidence.
- shot_size: interval, label, confidence.

Bounding boxes should use normalized [x0,y0,x1,y1] on the displayed frame after rotation correction. Empty items is valid for no detections. Failures must raise errors rather than return empty success. Missing audio must be recorded explicitly rather than replaced with fabricated beats.
