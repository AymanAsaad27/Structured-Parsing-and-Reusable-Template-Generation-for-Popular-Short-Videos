# Architecture

Reference video → independent parsing results → semantic labeling → content-independent template → material matching → rendering → evaluation.

Parsing backends share a result envelope but own task-specific payloads. Analysis and evaluation remain separate: detectors must never access test annotations. Future orchestration should call module interfaces rather than embed model logic.

The shipped schema covers parsing results only. A reusable template schema is future work and needs content-independent slots, duration constraints, subtitle layout and rhythm policy. A parsing result is not yet a template.
