# Contributing

Use focused branches such as `feat/shot-segmentation`. Explain resulting behavior, validation, assumptions and limitations in pull requests.

Run `python -m pytest`, `ruff check .`, `ruff format --check .`, and `python -m build` before submitting. Unit tests must not download models or require a GPU or network. Add meaningful integration fixtures when implementing detectors.

Backends accept a video path and return the shared JSON envelope. Keep optional model imports inside execution paths so core commands remain lightweight. Record model revision, parameters and environment for experiments. Test original-timeline units, missing audio, empty detections, bad input and missing dependencies.

Do not commit private videos, secrets, weights or unsupported benchmark claims. Attribute adopted code and check its license.
