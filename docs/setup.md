# Environment setup

1. Create an isolated Python 3.11/3.12 environment and install the development extra using the README.
2. Install FFmpeg separately and check `ffmpeg -version` and `ffprobe -version`.
3. Run `bvt doctor`; this checks executable availability, not GPU inference.
4. Install only the parsing extras needed for your current module.
5. Select PyTorch for your hardware using https://pytorch.org/get-started/locally/.
6. Follow the selected backend's official model instructions and record its exact revision and license.
7. Save `python scripts/environment_report.py > artifacts/environment.json` and `python -m pip freeze > artifacts/requirements-resolved.txt`.

Dependency ranges are installation constraints, not a tested GPU lockfile. Capture resolved dependencies per experiment. CI is configured for the lightweight core on Windows/Linux and Python 3.11/3.12. Model stacks require separate integration validation.

Optional credentials belong in environment variables, never source files. The core does not consume `.env` automatically.
