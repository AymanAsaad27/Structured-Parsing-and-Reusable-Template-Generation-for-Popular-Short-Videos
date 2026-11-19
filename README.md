# Beauty Video Templates

Structured parsing and reusable editing templates for 15–60 second vertical beauty videos.

**Status: development scaffold.** The command-line tool, parsing-result validation, and core tests work. AI backends, template generation, rendering, and evaluation metrics are planned. Synthetic examples are not research results.

## Quick start

Use Python 3.11 or 3.12 for model environments. The lightweight core also supports newer Python; model packages may have narrower compatibility.

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
bvt doctor
bvt validate examples/results/shots.json
python -m pytest
ruff check .
ruff format --check .
```

FFmpeg is installed separately. Install optional model dependencies only when needed, for example `python -m pip install -e ".[shots]"`. No models download during core installation.

## Repository layout

```text
.github/                  CI, issue templates, pull request checklist
configs/phase2/           Per-module experiment configuration drafts
src/beauty_video_templates/
  parsing/               Shots, beats, OCR, ASR, transitions, shot size
  semantics/             Shot function labeling (Phase 3)
  templates/             Abstraction and clustering (Phase 3)
  matching/              Material assignment (Phase 4)
  rendering/             Timeline assembly and synthesis (Phase 4)
  evaluation/            Module and system metrics (Phases 2/5)
  pipelines/             Future orchestration
  schemas/               Packaged parsing-result schema
  contracts.py           Shared validation and backend interface
  cli.py                 Working doctor and validate commands
data/                    Raw/interim/processed data, metadata, annotations, splits
models/                  Provenance conventions; weights stay local
examples/                Small synthetic examples
experiments/             Experiment protocol and aggregate results
reports/                 Deliverables and Phase 2 comparison outline
prompts/                 Versioned semantic labeling prompt
notebooks/               Exploration conventions
apps/                    Future demonstration interface
scripts/                 Environment snapshot utility
tests/                   Contract and CLI behavior tests
artifacts/               Local run outputs, excluded from Git
docs/                    Setup, architecture, contracts, roadmap, publishing
```

## Research roadmap

1. Research and proposal.
2. Dataset collection and annotation.
3. **Phase 2: independent parsing modules and baseline comparisons.**
4. Semantic analysis, reusable templates and clustering.
5. Material matching and video rendering.
6. Structural/content similarity and user evaluation.
7. Integrated demonstration and reproducibility report.

Start with [Phase 2 tasks](docs/phase2.md), [setup](docs/setup.md), and [architecture](docs/architecture.md).

## Data and licensing

Do not commit original videos, private media, weights or credentials. Store permitted metadata and video IDs. Record data permissions and model licenses independently. Similarity scores are not proof of originality.

No open-source code license has been selected automatically; see [licensing](docs/licensing.md). Read [contribution guidance](CONTRIBUTING.md) and [publishing instructions](docs/publishing.md).
