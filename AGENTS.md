# Vflash contributor map

Vflash is a reusable inference engine, not an account, billing or media-studio platform.
Keep it installable from this repository alone, without external project paths or private data.

- `contracts.py`, `catalog.py`, `planner.py`: immutable profiles and resolved hardware plans.
- `native/runner.py`: fixed-profile engine sessions; `native/worker.py`: isolated process ownership.
- Other `native/` modules: H3 mathematics, artifacts, scheduling and kernels.
- `server.py`: transport, bounded temporary job queue and output delivery, not model mathematics.
- `docker/`: standalone deployment. `docs/`: matching English and Chinese guides.

Inspect the current implementation and supported profiles before extending a surface. Preserve numerical evidence,
licenses and attribution, but do not treat old module layout as a constraint. State the actual runnable boundary;
planned modes and full video generation must not be presented as released features.

Install development dependencies with `python -m pip install -e '.[dev,server]'`. Use focused tests during iteration;
public commits must pass the existing privacy hook. Never commit model payloads, private cases, credentials or GPU
identifiers. Record reproducible public evidence, not private development chronology.
