# Vflash contributor map

Vflash is a reusable inference engine, not an account, billing or media-studio platform.
Keep it installable from this repository alone, without external project paths or private data.

- `contracts.py`, `catalog.py`, `planner.py`: immutable profiles and resolved hardware plans.
- `native/runner.py`: fixed-profile engine sessions; `native/worker.py`: isolated process ownership.
- `native/h3_native_conditioning_runtime.py`: owned input, head and trunk resources per session.
- `native/h3_tensor_file.py`: owned tensor reads and short-lived mapped load scopes.
- Other `native/` modules: H3 mathematics, artifacts, scheduling and kernels.
- `pipeline/`: explicit complete-request contracts, prepared assets and stage ownership.
- `adapters/`, `media/`: pinned official encoder/VAE adapters and local MP4 delivery.
- `compiler/`, `model_assets.py`: official raw-weight identity, conversion and artifact writing.
- `server.py`: transport, bounded temporary job queue and output delivery, not model mathematics.
- `docker/`: standalone deployment. `docs/`: matching English and Chinese guides.

Inspect the current implementation and supported profiles before extending a surface. Preserve numerical evidence,
licenses and attribution, but do not treat old module layout as a constraint. `docs/guide/profiles.md` owns hardware
and capacity qualification; `docs/guide/complete-pipeline.md` owns complete-request modes, durations and optional
repaint interfaces. Native sessions/HTTP return latents; Python/CLI complete pipelines deliver MP4s. Historical
release notes and image inventories describe their exact versions, not current defaults or newly qualified GPUs.
Keep English/Chinese installation commands, Docker examples and actual published image versions consistent.
Use a minor release for a substantial compatible capability/integration update and a major release for a declared
breaking public contract; reserve patch releases for bounded fixes. Do not expand quality claims from CPU checks.

Install development dependencies with `python -m pip install -e '.[dev,server]'`. Use focused tests during iteration;
public commits must pass the existing privacy hook. Never commit model payloads, private cases, credentials or GPU
identifiers. Record reproducible public evidence, not private development chronology.
