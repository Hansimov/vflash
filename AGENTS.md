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
licenses and attribution, but do not treat old module layout as a constraint. State the actual runnable boundary;
the Python/container pipeline generates complete Ref4 videos from one to three ordered references on one SM89 GPU
or a cooperating SM86 pair, and Base4 T2VA on one SM89 GPU. Complete SM86 uses `sequence-head`; single-SM86
and Turbo8 retain their latent interface. The current profile guide owns the supported boundary; archived release
notes describe their original versions, not additional requirements for a new release.

Install development dependencies with `python -m pip install -e '.[dev,server]'`. Use focused tests during iteration;
public commits must pass the existing privacy hook. Never commit model payloads, private cases, credentials or GPU
identifiers. Record reproducible public evidence, not private development chronology.
