# Contributing

Vflash is a reusable inference engine. Product accounts, billing, private prompts, fleet credentials
and website code do not belong in this repository. Read [`AGENTS.md`](https://github.com/Hansimov/vflash/blob/main/AGENTS.md)
for the current ownership map before changing a runtime boundary.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,server]'
pre-commit install

pytest
pre-commit run --all-files
npm ci
npm run docs:build
```

CPU tests must not initialize CUDA. Hardware tests run in a dedicated process with an explicitly
selected device or pair. Keep model payloads and generated media outside Git.

## Change one contract at a time

Start from a fixed profile, artifact and request. State whether the change affects numerical
arithmetic, memory layout, loading, scheduling, media delivery or only diagnostics. Avoid mixing an
adapter, step-count or precision change into a kernel speed comparison.

An exact layout optimization should first prove element equality at representative shapes, then run
a complete request. Approximation must be named and reported as non-exact, never introduced through
an environment-dependent silent fallback. Version 0.5.0 explicitly changes the single-SM89 Base16
default to Sol while retaining the prepared model identity and explicit dense selection. This release
decision does not close the full media-quality gate or permit other unqualified approximations.

Add fail-closed input validation and a CPU reference path where appropriate. A runtime fallback must
remain visible in result metadata or be limited to a semantics-preserving implementation.

## Target-hardware qualification

SM86 and SM89 are independent targets. Passing on one does not qualify the other. Record:

- exact source revision, dependency versions and profile identity;
- GPU model/count and topology class, without public device UUIDs;
- cold preparation separately from repeated request time;
- A/B/A or repeated measurements on the same device group;
- peak device/host memory, sustained temperature and active throttling;
- complete output, cleanup and cancellation behavior;
- decoded multi-frame and audio evidence for any arithmetic approximation.

Kernel timings establish a mechanism, not an end-to-end claim. Keep latency for one cooperative
request separate from replica throughput across the same number of GPUs.

## Public evidence and privacy

Commit code, tests, reproducible public commands and aggregate evidence. Do not commit credentials,
private media, prompts, service configuration, hostnames, device UUIDs or application-specific
storage paths. Avoid hashing large model files during iteration; use pinned source manifests and
one-time preparation receipts.

When upstream work informs an implementation, pin the reviewed revision, preserve its license and
give specific attribution. Document which mechanisms were adopted and which remain unqualified.

## Release checklist

Before tagging a release:

1. run focused tests, the full CPU suite, privacy hooks and the bilingual docs build;
2. install the package from a clean checkout and rerun the public CLI smoke;
3. confirm README, Pages, CLI profiles and version metadata describe the same boundary;
4. record target-hardware evidence without claiming untested architectures or shapes;
5. tag one immutable commit and publish release notes that distinguish exact, approximate and
   experimental behavior.

Model checkpoints retain their own licenses and are never release assets.
