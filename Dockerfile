# syntax=docker/dockerfile:1.7

FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 vflash \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin vflash \
    && install -d -o 10001 -g 10001 \
        /cache/cuda /cache/torch /cache/torch-extensions /cache/triton /cache/inductor /outputs

WORKDIR /opt/vflash
COPY docker/requirements-api.txt /tmp/requirements-api.txt

# Small target used by CI and maintainers to validate the HTTP image without
# downloading the CUDA-enabled PyTorch wheel.
FROM base AS api-test
RUN python -m pip install --requirement /tmp/requirements-api.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-deps .
USER 10001:10001
RUN python -c "from vflash.server import create_app; assert create_app().title"

FROM base AS runtime

COPY docker/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0+cu130 \
    && python -m pip install --requirement /tmp/requirements-runtime.txt \
    && rm /tmp/requirements-api.txt /tmp/requirements-runtime.txt

RUN apt-get update \
    && apt-get install --yes --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Code-only edits preserve the CUDA, Python dependency and compiler layers.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-deps .

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0" \
    CUDA_CACHE_PATH=/cache/cuda \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TORCH_HOME=/cache/torch \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    XDG_CACHE_HOME=/cache \
    VFLASH_PROFILE_ID=ref2va-turbo4-exact-sm89 \
    VFLASH_GPU_INDEX=0 \
    VFLASH_ARTIFACT_PATH=/runtime/artifact \
    VFLASH_SCHEDULE_OVERLAY_PATH=/runtime/schedule \
    VFLASH_AUXILIARY_TENSOR_PATH=/runtime/auxiliary.safetensors \
    VFLASH_BUNDLE_ROOT=/runtime/bundles \
    VFLASH_OUTPUT_ROOT=/outputs

LABEL org.opencontainers.image.title="Vflash" \
    org.opencontainers.image.description="H3 video and audio inference with explicit GPU profiles" \
    org.opencontainers.image.source="https://github.com/Hansimov/vflash"

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"]

CMD ["python", "-m", "vflash.server"]

# Complete Python and command-line pipeline. The separate runtime target keeps
# the resident HTTP bundle-to-latent service.
FROM runtime AS pipeline
USER root
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-deps \
        --index-url https://download.pytorch.org/whl/cu130 \
        torchvision==0.26.0+cu130
# This installer can resume interrupted dependency downloads with hash checks.
RUN python -m pip install --timeout 120 --retries 5 pip==25.3
# Resolve the installed wheel's extras without rebuilding or replacing it.
RUN python -m pip install --timeout 120 --retries 5 --resume-retries 5 'vflash[pipeline]'
USER 10001:10001
RUN python -c "import torch; from vflash.adapters.diffusers_h3 import validate_adapter_dependencies; from vflash.media.encoding import media_executables; validate_adapter_dependencies(); media_executables(); assert not torch.cuda.is_initialized()"
HEALTHCHECK NONE
ENTRYPOINT ["vflash"]
CMD ["--help"]

# A plain build and the existing Compose service keep the native-only image.
FROM runtime AS default
