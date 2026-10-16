# Reuse immutable, previously published dependencies without downloading CUDA again.
# Build the wheel from the selected public source before invoking this recipe.
ARG VFLASH_BASE_IMAGE=hansimov/vflash@sha256:f6706267bc316d1b5fbc33154b96288f1ef410abdb9a22a943931b7d5e56afaf
FROM ${VFLASH_BASE_IMAGE}
ARG VFLASH_VERSION=0.11.0
ARG VFLASH_REVISION
USER root
WORKDIR /opt/vflash
# Keep the source visible in the image consistent with its installed wheel.
COPY src ./src
COPY pyproject.toml README.md LICENSE ./
COPY dist/vflash-${VFLASH_VERSION}-py3-none-any.whl /tmp/vflash-release.whl
RUN mv /tmp/vflash-release.whl /tmp/vflash-${VFLASH_VERSION}-py3-none-any.whl \
    && python -m pip install --no-index --no-deps --force-reinstall /tmp/vflash-${VFLASH_VERSION}-py3-none-any.whl \
    && rm /tmp/vflash-${VFLASH_VERSION}-py3-none-any.whl \
    && chmod -R a+rX /opt/vflash/src \
    && chmod a+r /opt/vflash/pyproject.toml /opt/vflash/README.md /opt/vflash/LICENSE
LABEL org.opencontainers.image.version="${VFLASH_VERSION}" \
      org.opencontainers.image.revision="${VFLASH_REVISION}"
USER 10001:10001
