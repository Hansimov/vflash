# Reuse immutable, previously published dependencies without downloading CUDA again.
# Build the wheel from the selected public source before invoking this recipe.
ARG VFLASH_BASE_IMAGE=hansimov/vflash@sha256:e7f0482bcd47564d2bba16e537644f81639e5f554bab7670945836f86c3f6e30
FROM ${VFLASH_BASE_IMAGE}
ARG VFLASH_VERSION=0.13.0
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
