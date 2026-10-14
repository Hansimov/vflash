# Reuse immutable, previously published dependencies without downloading CUDA again.
# Build the wheel from the selected public source before invoking this recipe.
ARG VFLASH_BASE_IMAGE
FROM ${VFLASH_BASE_IMAGE}
ARG VFLASH_VERSION=0.8.0
ARG VFLASH_REVISION
USER root
WORKDIR /opt/vflash
# Keep the source visible in the image consistent with its installed wheel.
COPY src ./src
COPY pyproject.toml README.md LICENSE ./
COPY dist/vflash-${VFLASH_VERSION}-py3-none-any.whl /tmp/vflash-release.whl
RUN mv /tmp/vflash-release.whl /tmp/vflash-${VFLASH_VERSION}-py3-none-any.whl \
    && python -m pip install --no-index --no-deps --force-reinstall /tmp/vflash-${VFLASH_VERSION}-py3-none-any.whl \
    && rm /tmp/vflash-${VFLASH_VERSION}-py3-none-any.whl
LABEL org.opencontainers.image.version="${VFLASH_VERSION}" \
      org.opencontainers.image.revision="${VFLASH_REVISION}"
USER 10001:10001
