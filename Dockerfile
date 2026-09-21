# mixengine in a container.
#
# Two stages so the image does not carry a compiler. librosa pulls in numba
# and llvmlite, which build from source on platforms without a wheel, and
# the toolchain that needs is several hundred megabytes that must not reach
# production.
#
# Rubber Band is installed from the distribution rather than left out: it is
# the difference between a formant-preserving stretch and a phase vocoder,
# which is the single largest quality difference available to this engine.

FROM python:3.11-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir ".[quality,web]"

# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 rubberband-cli ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MIXENGINE_DATA=/data

# Analysis and rendering write nothing outside the data volume, so the
# process has no reason to own its own code.
RUN useradd --create-home --uid 10001 mixengine && \
    mkdir -p /data && chown -R mixengine:mixengine /data
USER mixengine
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).status == 200 else 1)"

# Bound to every interface here because the container's network namespace is
# the boundary. Outside a container this binds to loopback, and that
# difference is deliberate: the interface has no authentication in front of
# it and exposes the filesystem paths it was given.
CMD ["mixengine", "serve", "--host", "0.0.0.0", "--port", "8000", "--data", "/data"]
