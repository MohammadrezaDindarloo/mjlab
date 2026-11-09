# mjlab environment image for EIDF (Kubernetes) and, via Apptainer, Slurm.
#
# Contains only the ENVIRONMENT: Python 3.13 and every package pinned in uv.lock,
# installed in /opt/venv. The repo's code is NOT in the image: each job mounts or
# uploads the repo and runs SETUP_CMD (`uv sync`, see README.md), which installs
# the repo itself into /opt/venv in seconds, offline.
#
# Build with docker/build.sh (from the repo root, the build context).

# CUDA 13 base: torch (cu130 wheels) and warp-lang 1.18 are CUDA 13 builds and
# bring their own CUDA libraries, so no -devel image (nvcc) is needed. They need an
# NVIDIA driver >= 580 on the host; this base image refuses to start on older ones
# with a clear "unsatisfied condition: cuda>=13.0" error.
# TODO: check the EIDF driver before the first real job:
#   ./eidf run gpu-test --gpu 1 -- nvidia-smi    (needs Driver Version >= 580)
# If it is older, switch to the cu126 extra (and a CUDA 12 warp build).
FROM nvidia/cuda:13.0.2-base-ubuntu24.04

# uv, pinned (same version as on the dev PC).
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/

# OS packages: git (train.sh prints the commit), EGL/GL for MuJoCo's headless
# rendering (videos, point-cloud viewers).
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'
# (safe.directory: the job runs as root in a repo copy owned by another user.)

# uv settings. Everything lives under /opt (not /root), so the image also works
# read-only and as a non-root user under Apptainer.
#   UV_PROJECT_ENVIRONMENT: `uv sync` / `uv run` in any repo copy use /opt/venv.
#   UV_NO_SYNC: `uv run` never re-syncs at run time (SETUP_CMD does it once).
#   UV_PYTHON_DOWNLOADS=never: never fetch another Python at run time.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_SYNC=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Layer 1: Python itself.
RUN uv python install 3.13 && UV_PYTHON_DOWNLOADS=never uv venv /opt/venv \
    --python 3.13

# Layer 2: all locked packages (torch, warp, mujoco-warp, ...), without the repo
# itself. Only pyproject.toml and uv.lock are mounted (not copied); the BuildKit
# cache mount keeps downloaded wheels between builds, so after a uv.lock change a
# rebuild downloads only what changed. --group dev: pytest etc., like `make sync`.
RUN --mount=type=cache,target=/root/.cache/uv-build \
    --mount=type=bind,source=uv.lock,target=/tmp/proj/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/tmp/proj/pyproject.toml \
    cd /tmp/proj && UV_CACHE_DIR=/root/.cache/uv-build \
    uv sync --frozen --extra cu130 --group dev --no-install-project

# Layer 3: pre-fetch the repo's build backend (uv_build, see pyproject.toml) into
# /opt/uv-cache, so SETUP_CMD's `uv sync` can install the repo with no internet.
RUN uv pip install --python /opt/venv --target /tmp/uvb "uv_build>=0.12.1,<0.13.0" \
    && rm -rf /tmp/uvb

CMD ["/bin/bash"]
