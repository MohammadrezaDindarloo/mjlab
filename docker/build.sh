#!/bin/bash
# Build the environment image: docker/build.sh <tag> [--push]
#   docker/build.sh v1           # build mohammadrezadindarloo/mjlab:v1
#   docker/build.sh v1 --push    # build, then push to Docker Hub
# Never reuse a tag: every rebuild gets a new one (v2, v3, ...).
# For another project, copy docker/ and change PROJECT (and the Dockerfile).
set -euo pipefail

DOCKER_USER="mohammadrezadindarloo"
PROJECT="mjlab"

TAG="${1:?usage: docker/build.sh <tag> [--push]}"
PUSH="${2:-}"
IMAGE="$DOCKER_USER/$PROJECT:$TAG"

cd "$(dirname "$0")/.."   # The build context is the repo root.

if [[ "$PUSH" == --push ]] && docker manifest inspect "docker.io/$IMAGE" \
    >/dev/null 2>&1; then
  echo "$IMAGE already exists on Docker Hub: use a new tag." >&2
  exit 1
fi

start=$(date +%s)
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
  -f docker/Dockerfile -t "$IMAGE" \
  --label "org.opencontainers.image.revision=$(git rev-parse --short HEAD)" .
echo "Built $IMAGE in $(($(date +%s) - start)) s," \
  "$(docker image inspect "$IMAGE" --format '{{.Size}}' |
    awk '{printf "%.1f GB", $1 / 1e9}')"

if [[ "$PUSH" == --push ]]; then
  docker push "$IMAGE"
fi
