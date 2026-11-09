# mjlab environment image

`mohammadrezadindarloo/mjlab:<tag>` holds only the environment: CUDA 13 base,
Python 3.13 and every package in `uv.lock` (cu130 torch, warp, mujoco-warp, ...)
in `/opt/venv`. The repo's code is not in it: a job mounts or uploads the repo
and runs

```sh
uv sync --frozen --extra cu130        # SETUP_CMD: installs the repo, ~1 s, offline
```

If `uv.lock` changed since the image was built, this downloads only the changed
packages (needs internet); rebuild the image when convenient.

## Rebuild

```sh
docker/build.sh v2            # build mohammadrezadindarloo/mjlab:v2
docker/build.sh v2 --push     # build and push (docker login first)
```

**Use a new tag for every rebuild and never reuse one.** Jobs and nodes cache
images by tag, so an overwritten tag gives different machines different
environments. `build.sh --push` refuses a tag that already exists on Docker Hub.
Rebuild when `uv.lock` or `docker/Dockerfile` changes.

The image is public: never put keys (W&B), licences or data in it. Pass
`WANDB_API_KEY` at job time.

## Run locally (as EIDF does: repo mounted, run as root)

```sh
mkdir -p /tmp/out
docker run --rm --gpus all -v "$PWD":/code -w /code -v /tmp/out:/out \
  -e OUTPUT_DIR=/out mohammadrezadindarloo/mjlab:v1 bash -c '
  uv sync --frozen --extra cu130 -q &&
  MJLAB_ROOT=$PWD LOG_ROOT=$OUTPUT_DIR WANDB_DIR=$OUTPUT_DIR WANDB_MODE=offline \
  MAX_ITERS=3 NUM_ENVS=4096 RUN_NAME=test \
  bash src/mjlab/tasks/sgs_assembly/scripts/train.sh'
```

Files written to `/out` are owned by root; remove them with
`docker run --rm -v /tmp:/t ubuntu:24.04 rm -rf /t/out`.

Interactive shell: `docker run --rm -it --gpus all -v "$PWD":/code -w /code
mohammadrezadindarloo/mjlab:v1`.

## Apptainer (Slurm)

`apptainer pull mjlab_v1.sif docker://mohammadrezadindarloo/mjlab:v1`, then
`apptainer exec --nv mjlab_v1.sif ...`. The image is read-only there, so instead
of `uv sync` use `PYTHONPATH=$PWD/src python -m mjlab.scripts.train ...`.

## For another project

Copy `docker/` and `.dockerignore`, set `PROJECT` in `build.sh`, and change the
Dockerfile's install layers.
