#!/bin/bash
# =============================================================================
#  Train the SGS FR3 peg-in-hole policy (PPO, RSL-RL via mjlab).
#
#  Works both ways:
#    bash  src/mjlab/tasks/sgs_assembly/scripts/train.sh     # inside an interactive job
#    sbatch src/mjlab/tasks/sgs_assembly/scripts/train.sh    # as a batch job (long runs)
#
#  Settings are environment variables (defaults below), e.g.
#    RUN_NAME=uniform_16k MAX_ITERS=1000 bash .../train.sh
#    sbatch --export=ALL,RUN_NAME=uniform_16k,MAX_ITERS=5000 .../train.sh
#  Anything after the script name is passed on to `uv run train`, e.g.
#    bash .../train.sh --agent.algorithm.learning-rate 5e-4
#
#  Where things go (printed at start):
#    checkpoints  $LOG_ROOT/sgs_fr3_peg_hole/<date>_<RUN_NAME>/model_<iter>.pt
#                 (+ params/ with the exact env and agent config of the run)
#    curves       W&B project "sgs-assembly" (and TensorBoard files in the same dir)
#    console      $HOME/slurm_logs/sgs_train_<jobid>.out when run with sbatch
#
#  Multi-GPU (one node): every GPU runs NUM_ENVS envs and PPO averages the
#  gradients, so 4 GPUs x 16384 = 65536 envs per iteration at ~the same time per
#  iteration. sbatch requests 4 A40s by default (header below); GPU_IDS=all uses
#  every GPU the job was given. For a 1-GPU batch job:
#    sbatch --gres=gpu:a40:1 --cpus-per-task=6 --mem=32G .../train.sh
#
#  Resume an interrupted run: set RESUME_RUN to its directory name, e.g.
#    RESUME_RUN=2026-10-26_16-00-00_uniform_16k bash .../train.sh
#  (loads its latest model_*.pt and continues in a new run directory).
# =============================================================================
#SBATCH --job-name=sgs_train
#SBATCH --partition=Open-Research
#SBATCH --gres=gpu:a40:4
# Training is GPU-bound (one process per GPU); few CPUs are needed, and asking
# for many keeps the job waiting on busy nodes.
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# Nodes that failed with GPU errors in earlier projects on this cluster.
#SBATCH --exclude=crannog01,crannog05

# No `set -e`: a transient Lustre I/O error on a harmless line must not kill a
# multi-hour job. The training command's own exit code is returned at the end.
set -uo pipefail

MJLAB_ROOT="${MJLAB_ROOT:-$HOME/repos/mjlab}"
TASK="${TASK:-Mjlab-SGS-PegHole-FR3-Deck}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-1000}"
RUN_NAME="${RUN_NAME:-uniform}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"     # Checkpoint every N iterations.
LOG_ROOT="${LOG_ROOT:-$MJLAB_ROOT/logs/rsl_rl}"   # Absolute: never depends on cwd.
RESUME_RUN="${RESUME_RUN:-}"
GPU_IDS="${GPU_IDS:-all}"   # "all", or indices like "0 1" (into the job's GPUs).

export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV   # A stray conda env would make uv warn and ignore it anyway.

# Run from the repo root: the deck path (logs/sgs_decks/...) is relative to it, and
# W&B writes its local files to ./wandb/ (git-ignored).
cd "$MJLAB_ROOT" || { echo "MJLAB_ROOT not found: $MJLAB_ROOT"; exit 1; }

# Under sbatch, keep the console log next to the other slurm logs.
if [[ -n "${SLURM_JOB_ID:-}" && -f "sgs_train_${SLURM_JOB_ID}.out" ]]; then
  mkdir -p "$HOME/slurm_logs"
  mv "sgs_train_${SLURM_JOB_ID}.out" "$HOME/slurm_logs/" 2>/dev/null
  exec >>"$HOME/slurm_logs/sgs_train_${SLURM_JOB_ID}.out" 2>&1
fi

# The reset deck is not in git; build it once if it is missing (~3 min).
if [[ ! -f logs/sgs_decks/fr3_peg_hole.npz ]]; then
  echo "[train.sh] reset deck missing -> building it"
  uv run python src/mjlab/tasks/sgs_assembly/scripts/build_deck.py
fi

RESUME_ARGS=()
if [[ -n "$RESUME_RUN" ]]; then
  RESUME_ARGS=(--agent.resume True --agent.load-run "$RESUME_RUN")
fi

echo "================================================================"
echo " task        : $TASK"
echo " envs        : $NUM_ENVS per GPU   iterations: $MAX_ITERS"
echo " GPUs        : $GPU_IDS  (visible: ${CUDA_VISIBLE_DEVICES:-all})"
echo " run name    : $RUN_NAME     checkpoint every $SAVE_INTERVAL its"
echo " checkpoints : $LOG_ROOT/sgs_fr3_peg_hole/<date>_${RUN_NAME}/"
echo " resume from : ${RESUME_RUN:-<new run>}"
echo " node / job  : $(hostname) / ${SLURM_JOB_ID:-interactive}"
echo " git commit  : $(git rev-parse --short HEAD) $(git diff --quiet || echo '(+ uncommitted changes)')"
echo "================================================================"

uv run train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERS" \
  --agent.run-name "$RUN_NAME" \
  --agent.save-interval "$SAVE_INTERVAL" \
  --log-root "$LOG_ROOT" \
  --gpu-ids $GPU_IDS \
  "${RESUME_ARGS[@]}" \
  "$@"
status=$?
echo "[train.sh] finished with exit code $status"
exit $status
