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
#  Resuming is automatic: RUN_NAME identifies the run. If checkpoints of RUN_NAME
#  exist, training continues from the newest one up to MAX_ITERS in total (in a
#  new <date>_<RUN_NAME>/ folder, same W&B run). After a crash the script restarts
#  by itself. Use a new RUN_NAME for a fresh run.
#  Runs longer than the 2-day job limit: scripts/submit_campaign.sh.
#  Frozen GPU processes write their Python stack to logs/hang_reports/.
#
# -----------------------------------------------------------------------------
#  Copy-paste commands (run from the repo root, ~/repos/mjlab)
# -----------------------------------------------------------------------------
#  Campaigns until a fixed end time (chained 2-day jobs), see submit_campaign.sh:
#    TASK=Mjlab-SGS-PegHole-FR3-Deck RUN_NAME=uniform_4x16k MAX_ITERS=1000000 DEADLINE="2026-11-04 10:00" bash src/mjlab/tasks/sgs_assembly/scripts/submit_campaign.sh
#    TASK=Mjlab-SGS-PegHole-FR3-Deck-SGS RUN_NAME=sgs_4x16k MAX_ITERS=1000000 DEADLINE="2026-11-04 10:00" bash src/mjlab/tasks/sgs_assembly/scripts/submit_campaign.sh
#
#  Uniform baseline (paper's "Uniform"), 4 A40s x 16384 envs, 1000 iterations:
#    sbatch --export=ALL,TASK=Mjlab-SGS-PegHole-FR3-Deck,RUN_NAME=uniform_4x16k,MAX_ITERS=1000 src/mjlab/tasks/sgs_assembly/scripts/train.sh
#
#  Success Guided Sampling (same everything, only the start sampling differs):
#    sbatch --export=ALL,TASK=Mjlab-SGS-PegHole-FR3-Deck-SGS,RUN_NAME=sgs_4x16k,MAX_ITERS=1000 src/mjlab/tasks/sgs_assembly/scripts/train.sh
#
#  Pose-observation variant (debugging; actor sees exact poses, no point cloud):
#    sbatch --export=ALL,TASK=Mjlab-SGS-PegHole-FR3-Deck-Pose,RUN_NAME=pose_4x16k,MAX_ITERS=1000 src/mjlab/tasks/sgs_assembly/scripts/train.sh
#
#  Short test (1 GPU, 3 iterations, ~5 min) before a long run:
#    sbatch --gres=gpu:a40:1 --cpus-per-task=6 --mem=32G --time=00:30:00 --export=ALL,RUN_NAME=test,MAX_ITERS=3,NUM_ENVS=4096 src/mjlab/tasks/sgs_assembly/scripts/train.sh
#
#  Inside an interactive GPU job instead of sbatch:
#    RUN_NAME=uniform_16k MAX_ITERS=1000 bash src/mjlab/tasks/sgs_assembly/scripts/train.sh
#
#  Watch / stop:
#    squeue -u $USER                                   # running and queued jobs
#    tail -f ~/slurm_logs/sgs_train_<jobid>.out         # live console (Ctrl+C to stop watching)
#    ls ~/repos/mjlab/logs/rsl_rl/sgs_fr3_peg_hole/     # run folders with checkpoints
#    scancel <jobid>                                   # stop a job
#  Curves: https://wandb.ai/s2816905-the-university-of-edinburgh/sgs-assembly
#    (Episode_Termination/success, SGS/success_rate/<family>, SGS/sample_share/<family>)
# =============================================================================
#SBATCH --job-name=sgs_train
# Either partition (whichever has 4 free A40s first); both may requeue a job,
# which train.sh then resumes like after a crash.
#SBATCH --partition=Open-Research,ICF-Free
#SBATCH --gres=gpu:a40:4
# Training is GPU-bound (one process per GPU); few CPUs are needed, and asking
# for many keeps the job waiting on busy nodes.
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
# Nodes with GPU errors (crannog05: CUDA fails to start, 2026-10-27).
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
# Optional end time (anything `date -d` reads, e.g. "2026-11-04 10:00"): training
# stops there even if MAX_ITERS is not reached; jobs starting later exit at once.
DEADLINE="${DEADLINE:-}"
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

# Run Python from the node's local disk: imports from the shared Lustre home can
# block for many minutes (seen 2026-10-27), and a GPU process blocked for 10 min
# makes NCCL abort the whole job. The environment is built from uv.lock (the same
# packages as ./.venv) once per node and reused by later jobs on that node. If
# that fails, the shared ./.venv is used as before.
LOCAL_DISK="/disk/scratch/$USER"
if mkdir -p "$LOCAL_DISK" 2>/dev/null; then
  export UV_CACHE_DIR="$LOCAL_DISK/uv_cache" WARP_CACHE_PATH="$LOCAL_DISK/warp_cache"
  if UV_PROJECT_ENVIRONMENT="$LOCAL_DISK/mjlab_venv" uv sync --frozen; then
    export UV_PROJECT_ENVIRONMENT="$LOCAL_DISK/mjlab_venv"
    echo "[train.sh] Python environment: $UV_PROJECT_ENVIRONMENT (local disk)"
  else
    echo "[train.sh] local environment failed -> using $MJLAB_ROOT/.venv"
    unset UV_CACHE_DIR WARP_CACHE_PATH
  fi
fi

# The reset deck is not in git; build it once if it is missing (~3 min).
if [[ ! -f logs/sgs_decks/fr3_peg_hole.npz ]]; then
  echo "[train.sh] reset deck missing -> building it"
  uv run python src/mjlab/tasks/sgs_assembly/scripts/build_deck.py
fi

# Experiment folder: one <date>_<RUN_NAME>/ folder per (re)start, plus per run
# <RUN_NAME>.wandb_id (keeps one W&B curve across restarts) and, when it gives up,
# <RUN_NAME>.stop.
EXP_DIR="$LOG_ROOT/sgs_fr3_peg_hole"
mkdir -p "$EXP_DIR"
STOP_FILE="$EXP_DIR/$RUN_NAME.stop"
ID_FILE="$EXP_DIR/$RUN_NAME.wandb_id"
[[ -s "$ID_FILE" ]] || tr -dc 'a-z0-9' </dev/urandom | head -c 8 >"$ID_FILE"
export WANDB_RUN_ID="$(cat "$ID_FILE")" WANDB_RESUME=allow
export SGS_HANG_DIR="$MJLAB_ROOT/logs/hang_reports"

echo "================================================================"
echo " task        : $TASK"
echo " envs        : $NUM_ENVS per GPU   total iterations: $MAX_ITERS"
echo " GPUs        : $GPU_IDS  (visible: ${CUDA_VISIBLE_DEVICES:-all})"
echo " run name    : $RUN_NAME     checkpoint every $SAVE_INTERVAL its"
echo " checkpoints : $EXP_DIR/<date>_${RUN_NAME}/"
echo " W&B run id  : $WANDB_RUN_ID"
echo " node / job  : $(hostname) / ${SLURM_JOB_ID:-interactive}"
echo " git commit  : $(git rev-parse --short HEAD) $(git diff --quiet || echo '(+ uncommitted changes)')"
echo "================================================================"

# Train until MAX_ITERS; after a crash, resume from the newest checkpoint. Stop
# for good (and write STOP_FILE, which later chained jobs obey) only if
# MAX_FAILS attempts in a row end without a new checkpoint: that is a real bug,
# not a passing cluster problem, and retrying would only burn GPU hours.
MAX_FAILS=3
fails=0
while true; do
  if [[ -e "$STOP_FILE" ]]; then
    echo "[train.sh] $STOP_FILE exists -> not training (delete it to continue)"
    exit 1
  fi
  TIME_LIMIT=()
  if [[ -n "$DEADLINE" ]]; then
    seconds_left=$(($(date -d "$DEADLINE" +%s) - $(date +%s)))
    if ((seconds_left < 600)); then
      echo "[train.sh] deadline $DEADLINE reached: done"
      exit 0
    fi
    TIME_LIMIT=(timeout "$seconds_left")
  fi
  latest=$(uv run python src/mjlab/tasks/sgs_assembly/scripts/latest_checkpoint.py \
    "$EXP_DIR" "$RUN_NAME")
  read -r ckpt_dir ckpt_file ckpt_iter <<<"$latest"
  ckpt_iter="${ckpt_iter:-0}"
  # model_<i>.pt is saved after iteration i (0-based): i + 1 iterations done.
  if ((ckpt_iter + 1 >= MAX_ITERS)); then
    echo "[train.sh] $RUN_NAME reached $MAX_ITERS iterations: done"
    exit 0
  fi
  RESUME_ARGS=()
  if [[ -n "$ckpt_dir" ]]; then
    RESUME_ARGS=(--agent.resume True --agent.load-run "$ckpt_dir"
      --agent.load-checkpoint "$ckpt_file")
  fi
  echo "[train.sh] $(date '+%F %T') start from ${ckpt_dir:-scratch}" \
    "${ckpt_file:-} (iteration $ckpt_iter), $((MAX_ITERS - ckpt_iter)) to go"

  "${TIME_LIMIT[@]}" uv run train "$TASK" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.max-iterations "$((MAX_ITERS - ckpt_iter))" \
    --agent.run-name "$RUN_NAME" \
    --agent.save-interval "$SAVE_INTERVAL" \
    --log-root "$LOG_ROOT" \
    --gpu-ids $GPU_IDS \
    "${RESUME_ARGS[@]}" \
    "$@"
  status=$?
  if [[ -n "$DEADLINE" ]] && (($(date +%s) >= $(date -d "$DEADLINE" +%s))); then
    echo "[train.sh] deadline $DEADLINE reached during training: done"
    exit 0
  fi
  find "$SGS_HANG_DIR" -empty -type f -delete 2>/dev/null
  echo "[train.sh] $(date '+%F %T') training exited with code $status"

  latest=$(uv run python src/mjlab/tasks/sgs_assembly/scripts/latest_checkpoint.py \
    "$EXP_DIR" "$RUN_NAME")
  read -r _ _ new_iter <<<"$latest"
  if ((${new_iter:-0} > ckpt_iter)); then
    fails=0
  else
    fails=$((fails + 1))
    echo "[train.sh] no new checkpoint ($fails/$MAX_FAILS in a row)"
    if ((fails >= MAX_FAILS)); then
      echo "[train.sh] giving up: see the errors above; wrote $STOP_FILE"
      touch "$STOP_FILE"
      exit 1
    fi
  fi
  sleep 60   # Let a passing GPU/filesystem problem clear before restarting.
done
