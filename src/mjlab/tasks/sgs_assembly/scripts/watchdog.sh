#!/bin/bash
# =============================================================================
#  Safety net for campaigns (submit_campaign.sh): every 20 min until DEADLINE,
#  any run in RUNS with no queued or running job (found by the job comment, which
#  submit_campaign.sh sets to RUN_NAME) gets a new chain of 5 jobs. That covers a
#  chain that ran out of jobs or was cancelled. A run with a <RUN_NAME>.stop file
#  (5 attempts in a row without a new checkpoint: a real bug) is left alone.
#
#  One CPU core; run from the repo root (~/repos/mjlab):
#    DEADLINE="2026-11-04 19:00" sbatch src/mjlab/tasks/sgs_assembly/scripts/watchdog.sh
#  Log: ~/slurm_logs/sgs_watchdog_<jobid>.out
# =============================================================================
#SBATCH --job-name=sgs_watchdog
#SBATCH --partition=ICF-Free
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out

set -uo pipefail
: "${DEADLINE:?set DEADLINE, e.g. \"2026-11-04 19:00\"}"
MJLAB_ROOT="${MJLAB_ROOT:-$HOME/repos/mjlab}"
MAX_ITERS="${MAX_ITERS:-1000000}"
EXP_DIR="$MJLAB_ROOT/logs/rsl_rl/sgs_fr3_peg_hole"
# "<task> <run name>" per run.
RUNS=(
  "Mjlab-SGS-PegHole-FR3-Deck uniform_4x16k"
  "Mjlab-SGS-PegHole-FR3-Deck-SGS sgs_4x16k"
)

cd "$MJLAB_ROOT" || exit 1
if [[ -n "${SLURM_JOB_ID:-}" && -f "sgs_watchdog_${SLURM_JOB_ID}.out" ]]; then
  mkdir -p "$HOME/slurm_logs"
  mv "sgs_watchdog_${SLURM_JOB_ID}.out" "$HOME/slurm_logs/"
  exec >>"$HOME/slurm_logs/sgs_watchdog_${SLURM_JOB_ID}.out" 2>&1
fi

echo "[watchdog] $(date '+%F %T') watching ${RUNS[*]} until $DEADLINE"
while (($(date +%s) < $(date -d "$DEADLINE" +%s) - 600)); do
  for run in "${RUNS[@]}"; do
    read -r task name <<<"$run"
    if squeue -u "$USER" -h -o "%k" | grep -qx "$name"; then
      continue
    fi
    if [[ -e "$EXP_DIR/$name.stop" ]]; then
      echo "[watchdog] $(date '+%F %T') $name has no jobs but is stopped: left alone"
      continue
    fi
    echo "[watchdog] $(date '+%F %T') $name has no jobs: submitting 5 more"
    TASK="$task" RUN_NAME="$name" MAX_ITERS="$MAX_ITERS" DEADLINE="$DEADLINE" \
      JOBS=5 bash src/mjlab/tasks/sgs_assembly/scripts/submit_campaign.sh
  done
  sleep 1200
done
echo "[watchdog] $(date '+%F %T') deadline $DEADLINE: done"
