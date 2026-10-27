#!/bin/bash
# =============================================================================
#  Train until a fixed end time, longer than the cluster's 2-day job limit, with
#  nobody logged in. All jobs are queued now as a chain: job k+1 starts when job
#  k ends, however it ended (time limit, crash, node failure: `afterany`), and
#  train.sh resumes the run from its newest checkpoint. Inside a job, train.sh
#  also restarts training after crashes, so a crash does not use up a job.
#  Jobs that start after DEADLINE exit within seconds, so JOBS is set generously
#  (default 10) to cover queue waits and jobs that end early.
#
#  Settings (environment variables): TASK, RUN_NAME, MAX_ITERS as for train.sh,
#  DEADLINE (required), JOBS. Run from the repo root (~/repos/mjlab):
#
#    TASK=Mjlab-SGS-PegHole-FR3-Deck RUN_NAME=uniform_4x16k MAX_ITERS=1000000 \
#      DEADLINE="2026-11-04 10:00" bash src/mjlab/tasks/sgs_assembly/scripts/submit_campaign.sh
#    TASK=Mjlab-SGS-PegHole-FR3-Deck-SGS RUN_NAME=sgs_4x16k MAX_ITERS=1000000 \
#      DEADLINE="2026-11-04 10:00" bash src/mjlab/tasks/sgs_assembly/scripts/submit_campaign.sh
#
#  The chain also ends early if the run reaches MAX_ITERS.
#  Stop it:   scancel <all job ids printed below>
#  or, gently (the current job keeps training, later jobs exit at once):
#    touch logs/rsl_rl/sgs_fr3_peg_hole/<RUN_NAME>.stop
#  Continue after a stop: delete that .stop file and submit again.
# =============================================================================
set -euo pipefail

: "${TASK:?set TASK}" "${RUN_NAME:?set RUN_NAME}" "${MAX_ITERS:?set MAX_ITERS}"
: "${DEADLINE:?set DEADLINE, e.g. \"2026-11-04 10:00\"}"
date -d "$DEADLINE" >/dev/null || exit 1
JOBS="${JOBS:-10}"
SCRIPT="src/mjlab/tasks/sgs_assembly/scripts/train.sh"
export TASK RUN_NAME MAX_ITERS DEADLINE

previous=""
ids=()
for ((k = 1; k <= JOBS; k++)); do
  dependency=()
  [[ -n "$previous" ]] && dependency=(--dependency="afterany:$previous")
  previous=$(sbatch --parsable "${dependency[@]}" --export=ALL "$SCRIPT")
  ids+=("$previous")
done
echo "$RUN_NAME: $JOBS chained jobs until $DEADLINE: ${ids[*]}"
