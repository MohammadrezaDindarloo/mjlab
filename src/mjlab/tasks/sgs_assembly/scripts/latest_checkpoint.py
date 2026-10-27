"""Print the newest loadable checkpoint of a run, for train.sh's automatic resume.

Looks in every <experiment_dir>/<date>_<run_name>/ folder (each restart writes a
new one) and prints "<folder name> <file name> <iteration>" of the checkpoint with
the highest iteration that torch can load. A checkpoint cut off by a crash or the
Slurm time limit fails to load and is skipped. Prints nothing if there is none.

  uv run python src/mjlab/tasks/sgs_assembly/scripts/latest_checkpoint.py \
    logs/rsl_rl/sgs_fr3_peg_hole sgs_4x16k
"""

import re
import sys
from pathlib import Path

import torch


def main(experiment_dir: str, run_name: str) -> None:
  found = []
  for path in Path(experiment_dir).glob(f"*_{run_name}/model_*.pt"):
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match and path.parent.name.endswith(f"_{run_name}"):
      found.append((int(match.group(1)), path))
  for iteration, path in sorted(found, reverse=True):
    try:
      torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # Cut-off file: try the previous one.
      print(f"skipping unreadable {path}: {error}", file=sys.stderr)
      continue
    print(path.parent.name, path.name, iteration)
    return


if __name__ == "__main__":
  main(*sys.argv[1:3])
