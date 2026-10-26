"""Cross-GPU merge of deck outcomes, run with 2 CPU processes (gloo backend).

Each "GPU" records different episodes; after ``sync`` both must hold identical
histories containing all episodes.

  uv run pytest src/mjlab/tasks/sgs_assembly/tests/test_sgs_multigpu_sync.py
"""

import socket
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import spawn

from mjlab.tasks.sgs_assembly.mdp.events import reset_from_deck
from mjlab.tasks.sgs_assembly.sampling.sampler import DeckSampler, SgsCfg


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def _worker(rank: int, port: int, results) -> None:
  dist.init_process_group(
    "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
  )
  term = reset_from_deck.__new__(reset_from_deck)  # Only the merge state is needed.
  term._env = cast(Any, SimpleNamespace(device="cpu"))  # Stand-in env.
  term.sampler = DeckSampler(4, "sgs", SgsCfg(), device="cpu")
  term.pending = []
  # Rank 0 saw 3 episodes, rank 1 saw 1; config 2 appears on both.
  if rank == 0:
    term._record(torch.tensor([0, 2, 2]), torch.tensor([True, False, True]))
  else:
    term._record(torch.tensor([2]), torch.tensor([True]))
  assert term.sampler.count.sum() == 0  # Nothing applied before the merge.
  term.sync()
  results[rank] = (term.sampler.count.tolist(), term.sampler.success_rate().tolist())
  dist.destroy_process_group()


def test_sync_gives_every_gpu_all_outcomes():
  results = mp.Manager().dict()
  spawn(_worker, args=(_free_port(), results), nprocs=2, join=True)
  assert results[0] == results[1]
  counts, rates = results[0]
  assert counts == [1, 0, 3, 0]
  assert abs(rates[2] - 2 / 3) < 1e-6
