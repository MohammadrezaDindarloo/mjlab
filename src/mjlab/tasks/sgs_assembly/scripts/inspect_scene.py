"""Print what MuJoCo actually compiled for a task, and the contacts after settling.

Builds the task's scene on CPU MuJoCo (same compiled MjModel that MuJoCo Warp
receives), prints the solver options and every colliding geom with its friction
and condim, then holds the home pose for ``--seconds`` and lists active contacts
with their normal and friction forces.

  uv run python src/mjlab/tasks/sgs_assembly/scripts/inspect_scene.py \
    Mjlab-SGS-PegHole-FR3 --seconds 1.0
"""

import mujoco
import numpy as np
import tyro

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def _geom_label(m: mujoco.MjModel, i: int) -> str:
  return m.geom(i).name or f"<geom {i} of {m.body(m.geom_bodyid[i]).name}>"


def main(task: str, /, seconds: float = 1.0) -> None:
  cfg = load_env_cfg(task, play=True)
  model = Scene(cfg.scene, device="cpu").compile()
  cfg.sim.mujoco.apply(model)
  o = model.opt
  print(
    f"timestep={o.timestep} decimation={cfg.decimation} "
    f"control_hz={1 / (o.timestep * cfg.decimation):.1f}"
  )
  print(
    f"cone={'elliptic' if o.cone else 'pyramidal'} impratio={o.impratio} "
    f"iterations={o.iterations} ls_iterations={o.ls_iterations}"
  )
  print(f"nu={model.nu} actuators={[model.actuator(i).name for i in range(model.nu)]}")

  print("\nColliding geoms: name | contype conaffinity | condim | friction")
  for i in range(model.ngeom):
    if model.geom_contype[i] or model.geom_conaffinity[i]:
      fr = np.array2string(model.geom_friction[i], precision=4)
      print(
        f"  {_geom_label(model, i):32s} | {model.geom_contype[i]} "
        f"{model.geom_conaffinity[i]} | {model.geom_condim[i]} | {fr}"
      )

  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)  # mjlab's merged "init_state" key.
  mujoco.mj_forward(model, data)
  nsteps = int(seconds / o.timestep)
  for _ in range(nsteps):
    mujoco.mj_step(model, data)

  print(f"\nAfter {seconds} s ({nsteps} steps): {data.ncon} contacts")
  print("  geom1 | geom2 | dim | combined mu | f_normal [N] | |f_tangent| [N]")
  wrench = np.zeros(6)
  for k in range(data.ncon):
    c = data.contact[k]
    # Wrench in the contact frame: [normal, tangent1, tangent2, torsion, ...].
    mujoco.mj_contactForce(model, data, k, wrench)
    print(
      f"  {_geom_label(model, c.geom1):24s} | {_geom_label(model, c.geom2):24s}"
      f" | {c.dim} | {c.friction[0]:.3f} | {wrench[0]:9.3f} |"
      f" {np.linalg.norm(wrench[1:3]):9.3f}"
    )
  for name in ("robot/gripper", "peg/peg_tip", "hole/hole_entrance"):
    try:
      print(f"site {name}: {np.round(data.site(name).xpos, 4)}")
    except KeyError:
      pass


if __name__ == "__main__":
  tyro.cli(main)
