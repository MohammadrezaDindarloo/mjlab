"""Scene-level checks for the SGS assembly tasks (CPU only, no simulation).

uv run pytest src/mjlab/tasks/sgs_assembly/tests
"""

import mujoco
import numpy as np
import pytest

from mjlab.scene import Scene
from mjlab.tasks.sgs_assembly.config.franka_peg.env_cfgs import fr3_peg_hole_env_cfg
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import PegHoleCfg
from mjlab.tasks.sgs_assembly.physics_cfg import PhysicsCfg, SurfaceCfg


def _compile(**kwargs) -> mujoco.MjModel:
  cfg = fr3_peg_hole_env_cfg(**kwargs)
  model = Scene(cfg.scene, device="cpu").compile()
  cfg.sim.mujoco.apply(model)
  return model


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return _compile()


def test_actuators_are_mjlabs_not_menageries(model):
  names = [model.actuator(i).name for i in range(model.nu)]
  assert names == [f"robot/fr3_joint{i}" for i in range(1, 8)] + ["robot/finger_joint1"]


def test_solver_settings_from_physics_cfg(model):
  solver = PhysicsCfg().solver
  assert model.opt.timestep == solver.timestep
  assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
  assert model.opt.impratio == solver.impratio
  assert model.opt.ls_iterations == solver.ls_iterations


def test_pad_friction_and_condim_follow_physics_cfg():
  physics = PhysicsCfg()
  physics.contact.pad = SurfaceCfg(sliding=3.3, torsional=0.01, condim=6)
  model = _compile(physics=physics)
  for side in ("left", "right"):
    g = model.geom(f"robot/{side}_finger_pad").id
    np.testing.assert_allclose(model.geom_friction[g], (3.3, 0.01, 1e-4))
    assert model.geom_condim[g] == 6
  # Non-pad finger geoms keep the generic robot surface.
  g = model.geom("robot/left_finger_box_1").id
  assert model.geom_friction[g][0] == physics.contact.robot.sliding


def test_hole_opening_equals_peg_plus_clearance():
  ph = PegHoleCfg(peg_width=0.02, clearance=0.0005)
  model = _compile(peg_hole=ph)
  px, nx = (model.geom(f"hole/hole_wall_{n}").id for n in ("px", "nx"))
  inner_px = model.geom_pos[px][0] - model.geom_size[px][0]
  inner_nx = model.geom_pos[nx][0] + model.geom_size[nx][0]
  assert inner_px - inner_nx == pytest.approx(0.0205)
  peg = model.geom("peg/peg_geom").id
  assert 2 * model.geom_size[peg][0] == pytest.approx(0.02)
