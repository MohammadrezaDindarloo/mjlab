"""Square peg and a fixed square hole, built with MjSpec from parameters.

Why square and not round: a round hole cannot be one convex primitive, and MuJoCo
Warp produces at most one contact per cylinder-box pair. Box-box contacts are
multi-point, so a square peg sliding in four box walls gives well-resolved wall
contacts. A round rod against a ring of boxes can be added later behind ``shape``.

Frames (all in the body frame of each entity):
- hole: origin on the table surface at the hole axis, +z up. The four walls stand
  on the table, so the hole bottom is the table itself (z = 0) and the entrance is
  at z = depth. Sites: ``hole_entrance`` (z = depth), ``hole_bottom`` (z = 0).
- peg: origin at the peg centre, +z along the peg axis. Sites: ``peg_tip`` (the
  end that goes in first, z = -length/2), ``peg_top`` (z = +length/2).
"""

from dataclasses import dataclass

import mujoco

from mjlab.entity import EntityCfg
from mjlab.tasks.sgs_assembly.physics_cfg import ContactCfg
from mjlab.utils.spec_config import CollisionCfg

PEG_RGBA = (0.85, 0.45, 0.1, 1.0)
HOLE_RGBA = (0.3, 0.4, 0.6, 1.0)


@dataclass
class PegHoleCfg:
  peg_width: float = 0.016
  """Side length of the square peg cross-section [m]."""
  peg_length: float = 0.08
  peg_mass: float = 0.05
  """About a 16 x 16 x 80 mm aluminium bar [kg]."""
  clearance: float = 0.001
  """Hole width minus peg width [m]; split equally on both sides."""
  hole_depth: float = 0.03
  wall_thickness: float = 0.01

  @property
  def hole_width(self) -> float:
    return self.peg_width + self.clearance


def make_peg_spec(cfg: PegHoleCfg) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="peg")
  body.add_freejoint(name="peg_joint")
  half = cfg.peg_width / 2
  body.add_geom(
    name="peg_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half, half, cfg.peg_length / 2),
    mass=cfg.peg_mass,
    rgba=PEG_RGBA,
  )
  body.add_site(name="peg_tip", pos=(0, 0, -cfg.peg_length / 2))
  body.add_site(name="peg_top", pos=(0, 0, cfg.peg_length / 2))
  return spec


def make_hole_spec(cfg: PegHoleCfg) -> mujoco.MjSpec:
  """Four walls around a square opening of side ``hole_width``.

  Top view (x right, y up); walls px/nx span the full outer width, py/ny fit
  between them:

      +-------------------+
      |        py         |
      +---+-----------+---+
      |nx |   hole    |px |
      +---+-----------+---+
      |        ny         |
      +-------------------+
  """
  spec = mujoco.MjSpec()
  # No joint: mjlab treats this as a fixed-base entity and wraps it as mocap so a
  # reset event can place it in each env.
  body = spec.worldbody.add_body(name="hole")
  w, t, d = cfg.hole_width, cfg.wall_thickness, cfg.hole_depth
  outer = w / 2 + t / 2  # Wall centre distance from the hole axis.
  walls = {
    "px": ((outer, 0.0), (t / 2, w / 2 + t)),
    "nx": ((-outer, 0.0), (t / 2, w / 2 + t)),
    "py": ((0.0, outer), (w / 2, t / 2)),
    "ny": ((0.0, -outer), (w / 2, t / 2)),
  }
  for name, ((x, y), (hx, hy)) in walls.items():
    body.add_geom(
      name=f"hole_wall_{name}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(hx, hy, d / 2),
      pos=(x, y, d / 2),
      rgba=HOLE_RGBA,
    )
  body.add_site(name="hole_entrance", pos=(0, 0, d))
  body.add_site(name="hole_bottom", pos=(0, 0, 0))
  return spec


def peg_entity_cfg(
  cfg: PegHoleCfg, contact: ContactCfg, pos: tuple[float, float, float]
) -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: make_peg_spec(cfg),
    init_state=EntityCfg.InitialStateCfg(pos=pos),
    collisions=(
      CollisionCfg(
        geom_names_expr=("peg_geom",),
        contype=1,
        conaffinity=1,
        condim=contact.object.condim,
        priority=0,
        friction=contact.object.friction,
      ),
    ),
  )


def hole_entity_cfg(
  cfg: PegHoleCfg, contact: ContactCfg, pos: tuple[float, float, float]
) -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: make_hole_spec(cfg),
    init_state=EntityCfg.InitialStateCfg(pos=pos),
    collisions=(
      CollisionCfg(
        geom_names_expr=("hole_wall_.*",),
        contype=1,
        conaffinity=1,
        condim=contact.fixture.condim,
        priority=0,
        friction=contact.fixture.friction,
      ),
    ),
  )
