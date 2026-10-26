"""All contact, friction and solver parameters for the SGS assembly tasks.

Every number that changes contact behaviour lives here, so a friction sweep or a
solver study only overrides one dataclass. Robot and object modules decide *which*
geoms collide (names and groups); this module decides *how* they collide.

How MuJoCo combines two touching geoms (both have priority 0 here):
- friction is the element-wise max of the two geoms' (sliding, torsional, rolling),
- condim is the max of the two geoms' condim.
So pad-peg sliding friction is max(pad, object). Keep ``object.sliding`` at or below
``pad.sliding`` when sweeping pad friction, or the object value silently wins.

condim: 3 = sliding friction only, 4 = + torsional (spin about the contact normal),
6 = + rolling. Torsional friction (needed for the friction-gap |tau_z| term) only
exists for condim >= 4. Its coefficient has units of length: max torsion is
torsional * f_n.
"""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.sim import MujocoCfg, SimulationCfg


@dataclass
class SurfaceCfg:
  """Friction coefficients and contact dimension for one group of geoms."""

  sliding: float
  torsional: float = 0.005
  rolling: float = 0.0001
  condim: Literal[1, 3, 4, 6] = 3

  @property
  def friction(self) -> tuple[float, float, float]:
    return (self.sliding, self.torsional, self.rolling)


@dataclass
class ContactCfg:
  """Per-surface contact parameters."""

  # Gripper fingertip pads. mu = 2 is the lowest value that trained in the SGS
  # paper; condim 4 so torsional friction exists.
  pad: SurfaceCfg = field(default_factory=lambda: SurfaceCfg(sliding=2.0, condim=4))
  # All other robot collision geoms (finger bodies, hand, optionally arm links).
  robot: SurfaceCfg = field(default_factory=lambda: SurfaceCfg(sliding=1.0))
  # The held object (peg).
  object: SurfaceCfg = field(default_factory=lambda: SurfaceCfg(sliding=0.5))
  # The fixed receptacle (hole walls).
  fixture: SurfaceCfg = field(default_factory=lambda: SurfaceCfg(sliding=0.5))
  # The table (ground plane).
  table: SurfaceCfg = field(default_factory=lambda: SurfaceCfg(sliding=0.5))


@dataclass
class SolverCfg:
  """MuJoCo solver and timing settings.

  Control rate = 1 / (timestep * decimation) = 25 Hz with the defaults, as in SGS.
  """

  timestep: float = 0.005
  decimation: int = 8
  integrator: Literal["euler", "implicitfast"] = "implicitfast"
  # Elliptic cones model the friction cone exactly (pyramidal approximates it with
  # facets). impratio > 1 makes friction stiffer relative to the normal direction,
  # which reduces slip in grasps; MuJoCo recommends it with elliptic cones.
  cone: Literal["pyramidal", "elliptic"] = "elliptic"
  impratio: float = 10.0
  iterations: int = 50
  tolerance: float = 1e-8
  # With 20, random actions hit the line-search limit on ~half of env steps (even
  # robot-only: stiff arm gains + joint limits); 50 removed every warning.
  ls_iterations: int = 50
  ls_tolerance: float = 0.01
  # Per-world contact / constraint-row buffer sizes. The MuJoCo Warp heuristic
  # chose njmax = 64, but random actions reached 58 rows in one world (a grasp is
  # ~8 contacts x 4 rows, elliptic condim 4) and at 16K envs some worlds
  # overflowed, which silently drops constraints. ~3x the measured peaks:
  nconmax: int | None = 48
  njmax: int | None = 160

  def to_simulation_cfg(self) -> SimulationCfg:
    return SimulationCfg(
      nconmax=self.nconmax,
      njmax=self.njmax,
      mujoco=MujocoCfg(
        timestep=self.timestep,
        integrator=self.integrator,
        cone=self.cone,
        impratio=self.impratio,
        iterations=self.iterations,
        tolerance=self.tolerance,
        ls_iterations=self.ls_iterations,
        ls_tolerance=self.ls_tolerance,
      ),
    )


@dataclass
class PhysicsCfg:
  contact: ContactCfg = field(default_factory=ContactCfg)
  solver: SolverCfg = field(default_factory=SolverCfg)
