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

Contact stiffness (solref/solimp; averaged when two geoms touch): MuJoCo contacts
are soft, and the overlap a force F causes grows like F / (object mass x
stiffness). With MuJoCo's defaults (solref 0.02, solimp dmax 0.95) and dt = 5 ms,
the 50 g peg went through the 10 mm hole walls at the ~20 N the arm pushes with
(measured in rollouts: median 22 N, 99th percentile 90 N), and PPO learned to
insert through the walls. A peg pushed into a wall alone:
  dt 5 ms, any stiffness       -> through at 50 N
  dt 2 ms, solref 4 ms, dmax .999 -> 0.11 mm at 100 N, through at 200 N
  dt 1 ms, same                -> 0.05 mm at 200 N
We use dt = 2 ms plus a penetration failure term for the rare larger pushes.
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
  # (timeconst, dampratio): stiffer than MuJoCo's (0.02, 1). MuJoCo raises
  # timeconst to 2 x timestep if it is smaller, so keep it at 2 x SolverCfg.timestep.
  solref: tuple[float, float] = (0.004, 1.0)
  # (dmin, dmax, width, midpoint, power): impedance near 1 = nearly hard contact.
  solimp: tuple[float, float, float, float, float] = (0.99, 0.999, 0.001, 0.5, 2.0)

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

  Control rate = 1 / (timestep * decimation) = 25 Hz, as in SGS. The paper uses
  5 ms x 8; MuJoCo's soft contacts need 2 ms x 20 here (see the module docstring).
  """

  timestep: float = 0.002
  decimation: int = 20
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
  # overflowed, which silently drops constraints. 160 still overflowed in long
  # training runs (11-03: "nefc overflow", then NaN observations and a crash), so
  # njmax is doubled; at 16384 envs one A40 peaked at 23 GB with 160.
  nconmax: int | None = 48
  njmax: int | None = 320

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
