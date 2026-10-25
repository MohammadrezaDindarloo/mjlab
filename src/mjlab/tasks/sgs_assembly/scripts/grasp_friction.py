"""Scripted grasp + pull-out test: is pad friction the parameter we think it is?

For each pad friction mu in ``--pad-mu``, on CPU MuJoCo (the same compiled model
MuJoCo Warp receives):
  1. hold the FR3 at home, place the peg vertically between the open fingers,
  2. close the gripper and settle for 1 s, then print every pad-peg contact
     (normal force, friction force, combined mu, torsion),
  3. pull the peg down with a force ramping from 0 at ``--ramp`` N/s until it
     slides off (relative speed > 2 cm/s).
What to look at:
- max per-contact |f_t| / f_n during the pull must saturate at exactly the
  combined contact mu = max(pad mu, peg mu) (MuJoCo's max rule): the friction
  cone limit. This is the direct check that pad friction is what we set.
- gross-slip force vs the Coulomb estimate mu * sum(f_n) - m g. MuJoCo friction
  is soft: the peg creeps slowly even below the limit (printed as "creep"), so
  slow pulls slip earlier than the Coulomb estimate.

  uv run python src/mjlab/tasks/sgs_assembly/scripts/grasp_friction.py
  uv run python src/mjlab/tasks/sgs_assembly/scripts/grasp_friction.py \
    --pad-mu 2.0 --verbose
"""

import mujoco
import numpy as np
import tyro

from mjlab.scene import Scene
from mjlab.tasks.sgs_assembly.config.franka_peg.env_cfgs import fr3_peg_hole_env_cfg
from mjlab.tasks.sgs_assembly.physics_cfg import PhysicsCfg

# Peg centre this far past the TCP along the finger axis, so the peg's upper end
# stays clear of the palm while the pads grip its top ~20 mm.
PEG_OFFSET_ALONG_FINGERS = 0.03  # [m]
GROSS_SLIP_SPEED = 0.02  # [m/s] of the peg relative to the hand.


def _build(pad_mu: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
  physics = PhysicsCfg()
  physics.contact.pad.sliding = pad_mu
  cfg = fr3_peg_hole_env_cfg(physics=physics)
  model = Scene(cfg.scene, device="cpu").compile()
  cfg.sim.mujoco.apply(model)
  return model, mujoco.MjData(model)


def _pad_peg_contacts(m: mujoco.MjModel, d: mujoco.MjData) -> list[dict]:
  peg = m.geom("peg/peg_geom").id
  pads = {m.geom(f"robot/{s}_finger_pad").id: s for s in ("left", "right")}
  out, wrench = [], np.zeros(6)
  for k in range(d.ncon):
    c = d.contact[k]
    pair = {c.geom1, c.geom2}
    pad = next((g for g in pads if g in pair), None)
    if pad is None or peg not in pair:
      continue
    # Contact-frame wrench: [normal, tangent1, tangent2, torsion, roll1, roll2].
    mujoco.mj_contactForce(m, d, k, wrench)
    out.append(
      {
        "pad": pads[pad],
        "fn": wrench[0],
        "ft": float(np.linalg.norm(wrench[1:3])),
        "torsion": wrench[3],
        "mu": c.friction[0],
      }
    )
  return out


def _peg_in_hand(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
  hand = m.body("robot/hand").id
  R = d.xmat[hand].reshape(3, 3)
  return R.T @ (d.xpos[m.body("peg/peg").id] - d.xpos[hand])


def run(pad_mu: float, ramp: float, verbose: bool) -> None:
  m, d = _build(pad_mu)
  mujoco.mj_resetDataKeyframe(m, d, 0)
  mujoco.mj_forward(m, d)

  # Hold the arm where it is; open the gripper.
  for i in range(m.nu):
    joint = m.actuator_trnid[i, 0]
    d.ctrl[i] = d.qpos[m.jnt_qposadr[joint]]
  grip = m.actuator("robot/finger_joint1").id
  d.ctrl[grip] = 0.04

  # Peg vertical between the fingers, faces parallel to the pads.
  site = m.site("robot/gripper").id
  R = d.site_xmat[site].reshape(3, 3)
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, d.site_xmat[site])
  adr = m.jnt_qposadr[m.joint("peg/peg_joint").id]
  d.qpos[adr : adr + 3] = d.site_xpos[site] + PEG_OFFSET_ALONG_FINGERS * R[:, 2]
  d.qpos[adr + 3 : adr + 7] = quat
  # Start the fingers 0.5 mm wider than the peg, otherwise the peg falls out
  # during the ~0.1 s it takes to close from fully open.
  half_width = m.geom_size[m.geom("peg/peg_geom").id][0]
  for name in ("robot/finger_joint1", "robot/finger_joint2"):
    d.qpos[m.jnt_qposadr[m.joint(name).id]] = half_width + 0.0005
  d.qvel[:] = 0.0
  mujoco.mj_forward(m, d)

  # Close and settle.
  d.ctrl[grip] = 0.0
  for _ in range(int(1.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
  contacts = _pad_peg_contacts(m, d)
  fn_total = sum(c["fn"] for c in contacts)
  ft_total = sum(c["ft"] for c in contacts)
  weight = m.body("peg/peg").mass[0] * -m.opt.gravity[2]
  mu_contact = contacts[0]["mu"] if contacts else float("nan")

  print(f"\n=== pad mu = {pad_mu}  (peg mu = 0.5; combined = {mu_contact:.2f})")
  print(
    f"after closing: {len(contacts)} pad-peg contacts, sum f_n = {fn_total:.2f} N,"
    f" sum |f_t| = {ft_total:.3f} N (peg weight {weight:.3f} N)"
  )
  if verbose:
    for c in contacts:
      print(
        f"  {c['pad']:5s} f_n={c['fn']:7.3f}  |f_t|={c['ft']:6.3f}  "
        f"|f_t|/f_n={c['ft'] / max(c['fn'], 1e-9):5.3f}  "
        f"torsion={c['torsion']:+.4f}  mu={c['mu']:.2f}"
      )

  # Pull down with a ramping force until the peg slides off (gross slip).
  peg_body = m.body("peg/peg").id
  p0 = _peg_in_hand(m, d)
  predicted = mu_contact * fn_total - weight  # Coulomb pull-out force.
  creep_at_half, max_ratio, slip_force = None, 0.0, None
  t, prev = 0.0, p0
  while ramp * t < 3.0 * predicted + 10.0:
    force = ramp * t
    d.xfrc_applied[peg_body, 2] = -force
    mujoco.mj_step(m, d)
    t += m.opt.timestep
    p = _peg_in_hand(m, d)
    speed = np.linalg.norm(p - prev) / m.opt.timestep
    prev = p
    for c in _pad_peg_contacts(m, d):
      max_ratio = max(max_ratio, c["ft"] / max(c["fn"], 1e-9))
    if creep_at_half is None and force >= 0.5 * predicted:
      creep_at_half = np.linalg.norm(p - p0)
    if speed > GROSS_SLIP_SPEED:
      slip_force = force
      break
  creep = f"{creep_at_half * 1e3:.2f} mm" if creep_at_half is not None else "n/a"
  print(
    f"max per-contact |f_t|/f_n during pull = {max_ratio:.3f}  "
    f"(cone limit = combined mu = {mu_contact:.2f})"
  )
  print(f"creep at half the Coulomb pull-out force: {creep}")
  if slip_force is None:
    print("no gross slip within the pull range")
  else:
    print(
      f"gross slip at pull = {slip_force:.2f} N  "
      f"(Coulomb estimate mu * sum f_n - m g = {predicted:.2f} N)"
    )


def main(
  pad_mu: tuple[float, ...] = (0.5, 1.0, 2.0),
  ramp: float = 20.0,
  verbose: bool = False,
) -> None:
  """Args:
  pad_mu: Pad sliding friction values to test.
  ramp: Pull-force ramp rate [N/s]; slower is closer to quasi-static.
  verbose: Print every pad-peg contact after closing.
  """
  for mu in pad_mu:
    run(mu, ramp, verbose)


if __name__ == "__main__":
  tyro.cli(main)
