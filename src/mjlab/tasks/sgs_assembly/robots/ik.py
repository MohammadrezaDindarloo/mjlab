"""Inverse kinematics for a site on a fixed-base arm (CPU MuJoCo, one pose at a time).

Damped least squares (Levenberg-Marquardt style) on the site Jacobian:

  e  = [p* - p ; log(R* R^T)]          6D pose error in the world frame
  dq = J^T (J J^T + lambda^2 I)^-1 e   J = [J_pos ; J_rot] over the arm joints
  q <- clip(q + dq, joint limits)      repeat until |e| is small

The damping lambda keeps steps bounded near singularities. Used by the scripted
insertion demo and by the reset-deck generator (gripper on grasp frame).
Robot-agnostic: give it the model, the site name and the arm joint names.

Redundancy: a 7-joint arm has one more joint than a 6D pose needs, so many arm
postures give the same gripper pose. If ``q_nominal`` is given, each step also
moves q toward it inside the null space of J (the joint motions that do not move
the gripper):  dq += (I - J^+ J) * gain * (q_nominal - q).

This is strict-priority redundancy resolution by gradient projection (Liegeois
1977; Siciliano et al., "Robotics: Modelling, Planning and Control", ch. 3): the
pose is exact, the posture uses only the spare DoF. mjlab's DifferentialIKAction
instead adds the posture as a *weighted* least-squares task (soft priority, as in
Pinocchio/mink-style solvers), which trades a small pose error for posture; deck
states need the gripper exactly on a grasp frame, hence strict priority here.
Joint limits are enforced by clipping (a QP solver would make them hard
inequality constraints); the deck generator additionally rejects near-limit
solutions.
"""

import mujoco
import numpy as np


class SiteIK:
  def __init__(
    self,
    model: mujoco.MjModel,
    site: str,
    joints: list[str],
    damping: float = 1e-2,
    pos_tol: float = 1e-5,
    rot_tol: float = 1e-4,
    max_iters: int = 100,
  ):
    self.model = model
    self.data = mujoco.MjData(model)  # Scratch data: IK never touches the sim.
    self.site = model.site(site).id
    ids = [model.joint(j).id for j in joints]
    self.qadr = np.array([model.jnt_qposadr[i] for i in ids])
    self.dadr = np.array([model.jnt_dofadr[i] for i in ids])
    self.lower = model.jnt_range[ids, 0]
    self.upper = model.jnt_range[ids, 1]
    self.damping, self.pos_tol, self.rot_tol = damping, pos_tol, rot_tol
    self.max_iters = max_iters
    self._jacp = np.zeros((3, model.nv))
    self._jacr = np.zeros((3, model.nv))

  def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Site position (3,) and rotation matrix (3, 3) for arm joints q."""
    self.data.qpos[self.qadr] = q
    mujoco.mj_kinematics(self.model, self.data)
    pos = self.data.site_xpos[self.site].copy()
    return pos, self.data.site_xmat[self.site].reshape(3, 3).copy()

  def solve(
    self,
    q_init: np.ndarray,
    pos: np.ndarray,
    rot: np.ndarray,
    q_nominal: np.ndarray | None = None,
    nullspace_gain: float = 0.5,
  ) -> tuple[np.ndarray, bool]:
    """Arm joints that put the site at (pos, rot). Returns (q, converged).

    With ``q_nominal`` the solve has two phases. Phase 1 tracks the pose *and*
    pulls the posture toward ``q_nominal`` until q stops changing; because the
    damped projector is not an exact null space, the two terms balance with a
    small residual pose error (~0.3 mm). Phase 2 then removes that residual with
    pose-only steps, which barely move the posture.
    """
    q = q_init.copy()
    if q_nominal is not None:
      for _ in range(self.max_iters):
        jac, err = self._jacobian_and_error(q, pos, rot)
        jac_pinv = self._damped_pinv(jac)
        null = np.eye(len(q)) - jac_pinv @ jac
        dq = jac_pinv @ err + null @ (nullspace_gain * (q_nominal - q))
        q_new = np.clip(q + dq, self.lower, self.upper)
        if np.linalg.norm(q_new - q) < 1e-6:
          break
        q = q_new
    for _ in range(self.max_iters):
      jac, err = self._jacobian_and_error(q, pos, rot)
      if (
        np.linalg.norm(err[:3]) < self.pos_tol
        and np.linalg.norm(err[3:]) < self.rot_tol
      ):
        return q, True
      q = np.clip(q + self._damped_pinv(jac) @ err, self.lower, self.upper)
    return q, False

  def _damped_pinv(self, jac: np.ndarray) -> np.ndarray:
    return jac.T @ np.linalg.inv(jac @ jac.T + self.damping**2 * np.eye(6))

  def _jacobian_and_error(
    self, q: np.ndarray, pos: np.ndarray, rot: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Site Jacobian over the arm joints (6, n) and 6D pose error at q."""
    cur_pos, cur_rot = self.forward(q)
    err = np.zeros(6)
    err[:3] = pos - cur_pos
    quat, target, quat_inv, dquat = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(target, rot.reshape(-1))
    mujoco.mju_mat2Quat(quat, cur_rot.reshape(-1))
    mujoco.mju_negQuat(quat_inv, quat)
    mujoco.mju_mulQuat(dquat, target, quat_inv)
    mujoco.mju_quat2Vel(err[3:], dquat, 1.0)  # Rotation vector of R* R^T.
    mujoco.mj_comPos(self.model, self.data)
    mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site)
    return np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]]), err
