"""Inverse kinematics for a site on a fixed-base arm (CPU MuJoCo, one pose at a time).

Damped least squares (Levenberg-Marquardt style) on the site Jacobian:

  e  = [p* - p ; log(R* R^T)]          6D pose error in the world frame
  dq = J^T (J J^T + lambda^2 I)^-1 e   J = [J_pos ; J_rot] over the arm joints
  q <- clip(q + dq, joint limits)      repeat until |e| is small

The damping lambda keeps steps bounded near singularities. Used by the scripted
insertion demo now and by the reset-deck generator later (gripper on grasp frame).
Robot-agnostic: give it the model, the site name and the arm joint names.
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
    self, q_init: np.ndarray, pos: np.ndarray, rot: np.ndarray
  ) -> tuple[np.ndarray, bool]:
    """Arm joints that put the site at (pos, rot). Returns (q, converged)."""
    q = q_init.copy()
    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, rot.reshape(-1))
    quat, quat_inv, dquat = np.zeros(4), np.zeros(4), np.zeros(4)
    err = np.zeros(6)
    for _ in range(self.max_iters):
      cur_pos, cur_rot = self.forward(q)
      err[:3] = pos - cur_pos
      mujoco.mju_mat2Quat(quat, cur_rot.reshape(-1))
      mujoco.mju_negQuat(quat_inv, quat)
      mujoco.mju_mulQuat(dquat, target_quat, quat_inv)
      mujoco.mju_quat2Vel(err[3:], dquat, 1.0)  # Rotation vector of R* R^T.
      if (
        np.linalg.norm(err[:3]) < self.pos_tol
        and np.linalg.norm(err[3:]) < self.rot_tol
      ):
        return q, True
      mujoco.mj_comPos(self.model, self.data)
      mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site)
      jac = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])
      jjt = jac @ jac.T + self.damping**2 * np.eye(6)
      q = np.clip(q + jac.T @ np.linalg.solve(jjt, err), self.lower, self.upper)
    return q, False
