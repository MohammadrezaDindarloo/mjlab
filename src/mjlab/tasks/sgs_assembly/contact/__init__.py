"""(Later) Contact-force logging and the friction-gap cost.

Planned: read per-contact normal force, tangential force and torsion between the
gripper pads and the held object (mjlab ContactSensor ``force``/``torque`` fields,
contact frame), and compute
  c_t = sum max(0, |f_t| - mu_real f_n) + max(0, |tau_z| - mu_r f_n),
to be used as a constraint cost for Lagrangian PPO.
"""
