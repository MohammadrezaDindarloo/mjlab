"""Reset events.

Step 4 needs no custom event code: the peg is reset by mjlab's
``reset_root_state_uniform`` with an x/y/yaw range around its upright init pose
(see config/franka_peg/env_cfgs.py). This module is where the reset-deck event
(sample a deck entry -> write robot joints, gripper aperture and peg pose) will go.
"""
