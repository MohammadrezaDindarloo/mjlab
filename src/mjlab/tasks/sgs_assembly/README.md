# SGS assembly (FR3 peg-in-hole)

Goal: reproduce the manipulation setup of *Success Guided Sampling* (SGS) on a
Franka Research 3 in mjlab, then extend it with friction studies, a friction-gap
constraint for Lagrangian PPO, and FARO feasibility labels as an SGS prior.

## Layout

```
sgs_assembly/
├── physics_cfg.py          ONLY place for friction / condim / solver / timing numbers
├── env_cfg.py              generic base env: scene, sim, actions, proprio obs, resets
├── robots/__init__.py      RobotInfo: what env_cfg needs to know about any robot
├── robots/franka/          FR3 + Franka Hand (menagerie, unmodified XML + LICENSE)
├── objects/peg_hole/       square peg + fixed 4-wall hole, all sizes are parameters
├── config/franka_peg/      task variants + registration + PPO cfg
├── mdp/                    (Step 4) obs, Eq. 5 reward, success, terminations, events
├── resets/                 (later) reset deck: generation, save/load, ResetState
├── sampling/               (later) uniform and SGS samplers, one interface
├── contact/                (later) contact-force logging, friction-gap cost
├── feasibility/            (later) import/export for offline FARO labels
├── scripts/                inspection tools (no __init__.py, see below)
└── tests/                  CPU tests (no __init__.py, see below)
```

Rules that keep this extendable:
- Numbers that change contact behaviour live in `physics_cfg.py`; robot/object
  modules only decide *which* geoms collide.
- A new robot = a new `robots/<name>/` exposing a `RobotInfo`. A new object pair =
  a new `objects/<name>/`. A new task variant = a new `config/<robot>_<object>/`.
- `scripts/` and `tests/` have no `__init__.py` on purpose: mjlab imports every
  package under `mjlab/tasks/` at startup, and these must not run on import.

## Tasks

| Task id | Contents |
|---|---|
| `Mjlab-SGS-Robot-FR3` | FR3 + hand on the table (robot checks) |
| `Mjlab-SGS-PegHole-FR3` | + fixed hole and a free peg standing on the table |
| `Mjlab-SGS-PegHole-FR3-Deck` | same task, every episode starts from a reset-deck entry |

## How to run

```bash
uv run list-envs                                       # both tasks listed
uv run play Mjlab-SGS-PegHole-FR3 --agent zero --viewer viser
uv run python src/mjlab/tasks/sgs_assembly/scripts/inspect_scene.py \
  Mjlab-SGS-PegHole-FR3 --seconds 1.0                  # compiled params + contacts
uv run pytest src/mjlab/tasks/sgs_assembly/tests       # not part of `make test`
uv run python src/mjlab/tasks/sgs_assembly/scripts/grasp_friction.py --verbose
uv run python src/mjlab/tasks/sgs_assembly/scripts/run_agents.py   # see docstring
# Scripted pick-and-place (IK + state machine), watch it or get statistics:
uv run python src/mjlab/tasks/sgs_assembly/scripts/scripted_insert.py --viewer
uv run python src/mjlab/tasks/sgs_assembly/scripts/scripted_insert.py
# Reset deck -> logs/sgs_decks/fr3_peg_hole.npz, then watch episodes start from it:
uv run python src/mjlab/tasks/sgs_assembly/scripts/build_deck.py
uv run play Mjlab-SGS-PegHole-FR3-Deck --agent zero --viewer viser
```

## Verification results (Step 5)

- Success/reward (`tests/test_sgs_mdp.py`): seated peg -> S = True, r = 100.00;
  beside (60 mm), hovering above, and only 10 mm in -> S = False, r = 0.00.
- Agents (`run_agents.py`, 256 envs x 10 s): no NaNs, no failure terminations,
  0 line-search warnings for both zero and random agents.
- Pad friction (`grasp_friction.py`, grip 14.5 N total): during a slow pull-out
  the max per-contact |f_t|/f_n is exactly 0.500 / 1.000 / 2.000 for pad mu =
  0.5 / 1 / 2, and gross slip happens 5-11 % above the Coulomb estimate. MuJoCo
  friction is soft: below the limit the peg creeps (up to ~2.6 mm at mu = 2 for a
  slow 5 N/s pull), which matters for the later friction-gap constraint.

## Scripted insertion demo

`scripts/scripted_insert.py` grasps the peg and inserts it through real contact,
using `robots/ik.py` (damped least-squares IK, reused later by the reset deck)
and the same 8-D action a policy outputs. 64 envs x 40 s: 223 / 236 episodes
succeed (94 %), ~9 s each. Lessons:
- A velocity step (0 -> 0.25 m/s in one control step) made ~25 % of grasps drop
  the peg; an acceleration limit (0.5 m/s^2) brought that to ~6 %.
- The remaining drops are not friction-limited (|f_t|/f_n <= 0.7 vs mu = 2): the
  peg slowly rotates in the grasp, the soft pad contacts penetrate unequally
  (~1.0 vs ~0.3 mm), and at ~4.5 deg the peg is squeezed out. Contact stiffness
  (solref/solimp on the pads) is the open physics question here.

## Design decisions (and why)

- **FR3 v1 + MJX Franka Hand.** Menagerie's FR3 has no gripper; the Franka Hand is
  the same hardware on Panda and FR3. The MJX hand collides via boxes (fast on GPU)
  and has one box per finger that is the fingertip pad (`*_finger_pad`).
- **Gravity compensation on the robot**, like the real Franka controller; otherwise
  the PD loop sags about 7 mm at the TCP.
- **Gripper force limit 70 N** (Franka Hand continuous force) so grasp normal forces,
  and hence friction limits, are realistic. Stiffness 2000 N/m is our choice.
- **Finger armature 0.1 kg** (reflected motor/gear inertia). Without it the 15 g
  fingers are numerically unstable at dt = 5 ms with that stiffness (they
  oscillate and drop the peg) and their contacts are very soft.
- **Square peg first.** Box-box contacts are multi-point in MuJoCo Warp;
  cylinder-box gives one contact per pair. Round rod later behind a switch.
- **Friction combine = max** (all priorities 0), as in SGS. Keep object mu at or below
  pad mu when sweeping pad friction, or the object value wins.
- **`scale_rewards_by_dt=False`** so Eq. 5's +100 is not multiplied by 0.04 s.
- **`ls_iterations=50`.** With 20, random actions hit the line-search limit on about
  half of env steps (even with the robot alone); with 50 there were no warnings.
- **Arm-link collisions off** by default (only hand/fingers collide); flip
  `arm_collisions` in `franka_constants.py` if the arm hitting the table matters.

## MDP (Step 4), following SGS App. A.2, A.3.2, C.3

| | Definition | Where |
|---|---|---|
| Action (8) | arm: delta joint targets, `q* += 0.05 rad * a`; gripper: binary, `a > 0` open | `mdp/actions.py` |
| Actor obs (53) | joint pos, EE pose in base, previous action, peg & hole in EE frame, peg in hole frame | `mdp/observations.py` |
| Critic obs (62) | actor + joint velocities | `env_cfg.py` |
| Success S | tilt <= 5 deg, lateral <= 3 mm, depth >= 25 mm (either peg end; yaw free) | `mdp/success.py` |
| Failure F | arm joint speed > 2x limit, or peg outside workspace box | `mdp/terminations.py` |
| Reward (Eq. 5) | `100*1[S] - 1*1[F] - 1e-5*C(a^2) - 1e-4*C(da^2) - 1e-4*C(qd_arm^2)` | `mdp/rewards.py` |
| Episode end | S, F (terminal) or 10 s time-out | `env_cfg.py` |
| Reset | robot home; peg upright, +-0.1 m in x/y, any yaw | `config/franka_peg/` |

Choices where the paper is silent:
- **Delta arm targets** (not "home + scale * a"), so a zero action holds the pose
  and reset-deck states far from home are not undone on the first step.
- **Pose observations in the gripper frame** stand in for the point cloud; poses
  use position + the first two rotation-matrix columns (continuous).
- With `--agent zero` the gripper *closes* (binary command, `a = 0` means close).

## Roadmap

- [x] Step 0: read mjlab, explain how a task is assembled
- [x] Step 1: skeleton
- [x] Step 2: FR3 + hand, robot-only task viewable in viser
- [x] Step 3: peg + fixed hole, physics_cfg applied, contacts visible
- [x] Step 4: MDP terms (privileged obs, Eq. 5 reward, success gates, failures,
      time-out, simple random peg reset)
- [x] Step 5: verification without training (seated peg -> success & r~100,
      zero/random agents stable, scripted grasp pad-contact forces, warning counts)
- [ ] Reset deck, saved to disk (`resets/`, `scripts/build_deck.py`)
  - [x] 6a: deck format, Reaching family (10922 states, 88 % of candidates kept,
        ~2 min), `reset_from_deck` event, deck task. With the gripper closed
        from the start, ~61 % of Reaching states end up holding the peg.
        Arm posture: the FR3 has 7 joints for a 6D gripper pose, so the IK pulls
        the spare DoF toward a randomised comfortable posture (home +-0.3 rad,
        strict-priority null-space projection), picks the least-wrist-turn
        equivalent grasp, and rejects joints beyond 90 % of their range and arm
        self-collision. Before: 10 % of states had a joint at its limit; now 0 %.
  - [ ] 6b: Stable Grasp family
  - [ ] 6c: Near-Goal family (along the insertion path, fraction 0..1.1)
- [ ] Uniform-sampling PPO baseline
- [ ] SGS sampler
- [ ] Friction sweep reproducing the paper's pad-mu result (mu = 1, 2, 100)
- [ ] Contact logging (pad-peg normal, tangential, torsion)
- [ ] Friction-gap constraint (Lagrangian PPO)
- [ ] FARO feasibility labels as Beta prior for SGS
