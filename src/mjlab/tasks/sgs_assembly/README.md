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

## How to run

```bash
uv run list-envs                                       # both tasks listed
uv run play Mjlab-SGS-PegHole-FR3 --agent zero --viewer viser
uv run python src/mjlab/tasks/sgs_assembly/scripts/inspect_scene.py \
  Mjlab-SGS-PegHole-FR3 --seconds 1.0                  # compiled params + contacts
uv run pytest src/mjlab/tasks/sgs_assembly/tests       # not part of `make test`
```

## Design decisions (and why)

- **FR3 v1 + MJX Franka Hand.** Menagerie's FR3 has no gripper; the Franka Hand is
  the same hardware on Panda and FR3. The MJX hand collides via boxes (fast on GPU)
  and has one box per finger that is the fingertip pad (`*_finger_pad`).
- **Gravity compensation on the robot**, like the real Franka controller; otherwise
  the PD loop sags about 7 mm at the TCP.
- **Gripper force limit 70 N** (Franka Hand continuous force) so grasp normal forces,
  and hence friction limits, are realistic. Stiffness 2000 N/m is our choice.
- **Square peg first.** Box-box contacts are multi-point in MuJoCo Warp;
  cylinder-box gives one contact per pair. Round rod later behind a switch.
- **Friction combine = max** (all priorities 0), as in SGS. Keep object mu at or below
  pad mu when sweeping pad friction, or the object value wins.
- **`scale_rewards_by_dt=False`** so Eq. 5's +100 is not multiplied by 0.04 s.
- **`ls_iterations=50`.** With 20, random actions hit the line-search limit on about
  half of env steps (even with the robot alone); with 50 there were no warnings.
- **Arm-link collisions off** by default (only hand/fingers collide); flip
  `arm_collisions` in `franka_constants.py` if the arm hitting the table matters.

## Roadmap

- [x] Step 0: read mjlab, explain how a task is assembled
- [x] Step 1: skeleton
- [x] Step 2: FR3 + hand, robot-only task viewable in viser
- [x] Step 3: peg + fixed hole, physics_cfg applied, contacts visible
- [ ] Step 4: MDP terms (privileged obs, Eq. 5 reward, success gates, failures,
      time-out, simple random peg reset)
- [ ] Step 5: verification without training (seated peg -> success & r~100,
      zero/random agents stable, scripted grasp pad-contact forces, warning counts)
- [ ] Reset deck (Near-Goal / Stable Grasp / Reaching), saved to disk
- [ ] Uniform-sampling PPO baseline
- [ ] SGS sampler
- [ ] Friction sweep reproducing the paper's pad-mu result (mu = 1, 2, 100)
- [ ] Contact logging (pad-peg normal, tangential, torsion)
- [ ] Friction-gap constraint (Lagrangian PPO)
- [ ] FARO feasibility labels as Beta prior for SGS
