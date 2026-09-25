# Expert intervention for a dVRK drawing policy

A Diffusion Policy draws a circle or a rectangle with the dVRK. When it starts
to drift out of distribution, the operator holds the **COAG footpedal**, the
policy stops commanding the arms, and the console's own teleoperation takes
over. Releasing the pedal hands control back. Every takeover is recorded and
can be folded into the training set, so the next round of training has seen the
situations the last one got wrong.

The imitation-learning half follows
[`Diffusion_Policy_for_dVRK`](../Diffusion_Policy_for_dVRK) — same 20D
end-effector representation, same converter conventions, same training entry
point — so a checkpoint trained here is interchangeable with one trained there.
The intervention half is new.

```text
        demonstrations                  data_collection.sh
              |
              v
        episodes -> Zarr                data_processing/convert_drawing_6d_abs.py
              |
              v
        train                           train_drawing_policy.sh
              |
              v
   +---> closed-loop deploy             deploy/deploy_with_intervention.py
   |          |
   |          |  COAG held -> align -> surgeon drives
   |          |  COAG up   -> align -> policy resumes
   |          v
   |    run directory (30 Hz, every frame labelled policy / expert)
   |          |
   |          v
   |    extract corrections             dagger/merge_interventions.py
   |          |
   +----------+  convert + retrain on demos + corrections
```

## 1. Where each step runs

Nothing has every dependency, and the split is not negotiable: the dVRK
desktop's ROS install has `rclpy` and OpenCV but no torch, and the training
container has torch, zarr and pytorch3d but its conda Python cannot load ROS's
C extensions.

| Step | Runs on | Needs |
|---|---|---|
| Collect demonstrations | dVRK desktop | ROS 2 Humble + dVRK stack |
| Convert to Zarr | training container | zarr, pytorch3d, OpenCV |
| Train | training container | the above + a GPU |
| Deploy with intervention | dVRK desktop | ROS 2 + dVRK stack + torch |
| Merge corrections | anywhere | Python standard library |

The container is the one from the sibling project and is already built on this
machine:

```bash
bash start_container.sh            # interactive shell
bash start_container.sh -- CMD ... # one command
```

The **deploy** step is the awkward one: it needs `rclpy` *and* torch in the same
interpreter. That is the dVRK control desktop's own environment, with torch
installed into it — not this container, which has no working `rclpy`.


## 2. Run book

Every command in the project, in the order you run it. Steps 1–3 happen once to
get a first checkpoint; steps 4–7 are the correction loop, and you go round them
as many times as it takes.

### The two shells

Everything below runs in one of these.

**ROS shell** — the dVRK desktop, or any machine with ROS 2 Humble, for anything
that touches a topic (collection and deploy):

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=111          # must match the console; 111 is data_collection.sh's default
cd /path/to/expert_intervention
```

Deploying additionally needs torch in that same interpreter — the dVRK desktop's
own environment, not the container, which has no working `rclpy`.

**Container shell** — converting and training only:

```bash
bash start_container.sh           # interactive shell; then, inside it:
conda activate robodiff
cd /docker-ros/ws
```

`bash start_container.sh -- CMD ...` runs a single command instead.

### First checkpoint

```bash
# --- ROS shell, dVRK desktop ------------------------------------------------
# 1. record demonstrations. PSM1 jaw pinch 3x -> start, PSM2 jaw pinch 3x ->
#    stop + save, Ctrl+C -> quit. Writes data/drawing_square/episode_NNNN/.
bash data_collection.sh --task square

# --- container shell --------------------------------------------------------
# 2. episodes -> Diffusion Policy Zarr (5 Hz, 160x120, absolute actions)
python data_processing/convert_drawing_6d_abs.py \
    data/drawing_square data/diffusion_policy/square.zarr

# 3. train. --check validates the dataset adapter and exits; --smoke is two
#    epochs of three batches; --train is the real thing. Extra args -> Hydra.
DP_DATASET=data/diffusion_policy/square.zarr \
DP_OUTPUT=outputs/square_round0 \
bash train_drawing_policy.sh --train
```

### The correction loop

```bash
# --- ROS shell, dVRK desktop (needs torch too) ------------------------------
# 4. dry run: every code path except publishing, including the pedal handover
python deploy/deploy_with_intervention.py \
    --checkpoint outputs/square_round0/checkpoints/latest.ckpt --task square

# 5. for real. Park the arms near where a demonstration starts first, then hold
#    COAG whenever the policy needs correcting.
python deploy/deploy_with_intervention.py \
    --checkpoint outputs/square_round0/checkpoints/latest.ckpt --task square \
    --execute --align-seconds 5 --inference-steps 16
#    -> data/intervention_runs/square/run_NNNN/ + deploy/logs/intervention_run.npz

#    audit what the handover actually did
python tools/check_handover.py deploy/logs/intervention_run.npz

# --- anywhere (standard library only) ---------------------------------------
# 6. see what the takeovers would yield, then extract them next to the demos
python dagger/merge_interventions.py data/intervention_runs/square --dry-run
python dagger/merge_interventions.py data/intervention_runs/square \
    --out data/dagger/square --include-demos data/drawing_square

# --- container shell --------------------------------------------------------
# 7. convert and retrain on demos + corrections, then go back to step 4 with
#    the new checkpoint. Interventions should get rarer; the session summary's
#    intervention count is the number to watch.
python data_processing/convert_drawing_6d_abs.py \
    data/dagger/square data/diffusion_policy/square_dagger.zarr --overwrite
DP_DATASET=data/diffusion_policy/square_dagger.zarr \
DP_OUTPUT=outputs/square_round1 \
bash train_drawing_policy.sh --train
```

Later rounds record into the same `data/intervention_runs/square/`, so step 6
sweeps up every run to date and each round trains on the demonstrations plus
every correction so far.

### The one required argument

`deploy_with_intervention.py` requires `--checkpoint`; everything else has a
default. Without `--execute` it is a dry run — the loop, the pedal handover, the
recording and the summary all run, and nothing is published. The full flag table
is in section 6.

### Tests

```bash
bash run_tests.sh            # both environments
bash run_tests.sh --ros      # just the ROS half
bash run_tests.sh --docker   # just the container half
```


## 3. Collect demonstrations

Draw the shape yourself, through the console, as many times as you have patience
for. Jaw pinches start and stop each episode so your hands never leave the
masters.

```bash
bash data_collection.sh --task circle       # -> data/drawing_circle/
bash data_collection.sh --task rectangle    # -> data/drawing_rectangle/
```

* pinch the **PSM1** jaw 3× within ~2 s → start recording
* pinch the **PSM2** jaw 3× within ~2 s → stop and save
* `Ctrl+C` → quit

Frames are written at 30 Hz. Each episode is a directory:

```text
episode_0000/
  colors/left_image_000000.jpg
  data.json                      # dual-arm state per frame
```

Both arms are stored as joint angles *and* end-effector pose. The pose path is
the one everything downstream uses:

```text
per arm:  3 position + 6D rotation + 1 gripper  = 10
both:     cutter (PSM2) then retraction (PSM1)  = 20
```

That arm order — **cutter first** — is fixed across the converter, the deploy
node and the recorder. Swapping it trains each arm on the other's trajectory,
with nothing anywhere to complain.

Images are saved exactly as they come off the camera.

Draw with one arm if you like; just leave the other still and let it be recorded
as a constant. The 20D representation is kept because it matches the sibling
project's checkpoints exactly.


## 4. Convert to a Diffusion Policy Zarr

```bash
python data_processing/convert_drawing_6d_abs.py \
    data/drawing_circle data/diffusion_policy/circle.zarr
```

Every 6th 30 Hz frame is sampled (5 Hz), images are resized to 160×120, and
`action` at frame `t` is the **absolute** state at `t + 6`, clamped at the
episode end — not a delta. The deploy node commands the predicted row directly;
if this were a delta, every motion would be doubled.

Pass `--overwrite` to replace an existing Zarr, which DAgger rounds will want.


## 5. Train

```bash
DP_DATASET=data/diffusion_policy/circle.zarr \
DP_OUTPUT=outputs/circle_round0 \
bash train_drawing_policy.sh --train
```

`--check` validates the dataset adapter and exits; `--smoke` runs two epochs of
three batches; `--train` is the real thing. Extra arguments pass through to
Hydra. State and action dimensions are read off the Zarr, so nothing needs
configuring per round.

`--train` writes into `DP_OUTPUT` itself, so the checkpoint the deploy step
wants is `$DP_OUTPUT/checkpoints/latest.ckpt`. `--smoke` writes to a timestamped
subdirectory instead, to keep throwaway runs from burying a real one.


## 6. Deploy, with the pedal

```bash
# dry run first: every code path except publishing
python deploy/deploy_with_intervention.py \
    --checkpoint outputs/circle_round0/checkpoints/latest.ckpt --task circle

# then for real
python deploy/deploy_with_intervention.py \
    --checkpoint outputs/circle_round0/checkpoints/latest.ckpt --task circle --execute
```

### What the pedal does

Control passes through an **alignment window in each direction**, during which
*nothing commands the arms* — they hold whatever pose they were left in.
`--align-seconds` sets its length; it defaults to **1 s**, and the examples here
pass `--align-seconds 5`:

```text
POLICY ──COAG down──► ALIGN_TO_EXPERT ──(align)──► SURGEON
SURGEON ──COAG up───► ALIGN_TO_POLICY ──(align)──► POLICY
```

Going to the surgeon, the window is for the MTM wrist to come into alignment
with the PSM tool before the master starts driving. Coming back, it is for the
arm to settle and the policy to re-condition on what you actually did, so its
first command is computed from reality rather than the scene it last saw.
`--align-seconds 0` switches straight over, as before.

Handover is by **strict exclusion**, not arbitration. Two publishers streaming
setpoints at one PSM is not a blend, it is a race, and the arm tracks whichever
message landed last. Exactly one predicate in the whole system permits motion —
`commands_allowed`, true in the `POLICY` phase alone — so "is the robot allowed
to move right now" has one answer and one place to check it.

| phase | this node | dVRK teleop |
|---|---|---|
| POLICY | publishes `servo_cp` + `jaw/servo_jp` | disengaged |
| ALIGN_TO_EXPERT | **nothing** | engaging |
| SURGEON | **nothing** | drives the arms |
| ALIGN_TO_POLICY | **nothing** | disengaged |

The pedal's current state always wins. Press again during a handback and the
machine goes back to `ALIGN_TO_EXPERT` and restarts the timer rather than
finishing its countdown — a surgeon reaching for the pedal a second time is not
asking to wait. Ping-ponging between the two windows is safe by construction,
because neither commands anything.

**What the window does not do:** it does not hold off dVRK's teleoperation.
dVRK engages teleop on its own schedule when COAG goes down, and this node has
no say in that. What the window guarantees is that *this* node stays silent
across the transition. Going the other way — pedal up — teleop has disengaged
and the window genuinely does own the arm.

This node never commands an MTM. It subscribes to `/MTML/measured_cp` and
`/MTMR/measured_cp` **read-only**, to report the wrist/tool angle during the
window (`--mtm-psm1` / `--mtm-psm2`, empty string to disable). Aligning a master
a surgeon already has their hand on is dVRK's job and its interlock's. If the
MTM topics are missing the window still holds for its full duration — its job is
to keep this node silent, which does not depend on measuring anything.

### Before you run it with `--execute`

* Your console must have a **teleop pair configured** (MTML-PSM1 / MTMR-PSM2 or
  whatever pairing you use) and the operator-present interlock satisfied. COAG
  alone moves nothing if the console does not think an operator is at the
  master. The node warns at startup if it sees no `MTM*_PSM*` topics, but it
  cannot check this properly — **verify the pedal drives the arms through the
  console first.**
* The node refuses to start with `--execute` if it has never seen a message on
  `/footpedals/coag`. A policy running with no takeover path is the one
  configuration this program exists to prevent. Tap the pedal once if it is
  silent; `--allow-no-pedal` overrides, for dry runs only.
* Position the arms near where a demonstration starts. The first engagement is a
  one-shot `move_cp`, and it aborts if the first target is more than
  `--max-initial-jump` (2 cm) away.

### Resuming after a takeover

The queued action chunk is dropped and the policy replans from what it can
currently see. It resumes into `servo_cp` streaming clamped in **both** position
and orientation, each measured against the arm's *measured* pose:

| channel | clamp |
|---|---|
| position | `--max-pos-step`, 5 mm per command |
| orientation | `--max-angle-step-deg`, 2° per command, a shortened geodesic on SO(3) |
| jaw | rate-limited by the controller's `jaw_vmax` |

The orientation clamp is not cosmetic. Without it the position was clamped and
the policy's quaternion went through raw, so on the first command after an
intervention the tool tip crept 5 mm while the wrist was free to snap to any
orientation the policy asked for.

The one-shot `move_cp` used for the first engagement is deliberately never
reused after an intervention: it is a blocking multi-second trajectory the arm
interpolates internally, which nothing here can clamp, and firing one into a
scene a human has just rearranged is the worst available option.

### Useful flags

| Flag | Default | |
|---|---|---|
| `--execute` | off | Actually publish. Without it, a full dry run. |
| `--rate` | 10 Hz | Control loop rate |
| `--max-pos-step` | 0.005 m | Per-command position clamp |
| `--align-seconds` | 1.0 s | Alignment window, both directions. Pass `5` for the window described above. |
| `--max-angle-step-deg` | 2.0° | Per-command orientation clamp |
| `--action-mode` | auto | `absolute` or `relative` |
| `--no-initial-move` | on | Skip the blocking opening `move_cp`, which otherwise blocks the loop for 4 s with the pedal unpolled. Use it when the arms are already parked at the start of a demonstration. |
| `--receding-horizon` | off | Execute only the first action of each chunk |
| `--inference-steps` | checkpoint's | Try 16–32 for live use |
| `--pedal-topic` | `/footpedals/coag` | Use `/footpedals/clutch` to rebind |
| `--no-record` | off | Do not record the session |

### What a session leaves behind

```text
data/intervention_runs/circle/run_0000/
  colors/left_image_NNNNNN.jpg
  data.json          # every frame at 30 Hz, tagged control_mode:
                     #   policy | align_to_expert | expert | align_to_policy
deploy/logs/intervention_run.npz   # per control cycle: mode, action, measured pose
```

and a summary on exit:

```text
================================================================
SESSION SUMMARY
================================================================
  control cycles   : 1043
  interventions    : 3
  under expert     : 210 cycles (20.1%)
    #01  t=   31.4s  for 6.2s
    #02  t=   88.7s  for 9.9s
    #03  t=  151.0s  for 4.8s
  frames recorded  : 2499 policy + 630 expert -> data/intervention_runs/circle/run_0000
================================================================
```

Sessions are recorded at 30 Hz on their own timer, not at the control rate. The
converter takes every 6th frame, so a run recorded at 10 Hz would come out at
1.7 Hz and the corrections would be sampled three times coarser than the
demonstrations they are meant to join.


## 7. Fold the corrections back in

```bash
# look first
python dagger/merge_interventions.py data/intervention_runs/circle --dry-run

# extract, alongside the original demos, into one directory
python dagger/merge_interventions.py data/intervention_runs/circle \
    --out data/dagger/circle --include-demos data/drawing_circle

# convert and retrain
python data_processing/convert_drawing_6d_abs.py \
    data/dagger/circle data/diffusion_policy/circle_dagger.zarr
DP_DATASET=data/diffusion_policy/circle_dagger.zarr \
DP_OUTPUT=outputs/circle_round1 \
bash train_drawing_policy.sh --train
```

Then deploy the new checkpoint and go round again. Interventions should get
rarer; the summary's intervention count is the number to watch.

### Which frames become training rows

The converter labels frame `t` with the state at `t + 6`, so the action attached
to a frame is *what happened next*. That is what decides the margins:

* **`--lead-in` (default 6)** — frames from just *before* the pedal went down.
  The policy was still driving then, but their action labels come from the six
  frames that follow, which are the expert's. These are the rows that actually
  teach the correction: this observation, which the policy handled badly, maps
  to the action the human chose instead. This is the point of HG-DAgger, so the
  default is exactly one action offset.
* **`--lead-out` (default 6)** — frames from just *after* the pedal came up.
  They exist so the segment's final rows get a full action horizon. The
  converter clamps `t + 6` at the episode end, which affects exactly **one row
  per episode**: its action is read from a frame less than 6 ahead, and when the
  episode length happens to satisfy `(length - 1) % 6 == 0` that frame is the
  row itself — an action identical to the current state, i.e. "stop here".
  Padding the segment pushes that degenerate row into policy-driven frames
  instead of landing it on the last thing the surgeon did. The cost is that
  those trailing rows are policy-driven observations with policy-driven labels.
  It is one row either way, so `--lead-out 0` is a perfectly reasonable choice
  if you would rather every row be genuinely expert-driven.
* **`--min-expert-frames` (default 15)** — half a second. Shorter segments are
  dropped as pedal slips rather than corrections.

**With the alignment windows enabled, both margins collapse to zero on their
own**, and that is intended. A margin stops at an alignment window, because
reaching back across five seconds of held-still arm would give the lead-in rows
an action label of "the held pose" — teaching the policy to freeze at exactly
the moment it was going wrong. The consequence is worth knowing: with a
5-second window the observation the policy mishandled and the surgeon's
corrective action are no longer six frames apart, so the HG-DAgger lead-in trick
no longer applies and extracted episodes are purely what the surgeon drove. Run
with `--align-seconds 0` if you want the original adjacency back.

Extracted episodes keep `source_run`, `source_idx` and `control_mode` per frame,
so the provenance of every training row survives into the dataset. Images are
hardlinked, not copied, so re-slicing a session with different margins is cheap.

`--include-demos` symlinks the demonstration episodes into the output directory
rather than copying them or appending corrections into the demo directory —
which would quietly make the pre-correction dataset unreproducible.


## 8. Files

```text
data_collection.sh                  record demonstrations
data_processing/
  convert_drawing_6d_abs.py         episodes -> Diffusion Policy Zarr
train_drawing_policy.sh             validate the Zarr and launch training
deploy/
  deploy_with_intervention.py       closed-loop policy + pedal handover
  handover.py                       the four-phase state machine
  pedal.py                          footpedal state, latched-QoS aware
  run_recorder.py                   session recording in the collector's format
  geometry.py                       rotations + the position and orientation clamps
  controller.py                     PSM publishers/subscribers  (vendored)
  deploy_lib.py                     checkpoint loading + inference  (vendored)
tools/
  check_handover.py                 audit a session log; non-zero exit on failure
dagger/
  merge_interventions.py            takeovers -> trainable episodes
src/
  diffusion/                        diffusion_policy  (vendored)
  arclab_dvrk/                      dVRK helpers + the demo collector  (vendored)
tests/                              see below
run_tests.sh                        runs the suite in both environments
```

Vendored code is a copy of the sibling project's, with two fixes: the
calibration asset path in `controller.py` resolved against this checkout instead
of a hard-coded container path, and its PSM1 fallback branch no longer leaving
`cam_T_psm1` undefined.


## 9. Tests

```bash
bash run_tests.sh            # both environments
bash run_tests.sh --ros      # just the ROS half
bash run_tests.sh --docker   # just the container half
```

121 tests. Neither environment can run all of them, so `tests/conftest.py`
leaves out what is unrunnable and prints what it left out — a run that quietly
covers half of what you expected says so.

| File | Covers | Where |
|---|---|---|
| `test_handover_machine.py` | the four-phase machine, on a fake clock | anywhere |
| `test_merge_interventions.py` | segment boundaries, alignment windows | anywhere |
| `test_geometry.py` | rotation conventions, both clamps | scipy |
| `test_pedal.py` | pedal state and edges, against a real ROS publisher | ROS |
| `test_run_recorder.py` | the recorded schema the converter reads | OpenCV |
| `test_pipeline_integration.py` | recorder → merge → **real converter** → Zarr | container |

`test_pedal.py` uses a real rclpy publisher rather than a mock because the
QoS behaviour is the thing being tested: dVRK publishes the pedal latched and
event-driven, so between presses the current state exists *only* as a retained
sample, and a volatile-only subscriber is never handed it. `PedalMonitor`
subscribes twice, with both profiles, and the test covers both halves.

**What is no longer covered:** the deploy node's own handover had an end-to-end
test that drove it with a scripted square policy, and both went when `--scripted`
did. `test_handover_machine.py` still covers the four-phase machine in isolation
— the phases, the timings, the pedal edges — but nothing now exercises the node
itself deciding whether to publish. That the node stays silent while the pedal is
held is checked on the robot, by `tools/check_handover.py` against a session log,
and nowhere else.


## 10. Not yet verified on hardware

Everything above has been exercised against synthetic data and a real ROS
publisher. What that cannot cover:

* **No run on a real dVRK.** The handover has never driven an actual console.
  It was previously rehearsed against a fake dVRK and a scripted square policy —
  two takeovers, the surgeon dragging the arm 60 mm, zero commands issued while
  held, every command back inside the 5 mm clamp — but that rig was removed along
  with `--scripted`, so there is no longer a robot-free way to exercise it. Dry-run
  first, then `--execute` on a phantom, and audit the log with
  `tools/check_handover.py` every time.
* **The teleop pair.** This node assumes your console drives the arms when COAG
  is held. If it does not, the pedal will stop the policy and nothing will take
  over — the arms will simply hold. Check this before trusting the takeover.
* **Training has not been run to convergence here.** A `--smoke` run does now
  complete on the GPU (RTX 4070 Ti SUPER, two epochs, validation loss falling),
  so the config, the dataset adapter and the optimiser all work end to end. What
  has not happened is a full `--train` on a real dataset, or any check that the
  resulting checkpoint is worth deploying.
* **Drawing quality.** Whether a diffusion policy trained on your demonstrations
  draws a recognisable circle is an empirical question this scaffolding does not
  answer. Expect the first round to need corrections; that is what section 8 is
  for.
