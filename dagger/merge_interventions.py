#!/usr/bin/env python3
"""Slice expert takeovers out of deploy runs into a trainable episode dataset.

``deploy/deploy_with_intervention.py`` records a whole session -- policy frames
and expert frames alike -- into one run directory, each frame tagged with who
was driving.  This turns the human-driven stretches of one or more runs into
``episode_NNNN/`` directories in the layout
``data_processing/convert_drawing_6d_abs.py`` already reads, so a round of
corrections goes back into training through the same converter and the same
training script as the original demonstrations.

Which frames end up in an episode
---------------------------------
The converter labels frame ``t`` with the state at ``t + 6`` (6 raw frames at
30 Hz = the 5 Hz action step).  The action attached to a frame is therefore
*what happened next*, which is what decides where each segment's edges belong:

``--lead-in`` (default 6)
    Frames from just **before** the pedal went down.  The robot was still under
    the policy in those frames, but their action labels are read from the
    six frames that follow -- which are the expert's.  These rows are the ones
    that actually teach the correction: this observation, which the policy
    responded to badly, maps to the action the human chose instead.  This is
    the point of HG-DAgger, so the default is exactly one action offset.

``--lead-out`` (default 6)
    Frames from just **after** the pedal came up.  These exist only so the last
    genuine expert frames get real action labels instead of the converter's
    end-of-episode clamp, which repeats the final state and would teach the
    policy to stop mid-stroke.  The cost is that those trailing rows are
    themselves policy-driven observations with policy-driven labels.  Pass
    ``--lead-out 0`` for strictly expert-only frames and accept the clamp.

``--min-expert-frames`` (default 15, i.e. half a second)
    Segments shorter than this are dropped as pedal slips rather than
    corrections.

Usage
-----
::

    # look first: what would be extracted, and from where
    python dagger/merge_interventions.py data/intervention_runs/circle --dry-run

    # extract, alongside the original demos, into one directory to convert
    python dagger/merge_interventions.py data/intervention_runs/circle \\
        --out data/dagger/circle --include-demos data/drawing_circle

    # then the usual two steps
    python data_processing/convert_drawing_6d_abs.py \\
        data/dagger/circle data/diffusion_policy/circle_dagger.zarr
    SURGFLOW_DP_DATASET=data/diffusion_policy/circle_dagger.zarr \\
        bash train_drawing_policy.sh --train
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

EXPERT = "expert"
POLICY = "policy"
# Both handover alignment windows. During these the arm is held and nobody is
# driving, so the frames are training-worthless in either direction -- but they
# are recorded, because they are exactly what you want to look at when auditing
# a handover.
ALIGN_PREFIX = "align"


@dataclass(frozen=True)
class Segment:
    run: Path
    start: int          # first frame index kept, including lead-in
    expert_start: int   # first genuinely expert frame
    expert_end: int     # last genuinely expert frame (inclusive)
    end: int            # last frame index kept, including lead-out

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1

    @property
    def n_expert(self) -> int:
        return self.expert_end - self.expert_start + 1


def load_run(run_dir: Path) -> list[dict]:
    json_path = run_dir / "data.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"no data.json in {run_dir}")
    with json_path.open() as handle:
        payload = json.load(handle)
    frames = payload.get("data")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{json_path} has no non-empty 'data' list")
    return frames


def find_runs(paths: list[Path]) -> list[Path]:
    """Accept run directories, or parents holding ``run_*`` subdirectories."""
    runs: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if (path / "data.json").is_file():
            runs.append(path)
            continue
        children = sorted(p for p in path.glob("run_*") if (p / "data.json").is_file())
        if not children:
            raise FileNotFoundError(
                f"{path} is neither a run directory (no data.json) nor a parent of any"
            )
        runs.extend(children)
    return runs


def segments_in_run(
    run: Path,
    frames: list[dict],
    lead_in: int,
    lead_out: int,
    min_expert_frames: int,
) -> list[Segment]:
    """Contiguous expert stretches, widened by the lead-in / lead-out margins.

    The margins stop at an alignment window. That is not a detail -- it is what
    keeps the lead-in honest. A lead-in frame earns its place because its action
    label, read six frames later, is the surgeon's correction; reach back across
    a five-second window in which the arm was held still and the label is the
    held pose, which teaches the policy to freeze at precisely the moment it was
    going wrong. With alignment windows enabled the margins therefore collapse
    to zero on their own, and the extracted episodes are purely what the surgeon
    drove. See the README for what that costs.
    """
    modes = [frame.get("control_mode", POLICY) for frame in frames]
    out: list[Segment] = []
    i = 0
    n = len(frames)
    while i < n:
        if modes[i] != EXPERT:
            i += 1
            continue
        expert_start = i
        while i + 1 < n and modes[i + 1] == EXPERT:
            i += 1
        expert_end = i
        i += 1
        if expert_end - expert_start + 1 < min_expert_frames:
            continue

        start = expert_start
        for _ in range(lead_in):
            prev = start - 1
            if prev < 0 or modes[prev].startswith(ALIGN_PREFIX):
                break
            start = prev
        end = expert_end
        for _ in range(lead_out):
            nxt = end + 1
            if nxt >= n or modes[nxt].startswith(ALIGN_PREFIX):
                break
            end = nxt

        out.append(Segment(run=run, start=start, expert_start=expert_start,
                           expert_end=expert_end, end=end))
    return out


def next_episode_index(out_dir: Path) -> int:
    """Continue the numbering already in ``out_dir`` rather than clobbering it."""
    used = []
    for path in out_dir.glob("episode_*"):
        suffix = path.name[len("episode_"):]
        if path.is_dir() and suffix.isdigit():
            used.append(int(suffix))
    return max(used) + 1 if used else 0


def place_image(src: Path, dst: Path) -> None:
    """Hardlink the image if possible, otherwise copy.

    A session is thousands of JPEGs and re-slicing with different margins is
    expected, so copying every extraction wastes real disk.  Hardlinks cost
    nothing on the same filesystem; the fallback covers the cross-device case.
    The images are never written again after the run, so sharing the inode is
    safe.
    """
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_episode(segment: Segment, frames: list[dict], out_dir: Path, index: int) -> Path:
    """Materialise one segment as ``episode_NNNN/`` with re-indexed frames."""
    episode_dir = out_dir / f"episode_{index:04d}"
    color_dir = episode_dir / "colors"
    color_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for new_idx, old_idx in enumerate(range(segment.start, segment.end + 1)):
        frame = frames[old_idx]
        relpath = frame["colors"]["left_image"]
        src = segment.run / relpath
        if not src.is_file():
            raise FileNotFoundError(f"missing image for {segment.run.name} frame {old_idx}: {src}")
        name = f"left_image_{new_idx:06d}.jpg"
        dst = color_dir / name
        if dst.exists():
            dst.unlink()
        place_image(src, dst)
        records.append({
            "idx": new_idx,
            "colors": {"left_image": f"colors/{name}"},
            "states": frame["states"],
            # Kept so the provenance of every row survives into the dataset:
            # which run, which original frame, and whether the human or the
            # policy was driving when it was recorded.
            "control_mode": frame.get("control_mode", POLICY),
            "source_run": segment.run.name,
            "source_idx": int(old_idx),
        })

    payload = {
        "info": {"image": {"fps": 30}},
        "text": {"goal": f"expert correction from {segment.run.name}"},
        "segment": {
            "run": str(segment.run),
            "frames": [segment.start, segment.end],
            "expert_frames": [segment.expert_start, segment.expert_end],
            "n_frames": segment.n_frames,
            "n_expert": segment.n_expert,
        },
        "data": records,
    }
    with (episode_dir / "data.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return episode_dir


def link_demos(demo_dir: Path, out_dir: Path, start_index: int) -> tuple[int, int]:
    """Symlink demonstration episodes into ``out_dir``; returns (linked, next index).

    The converter takes a single input directory, so training on demos plus
    corrections means having both under one roof.  Symlinks keep the original
    demonstration set untouched -- appending corrections into the demo directory
    itself would quietly make the pre-correction dataset unreproducible.

    The links must be named ``episode_NNNN`` and numbered into the same sequence
    as the extracted segments: the converter globs ``episode_*``, so a link
    under any other name is silently skipped and you train on the corrections
    alone, wondering why the policy forgot the task.
    """
    demo_dir = demo_dir.expanduser().resolve()
    episodes = sorted(p for p in demo_dir.glob("episode_*") if p.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episode_* directories under {demo_dir}")
    index = start_index
    linked = 0
    for episode in episodes:
        target = out_dir / f"episode_{index:04d}"
        while target.is_symlink() or target.exists():
            index += 1
            target = out_dir / f"episode_{index:04d}"
        target.symlink_to(episode)
        linked += 1
        index += 1
    return linked, index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path,
                   help="Run directories, or parents containing run_* directories")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dataset directory (required unless --dry-run)")
    p.add_argument("--lead-in", type=int, default=6,
                   help="Frames kept before each takeover; one action offset by default, "
                        "because those rows carry the correction's action labels")
    p.add_argument("--lead-out", type=int, default=6,
                   help="Frames kept after each handback, so the final expert frames get "
                        "real action labels instead of the end-of-episode clamp. "
                        "Use 0 for strictly expert-driven frames.")
    p.add_argument("--min-expert-frames", type=int, default=15,
                   help="Drop segments shorter than this (30 Hz frames); filters pedal slips")
    p.add_argument("--include-demos", type=Path, default=None,
                   help="Also symlink this demonstration directory's episodes into --out, "
                        "so one convert covers demos + corrections")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be extracted and write nothing")
    args = p.parse_args(argv)
    if args.out is None and not args.dry_run:
        p.error("--out is required unless --dry-run is given")
    for name in ("lead_in", "lead_out", "min_expert_frames"):
        if getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = find_runs(list(args.runs))

    plan: list[tuple[Segment, list[dict]]] = []
    for run in runs:
        frames = load_run(run)
        found = segments_in_run(
            run, frames, args.lead_in, args.lead_out, args.min_expert_frames)
        n_expert_total = sum(1 for f in frames if f.get("control_mode") == EXPERT)
        print(f"{run.name}: {len(frames)} frames, {n_expert_total} expert, "
              f"{len(found)} segment(s) kept")
        for segment in found:
            print(f"    frames {segment.start:6d}-{segment.end:<6d} "
                  f"({segment.n_frames:4d} kept, {segment.n_expert:4d} expert)")
        plan.extend((segment, frames) for segment in found)

    if not plan:
        print("\nNo intervention segments met the criteria. Nothing to do.")
        print("If you expected some, check --min-expert-frames against how long "
              "the pedal was actually held.")
        return 1

    total_frames = sum(segment.n_frames for segment, _ in plan)
    total_expert = sum(segment.n_expert for segment, _ in plan)
    print(f"\n{len(plan)} segment(s), {total_frames} frames "
          f"({total_expert} expert), from {len(runs)} run(s)")

    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index = next_episode_index(out_dir)
    if index:
        print(f"{out_dir} already holds episodes; continuing numbering at {index:04d}")

    # Demos first, so they keep the low episode numbers and each re-run of this
    # tool appends corrections after them instead of renumbering the set.
    if args.include_demos is not None:
        linked, index = link_demos(args.include_demos, out_dir, index)
        print(f"  symlinked {linked} demonstration episode(s) from {args.include_demos}")

    for segment, frames in plan:
        episode_dir = write_episode(segment, frames, out_dir, index)
        print(f"  wrote {episode_dir.name}  <- {segment.run.name} "
              f"[{segment.start}:{segment.end}]")
        index += 1

    print(f"\ndataset: {out_dir}")
    print("next:")
    print(f"  python data_processing/convert_drawing_6d_abs.py {out_dir} "
          f"data/diffusion_policy/<name>.zarr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
