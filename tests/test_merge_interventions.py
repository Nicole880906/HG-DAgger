"""Segment extraction: which recorded frames become training rows.

Pure stdlib plus the synthetic run built by the fixture -- no ROS, no robot.
The margins these tests pin down are the difference between a correction that
teaches the policy what to do instead and one that teaches it to stop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dagger"))

from merge_interventions import (  # noqa: E402
    find_runs,
    link_demos,
    load_run,
    next_episode_index,
    segments_in_run,
    write_episode,
)

EXPERT = "expert"
POLICY = "policy"
ALIGN_TO_EXPERT = "align_to_expert"
ALIGN_TO_POLICY = "align_to_policy"


def make_run(tmp_path: Path, modes: list[str], name: str = "run_0000") -> Path:
    """A run directory with one tiny JPEG per frame and the given mode labels."""
    run = tmp_path / name
    (run / "colors").mkdir(parents=True)
    frames = []
    for idx, mode in enumerate(modes):
        image_name = f"left_image_{idx:06d}.jpg"
        # Content does not matter here; only that the file exists and is unique.
        (run / "colors" / image_name).write_bytes(b"\xff\xd8\xff" + bytes([idx % 256]))
        frames.append({
            "idx": idx,
            "colors": {"left_image": f"colors/{image_name}"},
            "states": {"marker": idx},
            "control_mode": mode,
            "t": idx / 30.0,
        })
    (run / "data.json").write_text(json.dumps({"data": frames}))
    return run


def modes(n_policy: int, n_expert: int, *rest: int) -> list[str]:
    out = [POLICY] * n_policy + [EXPERT] * n_expert
    for i, count in enumerate(rest):
        out += [EXPERT if i % 2 else POLICY] * count
    return out


class TestSegmentBoundaries:
    def test_one_intervention_is_found_with_its_margins(self, tmp_path):
        run = make_run(tmp_path, modes(50, 30, 50))
        [segment] = segments_in_run(run, load_run(run), lead_in=6, lead_out=6,
                                    min_expert_frames=15)
        assert (segment.expert_start, segment.expert_end) == (50, 79)
        assert (segment.start, segment.end) == (44, 85)
        assert segment.n_expert == 30
        assert segment.n_frames == 42

    def test_two_interventions_stay_separate(self, tmp_path):
        run = make_run(tmp_path, modes(20, 30, 40, 30, 20))
        found = segments_in_run(run, load_run(run), lead_in=6, lead_out=6,
                                min_expert_frames=15)
        assert [(s.expert_start, s.expert_end) for s in found] == [(20, 49), (90, 119)]

    def test_short_segments_are_dropped_as_pedal_slips(self, tmp_path):
        run = make_run(tmp_path, modes(20, 5, 20))
        assert segments_in_run(run, load_run(run), 6, 6, min_expert_frames=15) == []

    def test_min_expert_frames_is_inclusive(self, tmp_path):
        run = make_run(tmp_path, modes(20, 15, 20))
        assert len(segments_in_run(run, load_run(run), 6, 6, min_expert_frames=15)) == 1

    def test_lead_in_cannot_run_off_the_front_of_the_run(self, tmp_path):
        """The pedal can be down before the first frame is recorded."""
        run = make_run(tmp_path, modes(2, 30, 20))
        [segment] = segments_in_run(run, load_run(run), lead_in=6, lead_out=6,
                                    min_expert_frames=15)
        assert segment.start == 0

    def test_lead_out_cannot_run_off_the_end_of_the_run(self, tmp_path):
        """Ctrl+C during an intervention leaves it open at the last frame."""
        run = make_run(tmp_path, modes(20, 30))
        [segment] = segments_in_run(run, load_run(run), lead_in=6, lead_out=6,
                                    min_expert_frames=15)
        assert segment.end == 49
        assert segment.expert_end == 49

    def test_zero_margins_give_exactly_the_expert_frames(self, tmp_path):
        run = make_run(tmp_path, modes(20, 30, 20))
        [segment] = segments_in_run(run, load_run(run), lead_in=0, lead_out=0,
                                    min_expert_frames=15)
        assert (segment.start, segment.end) == (segment.expert_start, segment.expert_end)

    def test_a_run_with_no_takeover_yields_nothing(self, tmp_path):
        run = make_run(tmp_path, [POLICY] * 100)
        assert segments_in_run(run, load_run(run), 6, 6, 15) == []

    def test_unlabelled_frames_count_as_policy(self, tmp_path):
        """Old or hand-made runs without the label must not be mistaken for expert data."""
        run = make_run(tmp_path, [POLICY] * 40)
        frames = load_run(run)
        for frame in frames:
            del frame["control_mode"]
        assert segments_in_run(run, frames, 6, 6, 1) == []


class TestAlignmentWindows:
    """Frames recorded while the arm was held for a handover.

    Nobody is driving during those five seconds, so they must never be taken
    for expert data -- and, less obviously, the lead-in must not reach back
    across one. A lead-in frame earns its place because its action label, read
    six frames later, is the surgeon's correction. Reach back over a window in
    which the arm was held still and the label is the held pose, which teaches
    the policy to freeze at exactly the moment it was going wrong.
    """

    def with_windows(self, tmp_path, align=150):
        modes = ([POLICY] * 60 + [ALIGN_TO_EXPERT] * align
                 + [EXPERT] * 60 + [ALIGN_TO_POLICY] * align + [POLICY] * 60)
        return make_run(tmp_path, modes), modes

    def test_align_frames_are_not_expert_data(self, tmp_path):
        run, _ = self.with_windows(tmp_path)
        frames = load_run(run)
        [segment] = segments_in_run(run, frames, 6, 6, 15)
        for record in frames[segment.start:segment.end + 1]:
            assert not record["control_mode"].startswith("align")

    def test_lead_in_stops_at_the_window_instead_of_crossing_it(self, tmp_path):
        run, _ = self.with_windows(tmp_path)
        [segment] = segments_in_run(run, load_run(run), lead_in=6, lead_out=6,
                                    min_expert_frames=15)
        assert segment.start == segment.expert_start
        assert segment.end == segment.expert_end

    def test_the_extracted_episode_is_purely_surgeon_driven(self, tmp_path):
        run, _ = self.with_windows(tmp_path)
        frames = load_run(run)
        [segment] = segments_in_run(run, frames, 6, 6, 15)
        out = tmp_path / "dataset"
        out.mkdir()
        episode = write_episode(segment, frames, out, 0)
        records = json.loads((episode / "data.json").read_text())["data"]
        assert {r["control_mode"] for r in records} == {EXPERT}

    def test_a_one_frame_window_still_blocks_the_lead_in(self, tmp_path):
        """The rule is the boundary, not its width."""
        run = make_run(tmp_path, [POLICY] * 40 + [ALIGN_TO_EXPERT] + [EXPERT] * 40)
        [segment] = segments_in_run(run, load_run(run), 6, 6, 15)
        assert segment.start == segment.expert_start == 41

    def test_without_windows_the_lead_in_still_reaches_back(self, tmp_path):
        """Runs recorded before the alignment windows existed, and
        --align-seconds 0, keep the original HG-DAgger behaviour."""
        run = make_run(tmp_path, modes(50, 30, 50))
        [segment] = segments_in_run(run, load_run(run), 6, 6, 15)
        assert segment.start == segment.expert_start - 6
        assert segment.end == segment.expert_end + 6

    def test_two_takeovers_separated_by_windows_stay_separate(self, tmp_path):
        run = make_run(tmp_path, (
            [POLICY] * 20 + [ALIGN_TO_EXPERT] * 30 + [EXPERT] * 30
            + [ALIGN_TO_POLICY] * 30 + [POLICY] * 20 + [ALIGN_TO_EXPERT] * 30
            + [EXPERT] * 30 + [ALIGN_TO_POLICY] * 30))
        found = segments_in_run(run, load_run(run), 6, 6, 15)
        assert len(found) == 2
        assert all(s.start == s.expert_start and s.end == s.expert_end for s in found)


class TestWriteEpisode:
    def test_episode_is_reindexed_and_self_contained(self, tmp_path):
        run = make_run(tmp_path, modes(50, 30, 50))
        frames = load_run(run)
        [segment] = segments_in_run(run, frames, 6, 6, 15)
        out = tmp_path / "dataset"
        out.mkdir()
        episode = write_episode(segment, frames, out, 0)

        payload = json.loads((episode / "data.json").read_text())
        records = payload["data"]
        assert len(records) == segment.n_frames
        assert [r["idx"] for r in records] == list(range(len(records)))
        for record in records:
            # Paths must be relative to the episode, since that is what the
            # converter joins them onto.
            assert record["colors"]["left_image"].startswith("colors/")
            assert (episode / record["colors"]["left_image"]).is_file()

    def test_states_and_provenance_survive_the_copy(self, tmp_path):
        run = make_run(tmp_path, modes(50, 30, 50))
        frames = load_run(run)
        [segment] = segments_in_run(run, frames, 6, 6, 15)
        out = tmp_path / "dataset"
        out.mkdir()
        episode = write_episode(segment, frames, out, 0)
        records = json.loads((episode / "data.json").read_text())["data"]

        assert records[0]["source_idx"] == segment.start
        assert records[0]["source_run"] == run.name
        assert records[0]["states"] == frames[segment.start]["states"]
        # The lead-in rows stay honestly labelled as policy-driven frames.
        assert records[0]["control_mode"] == POLICY
        assert records[6]["control_mode"] == EXPERT

    def test_images_are_shared_with_the_run_not_duplicated(self, tmp_path):
        """Hardlinks: re-slicing a long session must not re-copy every JPEG."""
        run = make_run(tmp_path, modes(50, 30, 50))
        frames = load_run(run)
        [segment] = segments_in_run(run, frames, 6, 6, 15)
        out = tmp_path / "dataset"
        out.mkdir()
        episode = write_episode(segment, frames, out, 0)
        source = run / frames[segment.start]["colors"]["left_image"]
        copied = episode / "colors" / "left_image_000000.jpg"
        assert copied.stat().st_ino == source.stat().st_ino
        assert copied.read_bytes() == source.read_bytes()

    def test_rewriting_the_same_episode_index_replaces_the_images(self, tmp_path):
        """Re-running the tool must not fail on the existing hardlinks."""
        run = make_run(tmp_path, modes(50, 30, 50))
        frames = load_run(run)
        [segment] = segments_in_run(run, frames, 6, 6, 15)
        out = tmp_path / "dataset"
        out.mkdir()
        write_episode(segment, frames, out, 0)
        write_episode(segment, frames, out, 0)  # must not raise


class TestDatasetAssembly:
    def test_numbering_continues_over_existing_episodes(self, tmp_path):
        out = tmp_path / "dataset"
        (out / "episode_0000").mkdir(parents=True)
        (out / "episode_0003").mkdir()
        assert next_episode_index(out) == 4

    def test_empty_directory_starts_at_zero(self, tmp_path):
        out = tmp_path / "dataset"
        out.mkdir()
        assert next_episode_index(out) == 0

    def test_linked_demos_are_named_so_the_converter_finds_them(self, tmp_path):
        """The converter globs episode_*; any other name is silently skipped."""
        demos = tmp_path / "demos"
        for i in range(3):
            (demos / f"episode_{i:04d}" / "colors").mkdir(parents=True)
        out = tmp_path / "dataset"
        out.mkdir()

        linked, next_index = link_demos(demos, out, 0)
        assert linked == 3
        assert next_index == 3
        found = sorted(p.name for p in out.glob("episode_*") if p.is_dir())
        assert found == ["episode_0000", "episode_0001", "episode_0002"]
        assert (out / "episode_0002").is_symlink()

    def test_linked_demos_do_not_collide_with_extracted_episodes(self, tmp_path):
        demos = tmp_path / "demos"
        (demos / "episode_0000" / "colors").mkdir(parents=True)
        out = tmp_path / "dataset"
        (out / "episode_0000").mkdir(parents=True)

        linked, next_index = link_demos(demos, out, next_episode_index(out))
        assert linked == 1
        assert next_index == 2
        assert (out / "episode_0001").is_symlink()
        assert not (out / "episode_0000").is_symlink()

    def test_demo_dir_without_episodes_is_an_error(self, tmp_path):
        empty = tmp_path / "demos"
        empty.mkdir()
        out = tmp_path / "dataset"
        out.mkdir()
        with pytest.raises(FileNotFoundError):
            link_demos(empty, out, 0)


class TestFindRuns:
    def test_a_run_directory_is_accepted_directly(self, tmp_path):
        run = make_run(tmp_path, modes(10, 20, 10))
        assert find_runs([run]) == [run]

    def test_a_parent_directory_expands_to_its_runs(self, tmp_path):
        parent = tmp_path / "runs"
        parent.mkdir()
        first = make_run(parent, modes(10, 20), name="run_0000")
        second = make_run(parent, modes(10, 20), name="run_0001")
        assert find_runs([parent]) == [first, second]

    def test_a_directory_that_is_neither_is_an_error(self, tmp_path):
        stray = tmp_path / "nothing"
        stray.mkdir()
        with pytest.raises(FileNotFoundError):
            find_runs([stray])
