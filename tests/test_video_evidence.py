from __future__ import annotations

import copy
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.research_vcs import research_payload_issues
from ai_scientist.utils.semantic_memory import estimate_text_tokens
from xscientist.research_cli import main
from xscientist.research_context import (
    build_research_context_snapshot,
    render_research_context_for_prompt,
)
from xscientist.research_git import ResearchGitError
from xscientist.research_vcs import ResearchRepository
from xscientist.video_evidence import (
    VIDEO_ASSET_PROTOCOL,
    VIDEO_EVIDENCE_PROTOCOL,
    _blob,
    build_video_evidence_pack,
    list_video_assets,
    prepare_video_evidence_prompt,
    record_video_evidence,
    register_video_asset,
)


class VideoContractTests(unittest.TestCase):
    def test_incomplete_video_payloads_cannot_enter_the_native_graph(self):
        for protocol, kind in (
            (VIDEO_ASSET_PROTOCOL, "source_snapshot"),
            (VIDEO_EVIDENCE_PROTOCOL, "evidence"),
        ):
            self.assertTrue(
                research_payload_issues(
                    kind, {"protocol_kind": protocol, "result": "video"}
                )
            )

    def test_invalid_budgets_are_rejected_before_repository_access(self):
        for limit in (0, -1, 17, True, 1.5):
            with self.assertRaises(ResearchGitError):
                build_video_evidence_pack("missing", ["rso-missing"], max_frames=limit)
        for budget in (0, True, 511, 16001):
            with self.assertRaises(ResearchGitError):
                build_video_evidence_pack(
                    "missing", ["rso-missing"], budget_tokens=budget
                )


@unittest.skipUnless(
    all(shutil.which(tool) for tool in ("git", "ffmpeg", "ffprobe")),
    "Git and FFmpeg are required",
)
class VideoEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_directory = tempfile.TemporaryDirectory()
        cls.video = Path(cls.media_directory.name) / "private-source-name.mkv"
        subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=4:duration=2",
                "-vf",
                "select='not(eq(n,1))'",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "ffv1",
                str(cls.video),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
        )

    @classmethod
    def tearDownClass(cls):
        cls.media_directory.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "study"
        self.repository = ResearchRepository.init(
            self.root,
            question="Video evidence",
            git_user_name="Video Test",
            git_user_email="video@example.invalid",
        )

    def _asset(self):
        return register_video_asset(self.root, self.video)["object"].object_id

    def _evidence(self, **kwargs):
        return record_video_evidence(
            self.root,
            self._asset(),
            start_seconds=0.25,
            end_seconds=1.75,
            result="The object moves across the scene",
            max_frames=3,
            **kwargs,
        )["object"].object_id

    def test_registration_deduplicates_and_keeps_original_bytes_out_of_git(self):
        registered = register_video_asset(self.root, self.video)
        asset = self.repository.get(registered["object"].object_id)
        self.assertEqual(asset["payload"]["media"]["width"], 64)
        self.assertNotIn(str(self.video), json.dumps(asset))
        self.assertNotIn(self.video.name, json.dumps(asset))
        self.assertEqual(
            _blob(self.repository, asset["payload"]["content_hash"]).read_bytes(),
            self.video.read_bytes(),
        )
        head = self.repository.status()["head"]
        repeated = register_video_asset(self.root, self.video)
        self.assertFalse(repeated["object"].created)
        self.assertEqual(self.repository.status()["head"], head)
        self.assertEqual(len(list_video_assets(self.root)), 1)
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertNotIn(".ara-store/objects", tracked)
        self.assertTrue(self.repository.fsck()["ok"])

    def test_variable_frame_timestamps_and_visual_prompt_are_source_bound(self):
        evidence = self._evidence()
        payload = self.repository.get(evidence)["payload"]
        # Frame at .25 was removed: timestamps must come from decoding, not requested times or average FPS.
        times = [row["timestamp_seconds"] for row in payload["frames"]]
        self.assertEqual(times, [0.5, 0.75, 1.25])
        self.assertEqual(payload["verification_status"], "unverified_observation")
        pack = build_video_evidence_pack(self.root, [evidence], max_frames=2)
        self.assertEqual(pack["frame_count"], 2)
        self.assertEqual(pack["items"][0]["omitted_frames"], 1)
        content = prepare_video_evidence_prompt(
            self.root, [evidence], message="Compare motion", max_frames=2
        )
        self.assertEqual(len([row for row in content if row["type"] == "image_url"]), 2)
        labels = " ".join(row["text"] for row in content if row["type"] == "text")
        self.assertIn("decoded timestamp 0.5", labels)
        self.assertIn(payload["source_content_hash"], labels)
        self.assertIn("audio and unsampled intervals have not been inspected", labels)
        self.assertTrue(self.repository.fsck()["ok"])

    def test_invalid_interval_or_source_does_not_create_evidence(self):
        asset = self._asset()
        head = self.repository.status()["head"]
        for start, end in (
            (-1, 1),
            (1, 1),
            (1, 0),
            (0, 3),
            (float("nan"), 1),
            (0, float("inf")),
            (True, 1),
        ):
            with self.assertRaises(ResearchGitError):
                record_video_evidence(
                    self.root,
                    asset,
                    start_seconds=start,
                    end_seconds=end,
                    result="Motion",
                )
        self.assertEqual(self.repository.status()["head"], head)
        self.assertFalse(self.repository.objects(kind="evidence"))

    def test_corrupt_or_missing_cas_payload_cannot_be_used_as_visual_evidence(self):
        evidence = self._evidence()
        pack = build_video_evidence_pack(self.root, [evidence])
        path = _blob(self.repository, pack["items"][0]["frames"][0]["artifact_hash"])
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        with self.assertRaisesRegex(ResearchGitError, "hash mismatch"):
            prepare_video_evidence_prompt(self.root, [evidence], message="Inspect")

    def test_failed_record_rolls_back_typed_record_and_new_pointers(self):
        head = self.repository.status()["head"]
        with mock.patch(
            "xscientist.video_evidence._finish",
            side_effect=ResearchGitError("checkpoint failed"),
        ):
            with self.assertRaisesRegex(ResearchGitError, "checkpoint failed"):
                register_video_asset(self.root, self.video)
        self.assertEqual(self.repository.status()["head"], head)
        self.assertFalse(list_video_assets(self.root))
        self.assertFalse(list((self.root / "research-objects").glob("*.json")))

    def test_missing_media_tool_and_source_links_are_rejected(self):
        with mock.patch("xscientist.video_evidence.shutil.which", return_value=None):
            with self.assertRaisesRegex(ResearchGitError, "ffprobe is required"):
                register_video_asset(self.root, self.video)
        link = Path(self.temporary.name) / "linked.mkv"
        link.symlink_to(self.video)
        with self.assertRaisesRegex(ResearchGitError, "symbolic link"):
            register_video_asset(self.root, link)
        self.assertFalse(list_video_assets(self.root))

    def test_temporal_schema_rejects_outside_frames_and_changed_commitments(self):
        payload = self.repository.get(self._evidence())["payload"]
        for timestamp in (0, 2, float("nan")):
            changed = copy.deepcopy(payload)
            changed["frames"][0]["timestamp_seconds"] = timestamp
            if timestamp == timestamp:
                changed["evidence_hash"] = canonical_content_hash(
                    {k: v for k, v in changed.items() if k != "evidence_hash"}
                )
            self.assertTrue(research_payload_issues("evidence", changed))
        changed = copy.deepcopy(payload)
        changed["result"] = "Changed observation"
        self.assertTrue(research_payload_issues("evidence", changed))

    def test_native_context_retains_video_locator_and_frame_budget_is_global(self):
        hypothesis = self.repository.record(
            "hypothesis",
            {"statement": "Object motion", "falsifier": "No object motion"},
        )
        self.repository.commit(stage="planning", subject="Motion hypothesis")
        evidence = self._evidence(supports=[hypothesis.object_id])
        context = build_research_context_snapshot(
            self.repository,
            target_ids=[hypothesis.object_id],
            decision_kind="inspect_motion",
            selected="inspect",
            options_considered=[
                "inspect",
                {
                    "option": "infer_from_single_frame",
                    "rejected_because": "motion requires temporal evidence",
                },
            ],
            evidence_objectives=["object motion scene"],
            budget_tokens=2000,
        )
        prompt = render_research_context_for_prompt(context)
        self.assertIn("video:sha256:", prompt)
        self.assertIn("#t=0.25,1.75", prompt)
        pack = build_video_evidence_pack(
            self.root, [evidence, evidence], max_frames=1, budget_tokens=512
        )
        self.assertLessEqual(pack["frame_count"], 1)
        self.assertLessEqual(estimate_text_tokens(pack), 512)

    def test_cli_register_list_observe_and_context(self):
        def run(arguments):
            with redirect_stdout(io.StringIO()) as output:
                code = main(["video", *arguments, "--repo", str(self.root), "--json"])
            self.assertEqual(code, 0, output.getvalue())
            return json.loads(output.getvalue())

        asset = run(["add", str(self.video)])["object"]["object_id"]
        self.assertEqual(len(run(["list"])["assets"]), 1)
        evidence = run(
            [
                "observe",
                "--asset",
                asset,
                "--start",
                "0",
                "--end",
                "1",
                "--result",
                "Moving object",
                "--max-frames",
                "2",
            ]
        )["object"]["object_id"]
        pack = run(["context", "--evidence", evidence, "--max-frames", "1"])
        self.assertEqual(pack["frame_count"], 1)

    def test_source_is_verified_once_per_multimodal_context(self):
        evidence = self._evidence()
        asset_id = self.repository.get(evidence)["payload"]["source_id"]
        other = record_video_evidence(
            self.root,
            asset_id,
            start_seconds=0,
            end_seconds=1,
            result="A second observation",
            max_frames=2,
        )["object"].object_id
        source_hash = self.repository.get(evidence)["payload"]["source_content_hash"]
        from xscientist.video_evidence import _verify_pointer_object

        with mock.patch(
            "xscientist.video_evidence._verify_pointer_object",
            wraps=_verify_pointer_object,
        ) as verify:
            prepare_video_evidence_prompt(
                self.root, [evidence, other], message="Inspect both", max_frames=4
            )
        self.assertEqual(
            sum(call.args[1] == source_hash for call in verify.call_args_list), 1
        )

    def test_nonzero_start_is_normalized_without_inventing_frame_times(self):
        delayed = Path(self.temporary.name) / "delayed.mkv"
        subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(self.video),
                "-vf",
                "setpts=PTS+5/TB",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "ffv1",
                str(delayed),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        asset = register_video_asset(self.root, delayed)["object"].object_id
        duration = self.repository.get(asset)["payload"]["media"]["duration_seconds"]
        # Matroska remuxing may omit the final packet's duration.
        self.assertGreaterEqual(duration, 1.75)
        self.assertLessEqual(duration, 2)
        observation = record_video_evidence(
            self.root,
            asset,
            start_seconds=0,
            end_seconds=1,
            result="Normalized motion",
            max_frames=2,
        )["object"].object_id
        self.assertEqual(
            [
                row["timestamp_seconds"]
                for row in self.repository.get(observation)["payload"]["frames"]
            ],
            [0, 0.5],
        )

    def test_new_observations_reuse_verified_frames_without_media_tools(self):
        evidence = self._evidence()
        asset = self.repository.get(evidence)["payload"]["source_id"]
        arguments = dict(
            start_seconds=0.25,
            end_seconds=1.75,
            max_frames=3,
            result="An alternative interpretation of the same frames",
        )
        with mock.patch(
            "xscientist.video_evidence._run_media_tool",
            side_effect=AssertionError("cache reuse must not run FFmpeg"),
        ):
            saved = record_video_evidence(self.root, asset, **arguments)
            repeated = record_video_evidence(self.root, asset, **arguments)
        self.assertTrue(saved["operation"]["cache_hit"])
        self.assertEqual(saved["operation"]["decoded_frames"], 0)
        self.assertEqual(saved["operation"]["reused_frames"], 3)
        self.assertFalse(repeated["object"].created)
        self.assertEqual(
            self.repository.get(saved["object"].object_id)["payload"]["frames"],
            self.repository.get(evidence)["payload"]["frames"],
        )
        self.assertTrue(self.repository.fsck()["ok"])

    def test_sampling_changes_and_refresh_require_decoding(self):
        evidence = self._evidence()
        asset = self.repository.get(evidence)["payload"]["source_id"]
        from xscientist.video_evidence import _extract_frames

        with mock.patch(
            "xscientist.video_evidence._extract_frames", wraps=_extract_frames
        ) as extract:
            for count, reuse in ((2, True), (3, False)):
                saved = record_video_evidence(
                    self.root,
                    asset,
                    start_seconds=0.25,
                    end_seconds=1.75,
                    max_frames=count,
                    result="Motion",
                    reuse_frames=reuse,
                )
                self.assertFalse(saved["operation"]["cache_hit"])
            self.assertEqual(extract.call_count, 2)

    def test_context_shares_frames_and_covers_multiple_intervals(self):
        evidence = self._evidence()
        asset = self.repository.get(evidence)["payload"]["source_id"]
        duplicate = record_video_evidence(
            self.root,
            asset,
            start_seconds=0.25,
            end_seconds=1.75,
            max_frames=3,
            result="A conflicting interpretation",
        )["object"].object_id
        pack = build_video_evidence_pack(self.root, [evidence, duplicate], max_frames=3)
        self.assertEqual(pack["frame_count"], 3)
        self.assertEqual([len(row["frames"]) for row in pack["items"]], [3, 3])
        content = prepare_video_evidence_prompt(
            self.root, [evidence, duplicate], message="Compare", max_frames=3
        )
        self.assertEqual(sum(row["type"] == "image_url" for row in content), 3)
        early = record_video_evidence(
            self.root,
            asset,
            start_seconds=0,
            end_seconds=0.25,
            result="Before motion",
            max_frames=1,
        )["object"].object_id
        pack = build_video_evidence_pack(self.root, [evidence, early], max_frames=2)
        self.assertEqual([len(row["frames"]) for row in pack["items"]], [1, 1])
        spread = build_video_evidence_pack(self.root, [evidence], max_frames=2)
        self.assertEqual(
            [row["timestamp_seconds"] for row in spread["items"][0]["frames"]],
            [0.5, 1.25],
        )

    def test_corrupt_cached_frame_is_rejected_before_reuse(self):
        evidence = self._evidence()
        payload = self.repository.get(evidence)["payload"]
        path = _blob(self.repository, payload["frames"][0]["artifact_hash"])
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        with mock.patch("xscientist.video_evidence._extract_frames") as extract:
            with self.assertRaisesRegex(ResearchGitError, "hash mismatch"):
                record_video_evidence(
                    self.root,
                    payload["source_id"],
                    start_seconds=0.25,
                    end_seconds=1.75,
                    max_frames=3,
                    result="Another observation",
                )
        extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
