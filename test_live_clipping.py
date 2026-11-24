import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import chat_reader
import config
from clip_maker import ClipMaker
from chat_reader import YouTubeChatReader
from hype_detector import HypeDetector
from media_tools import probe_media
from recorder import SegmentInfo, StreamRecorder
from timestamps import (
    ClipRange,
    TimestampError,
    build_hype_range,
    parse_range,
    parse_timestamp,
    validate_range,
)
from video_processor import (
    describe_ytdlp_error,
    normalize_youtube_url,
    probe_video,
    VideoInfo,
    NullChatReader,
    OfflineSegmentProvider,
    VODHypeDetector,
    _has_video_stream,
)


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

class HypeDetectorTests(unittest.TestCase):
    def _detector(self, chat_count, rms_history, current_rms):
        reader = Mock()
        reader.recent.return_value = [object()] * chat_count
        callback = Mock()
        detector = HypeDetector(reader, Mock(), callback)
        detector._chat_history.extend([0.0] * 5)
        detector._loudness_history.extend(rms_history)
        detector._measure_loudness = Mock(return_value=current_rms)
        return detector, callback

    def test_audio_spike_triggers_without_chat(self):
        detector, callback = self._detector(0, [-24.0] * 5, -17.0)

        detector._analyse()

        callback.assert_called_once()
        self.assertEqual(detector.spike_count, 1)

    def test_chat_spike_triggers_without_audio_spike(self):
        detector, callback = self._detector(30, [-20.0] * 5, -20.0)

        detector._analyse()

        callback.assert_called_once()
        self.assertEqual(detector.spike_count, 1)


class YouTubeChatReaderTests(unittest.TestCase):
    def test_uses_fresh_chat_file_and_deno_runtime(self):
        reader = YouTubeChatReader("https://www.youtube.com/watch?v=stream")
        reader._stop_event.set()
        process = Mock()

        with (
            patch.object(chat_reader.uuid, "uuid4", return_value=SimpleNamespace(hex="fresh")),
            patch.object(chat_reader.subprocess, "Popen", return_value=process) as popen,
        ):
            reader._run()

        command = popen.call_args.args[0]
        self.assertIn("--js-runtimes", command)
        self.assertIn("deno", command)
        output_template = command[command.index("-o") + 1]
        self.assertEqual(
            Path(output_template.replace("%(ext)s", "live_chat.json")).name,
            "livechat_fresh.live_chat.json",
        )


class RecorderTests(unittest.TestCase):
    def test_segment_watcher_ignores_files_from_prior_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(config, "TMP_DIR", Path(temp_dir)):
                recorder = StreamRecorder("https://www.youtube.com/watch?v=stream")
                (Path(temp_dir) / "seg_00001.ts").write_bytes(b"stale")
                (Path(temp_dir) / f"{recorder._segment_prefix}_00001.ts").write_bytes(b"current")
                (Path(temp_dir) / f"{recorder._segment_prefix}_00002.ts").write_bytes(b"current")

                with patch(
                    "recorder.time.sleep",
                    side_effect=lambda _seconds: recorder._stop_event.set(),
                ):
                    recorder._watch_segments()

                self.assertEqual([segment.index for segment in recorder._segments], [1])
                self.assertTrue(all(
                    segment.path.name.startswith(recorder._segment_prefix)
                    for segment in recorder._segments
                ))


# ---------------------------------------------------------------------------
# New tests — URL normalization
# ---------------------------------------------------------------------------

class NormalizeYouTubeURLTests(unittest.TestCase):

    def test_watch_url_unchanged(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(normalize_youtube_url(url), url)

    def test_youtu_be_shortlink(self):
        self.assertEqual(
            normalize_youtube_url("https://youtu.be/dQw4w9WgXcQ"),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )

    def test_youtube_live_url(self):
        self.assertEqual(
            normalize_youtube_url("https://www.youtube.com/live/dQw4w9WgXcQ"),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )

    def test_youtube_shorts_url(self):
        self.assertEqual(
            normalize_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )

    def test_non_youtube_url_unchanged(self):
        url = "https://www.twitch.tv/shroud"
        self.assertEqual(normalize_youtube_url(url), url)


# ---------------------------------------------------------------------------
# New tests — VideoInfo / probe_video
# ---------------------------------------------------------------------------

class ProbeVideoTests(unittest.TestCase):

    def _make_meta(self, **kwargs):
        """Build a minimal yt-dlp JSON output dict."""
        defaults = {
            "id": "testid123",
            "title": "Test Video",
            "duration": 300.0,
            "is_live": False,
            "was_live": False,
            "is_upcoming": False,
            "uploader": "TestChannel",
        }
        defaults.update(kwargs)
        return defaults

    def _run_probe(self, meta_dict, returncode=0):
        import json
        with patch("video_processor.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=returncode,
                stdout=json.dumps(meta_dict),
                stderr="",
            )
            return probe_video("https://www.youtube.com/watch?v=testid123")

    def test_regular_video_detected(self):
        info = self._run_probe(self._make_meta())
        self.assertIsNotNone(info)
        self.assertEqual(info.source_type, "video")
        self.assertFalse(info.is_live)
        self.assertFalse(info.was_live)

    def test_live_stream_detected(self):
        info = self._run_probe(self._make_meta(is_live=True, duration=0))
        self.assertIsNotNone(info)
        self.assertEqual(info.source_type, "live")
        self.assertTrue(info.is_live)

    def test_archived_livestream_detected(self):
        info = self._run_probe(self._make_meta(was_live=True))
        self.assertIsNotNone(info)
        self.assertEqual(info.source_type, "archived")
        self.assertTrue(info.was_live)

    def test_private_video_returns_none(self):
        with patch("video_processor.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="ERROR: Private video",
            )
            info = probe_video("https://www.youtube.com/watch?v=private")
        self.assertIsNone(info)

    def test_yt_dlp_timeout_returns_none(self):
        import subprocess
        with patch("video_processor.subprocess.run", side_effect=subprocess.TimeoutExpired("yt-dlp", 60)):
            info = probe_video("https://www.youtube.com/watch?v=slow")
        self.assertIsNone(info)


# ---------------------------------------------------------------------------
# New tests — NullChatReader
# ---------------------------------------------------------------------------

class NullChatReaderTests(unittest.TestCase):

    def test_messages_empty(self):
        reader = NullChatReader()
        self.assertEqual(len(reader.messages), 0)

    def test_recent_returns_empty(self):
        reader = NullChatReader()
        self.assertEqual(reader.recent(60), [])

    def test_start_stop_no_error(self):
        reader = NullChatReader()
        reader.start()
        reader.stop()


# ---------------------------------------------------------------------------
# New tests — OfflineSegmentProvider
# ---------------------------------------------------------------------------

class OfflineSegmentProviderTests(unittest.TestCase):

    def test_prepare_segments_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create a real (empty) dummy video file so the exists() check passes
            dummy_video = tmp_path / "testvid.mp4"
            dummy_video.write_bytes(b"\x00" * 100)
            # Create fake .ts output files so glob() finds them after the mocked ffmpeg call
            for i in range(3):
                (tmp_path / f"vod_testvid_{i:05d}.ts").write_bytes(b"\x00" * 100)

            provider = OfflineSegmentProvider(dummy_video, 90.0)

            with (
                patch.object(config, "TMP_DIR", tmp_path),
                patch("video_processor.subprocess.run") as mock_run,
            ):
                mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
                result = provider.prepare()

            self.assertTrue(result)
            self.assertGreater(provider.segment_count(), 0)

    def test_prepare_fails_on_ffmpeg_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dummy_video = tmp_path / "dummy.mp4"
            dummy_video.write_bytes(b"\x00" * 100)
            provider = OfflineSegmentProvider(dummy_video, 90.0)

            with (
                patch.object(config, "TMP_DIR", tmp_path),
                patch("video_processor.subprocess.run") as mock_run,
            ):
                mock_run.return_value = Mock(returncode=1, stderr="ffmpeg error", stdout="")
                result = provider.prepare()

            self.assertFalse(result)

    def test_get_segments_covering(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dummy_video = tmp_path / "testvid.mp4"
            dummy_video.write_bytes(b"\x00" * 100)
            for i in range(5):
                (tmp_path / f"vod_testvid_{i:05d}.ts").write_bytes(b"\x00" * 100)

            provider = OfflineSegmentProvider(dummy_video, 150.0)

            with (
                patch.object(config, "TMP_DIR", tmp_path),
                patch("video_processor.subprocess.run") as mock_run,
            ):
                mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
                provider.prepare()

            # Requesting 60 s worth should give us at least 2 segments (each 30 s)
            segs = provider.get_segments_covering(60.0)
            total = sum(s.duration for s in segs)
            self.assertGreaterEqual(total, 60.0)

    def test_get_segments_for_window_selects_spike_region(self):
        provider = OfflineSegmentProvider(Path("video.mp4"), 600.0)
        provider._segments = [
            SegmentInfo(Path(f"segment-{i}.ts"), i, 1000.0 + i * 30, 30.0)
            for i in range(20)
        ]

        segments = provider.get_segments_for_window(1300.0, 240.0)

        self.assertEqual(segments[0].index, 9)
        self.assertEqual(segments[-1].index, 17)


class VODHypeDetectorTests(unittest.TestCase):

    def test_video_stream_validation_rejects_audio_only_file(self):
        with patch("video_processor.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            self.assertFalse(_has_video_stream(Path("audio-only.webm")))

        command = mock_run.call_args.args[0]
        self.assertEqual(command[command.index("-select_streams") + 1], "v:0")

    def test_loudness_measurement_uses_ffmpeg_and_parses_mean_volume(self):
        detector = VODHypeDetector()
        with patch("video_processor.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="",
                stderr="[Parsed_volumedetect_0] mean_volume: -18.5 dB",
            )
            reading = detector._measure_loudness_at(Path("video.mp4"), 40.0, 10.0)

        self.assertEqual(reading, -18.5)
        command = mock_run.call_args.args[0]
        self.assertEqual(command[0], config.FFMPEG_BIN)
        self.assertEqual(command[command.index("-ss") + 1], "40.0")

    def test_detects_relative_audio_peak_using_vod_threshold(self):
        detector = VODHypeDetector()
        with (
            patch("video_processor._probe_duration", return_value=600.0),
            patch.object(detector, "_measure_all_windows", return_value=None),
            patch.object(
                detector,
                "_measure_loudness_at",
                side_effect=lambda _path, offset, _duration: -15.0 if 300 <= offset < 320 else -22.0,
            ),
        ):
            spikes = detector.find_spikes(Path("video.mp4"), 1000.0, max_clips=2)

        self.assertEqual(spikes, [1300.0])

    def test_single_pass_scan_results_are_used(self):
        detector = VODHypeDetector()
        samples = [(t * 10.0, -15.0 if t == 30 else -22.0) for t in range(60)]
        with (
            patch.object(detector, "_measure_all_windows", return_value=samples),
            patch.object(detector, "_measure_loudness_at") as per_window,
        ):
            spikes = detector.find_spikes(Path("video.mp4"), 0.0, max_clips=2, duration=600.0)

        self.assertEqual(spikes, [300.0])
        per_window.assert_not_called()

    def test_uses_loudest_windows_when_no_threshold_spikes(self):
        detector = VODHypeDetector()
        with (
            patch("video_processor._probe_duration", return_value=600.0),
            patch.object(detector, "_measure_all_windows", return_value=None),
            patch.object(detector, "_measure_loudness_at", return_value=-20.0),
        ):
            spikes = detector.find_spikes(Path("video.mp4"), 1000.0, max_clips=2)

        self.assertEqual(spikes, [1000.0, 1240.0])

    def test_vod_shorter_than_clip_still_yields_one_moment(self):
        detector = VODHypeDetector()
        samples = [(t * 10.0, -20.0) for t in range(12)]
        with patch.object(detector, "_measure_all_windows", return_value=samples):
            spikes = detector.find_spikes(Path("video.mp4"), 0.0, max_clips=5, duration=120.0)

        self.assertEqual(len(spikes), 1)

    def test_unknown_duration_returns_no_spikes(self):
        detector = VODHypeDetector()
        with (
            patch("video_processor._probe_duration", return_value=0.0),
            patch.object(detector, "_measure_all_windows") as scan,
        ):
            self.assertEqual(detector.find_spikes(Path("video.mp4"), 0.0), [])
        scan.assert_not_called()


class ClipMakerTests(unittest.TestCase):

    def test_blur_filtergraph_connects_overlay_to_title_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_clip = temp_path / "raw.mp4"
            srt_path = temp_path / "subs.srt"
            out_path = temp_path / "output.mp4"
            raw_clip.write_bytes(b"raw")
            srt_path.write_text("", encoding="utf-8")
            maker = ClipMaker(Mock(), Mock())

            try:
                with (
                    patch(
                        "clip_maker.run_ffmpeg",
                        side_effect=lambda command, **_kwargs: (
                            Path(command[-1]).write_bytes(b"encoded"),
                            Mock(returncode=0, stderr="", cancelled=False, timed_out=False),
                        )[1],
                    ) as run,
                    patch.object(maker, "_verify_output", return_value=None),
                ):
                    self.assertTrue(maker._encode_final(
                        raw_clip, srt_path, "100% Can't: believe", out_path, duration=10.0,
                    ))
                    self.assertTrue(out_path.exists())
                    self.assertIn(".partial.mp4", Path(run.call_args.args[0][-1]).name)
                    title_files = list(temp_path.glob("title_*.txt"))
                    self.assertEqual(title_files[0].read_text(encoding="utf-8"), "100% Can't: believe")
            finally:
                maker.shutdown()

        command = run.call_args.args[0]
        filtergraph = command[command.index("-filter_complex") + 1]
        self.assertIn("[bg][fg]overlay=(W-w)/2:(H-h)/2[base];[base]drawtext=", filtergraph)
        self.assertIn("expansion=none", filtergraph)
        self.assertNotIn("[base][base]", filtergraph)
        # Input seeking: -ss/-t must come before -i for accurate re-encoded cuts.
        self.assertLess(command.index("-t"), command.index("-i"))

    def test_encoded_clip_validation_rejects_missing_moov(self):
        maker = ClipMaker(Mock(), Mock())
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                clip_path = Path(temp_dir) / "incomplete.mp4"
                clip_path.write_bytes(b"mdat" * 1000)
                with patch(
                    "clip_maker.subprocess.run",
                    return_value=Mock(
                        returncode=1,
                        stdout="",
                        stderr="moov atom not found",
                    ),
                ):
                    self.assertFalse(maker._validate_encoded_clip(clip_path))
        finally:
            maker.shutdown()


# ---------------------------------------------------------------------------
# New tests — timestamps / clip ranges
# ---------------------------------------------------------------------------

class TimestampTests(unittest.TestCase):

    def test_supported_formats(self):
        self.assertEqual(parse_timestamp("00:02:15"), 135.0)
        self.assertEqual(parse_timestamp("2:15"), 135.0)
        self.assertEqual(parse_timestamp("135"), 135.0)
        self.assertEqual(parse_timestamp("1:00:00.5"), 3600.5)
        self.assertEqual(parse_range("00:02:15-00:02:55"), (135.0, 175.0))

    def test_rejects_negative_and_malformed(self):
        for bad in ("-5", "-00:01:00", "abc", "1:75", "1:61:00", ""):
            with self.assertRaises(TimestampError, msg=bad):
                parse_timestamp(bad)

    def test_validate_range_errors(self):
        with self.assertRaisesRegex(TimestampError, "must be after"):
            validate_range(60, 30, 600)
        with self.assertRaisesRegex(TimestampError, "beyond the video duration"):
            validate_range(500, 700, 600)
        with self.assertRaisesRegex(TimestampError, "beyond the video duration"):
            validate_range(650, 700, 600)
        self.assertEqual(validate_range(135, 175, 600).duration, 40.0)

    def test_hype_range_is_shifted_inside_video(self):
        r = build_hype_range(5.0, 20, 220, 600)
        self.assertEqual((r.start, r.end), (0.0, 240.0))
        r = build_hype_range(590.0, 20, 220, 600)
        self.assertEqual((r.start, r.end), (360.0, 600.0))
        r = build_hype_range(300.0, 20, 220, 600)
        self.assertEqual((r.start, r.end, r.hype_time), (280.0, 520.0, 300.0))

    def test_hype_range_shrinks_for_short_video(self):
        r = build_hype_range(30.0, 20, 220, 90.0)
        self.assertEqual((r.start, r.end), (0.0, 90.0))


class YtDlpErrorTests(unittest.TestCase):

    def test_common_errors_are_described(self):
        self.assertIn("private", describe_ytdlp_error("ERROR: [youtube] x: Private video. Sign in"))
        self.assertIn("age-restricted", describe_ytdlp_error("Sign in to confirm your age"))
        self.assertIn("not started", describe_ytdlp_error("ERROR: This live event will begin in 3 hours"))
        self.assertEqual(
            describe_ytdlp_error("ERROR: [youtube] AAAAAAAAAAA: This video is unavailable"),
            "The video is unavailable.",
        )

    def test_upcoming_live_status_detected(self):
        import json
        meta = {"id": "x", "title": "t", "duration": None, "live_status": "is_upcoming"}
        with patch("video_processor.subprocess.run") as run:
            run.return_value = Mock(returncode=0, stdout=json.dumps(meta), stderr="")
            info = probe_video("https://www.youtube.com/watch?v=x")
        self.assertTrue(info.is_upcoming)
        self.assertFalse(info.is_live)


class FfmpegErrorReportingTests(unittest.TestCase):

    def test_root_cause_line_is_reported_not_cascade_noise(self):
        from media_tools import ProcessResult
        stderr = [
            "Fontconfig error: Cannot load default config file: File not found",
            "[libx264 @ 0000029ceb86cd80] width not divisible by 2 (1921x1080)",
            "[vost#0:0/libx264 @ 0000029ceb86cb00] [enc:libx264 @ 0000029ceb4db240] Error while "
            "opening encoder - maybe incorrect parameters such as bit_rate, rate, width or height.",
            "[fc#0 @ 0000029ce7ad6bc0] Task finished with error code: -542398533",
            "[out#0/mp4 @ 0000029ce7ad5000] Nothing was written into output file, because at "
            "least one of its streams received no packets.",
        ]
        result = ProcessResult(3752568763, [], stderr)
        self.assertEqual(result.last_error_line(), "width not divisible by 2 (1921x1080)")
        self.assertEqual(result.signed_returncode, -542398533)

    def test_urls_are_redacted_in_logged_commands(self):
        from media_tools import format_cmd
        logged = format_cmd(["ffmpeg", "-i", "https://rr1.googlevideo.com/videoplayback?sig=SECRET&key=x"])
        self.assertNotIn("SECRET", logged)
        self.assertIn("googlevideo.com/videoplayback", logged)


class RecorderWindowTests(unittest.TestCase):

    def test_get_segments_between_returns_contiguous_overlap(self):
        recorder = StreamRecorder("https://www.youtube.com/watch?v=stream")
        for idx in (1, 2, 4, 5, 6):
            recorder._segments.append(SegmentInfo(Path(f"s{idx}.ts"), idx, idx * 30.0, 30.0))
        segments = recorder.get_segments_between(100.0, 200.0)
        self.assertEqual([s.index for s in segments], [4, 5, 6])


# ---------------------------------------------------------------------------
# Integration tests — real ffmpeg clip rendering (skipped without ffmpeg)
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    import shutil as _sh
    return bool(_sh.which(config.FFMPEG_BIN) or Path(config.FFMPEG_BIN).exists())


@unittest.skipUnless(_ffmpeg_available(), "ffmpeg not installed")
class SourceClipRenderingTests(unittest.TestCase):
    """Render real clips from a synthetic YouTube-like VP9/Opus source."""

    @classmethod
    def setUpClass(cls):
        import subprocess
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.source = cls.tmp / "source.mp4"
        base = [config.FFMPEG_BIN, "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "20"]
        vp9 = subprocess.run(
            base + ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
                    "-c:a", "libopus", str(cls.source)],
            capture_output=True,
        )
        if vp9.returncode != 0:   # ffmpeg build without libvpx/libopus
            subprocess.run(base + ["-c:v", "libx264", "-c:a", "aac", str(cls.source)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.clips = self.tmp / f"clips_{self._testMethodName}"
        self.work = self.tmp / f"work_{self._testMethodName}"
        self.patches = [
            patch.object(config, "CLIPS_DIR", self.clips),
            patch.object(config, "TMP_DIR", self.work),
            patch.object(config, "OUTPUT_WIDTH", 640),
            patch.object(config, "OUTPUT_HEIGHT", 360),
        ]
        for p in self.patches:
            p.start()
        self.maker = ClipMaker(None, None, captions=False)

    def tearDown(self):
        self.maker.shutdown()
        for p in self.patches:
            p.stop()

    def test_manual_clip_has_exact_duration_and_streams(self):
        result = self.maker.render_source_clip(
            self.source, ClipRange(5.0, 9.0), title="100% it's: [hype]", video_label="vid",
        )
        self.assertTrue(result.ok, result.error)
        media = probe_media(result.path)
        self.assertTrue(media.has_video and media.has_audio)
        self.assertEqual(media.video_codec, "h264")
        self.assertAlmostEqual(media.duration, 4.0, delta=config.CLIP_DURATION_TOLERANCE)
        self.assertEqual(list(self.work.iterdir()), [], "temp job dirs must be cleaned up")

    def test_out_of_range_timestamps_fail_validation(self):
        result = self.maker.render_source_clip(self.source, ClipRange(15.0, 60.0))
        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "validate")
        self.assertIn("beyond the video duration", result.error)
        self.assertEqual(list(self.clips.glob("*.mp4")), [])

    def test_ffmpeg_failure_is_reported_and_not_marked_successful(self):
        from media_tools import ProcessResult
        failing = ProcessResult(1, [], ["Error opening output file: Invalid argument"])
        with patch("clip_maker.run_ffmpeg", return_value=failing):
            result = self.maker.render_source_clip(self.source, ClipRange(1.0, 3.0))
        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "encode")
        self.assertIn("FFmpeg processing failed", result.error)
        self.assertIn("exit code 1", result.error)
        self.assertEqual(self.maker.clips_made, 0)
        self.assertEqual(self.maker.clips_failed, 1)
        self.assertEqual(list(self.clips.glob("*.mp4")), [])

    def test_truncated_output_fails_verification(self):
        result = self.maker.render_source_clip(self.source, ClipRange(0.0, 2.0))
        self.assertTrue(result.ok, result.error)
        # A 2 s file must be rejected when 10 s was requested.
        reason = self.maker._verify_output(result.path, expected_duration=10.0)
        self.assertIn("requested", reason)

    def test_parallel_clips_get_unique_files(self):
        ranges = [ClipRange(0.0, 2.0), ClipRange(0.0, 2.0), ClipRange(4.0, 6.0)]
        futures = [
            self.maker.submit_source_clip(self.source, r, index=i, video_label="vid")
            for i, r in enumerate(ranges, start=1)
        ]
        results = [f.result() for f in futures]
        self.assertTrue(all(r.ok for r in results), [r.error for r in results])
        paths = {r.path for r in results}
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(p.exists() for p in paths))


# ---------------------------------------------------------------------------
# New tests — config defaults
# ---------------------------------------------------------------------------

class ConfigTests(unittest.TestCase):

    def test_output_is_16x9(self):
        ratio = config.OUTPUT_WIDTH / config.OUTPUT_HEIGHT
        self.assertAlmostEqual(ratio, 16 / 9, places=2)

    def test_clip_duration_is_240(self):
        self.assertEqual(config.CLIP_DURATION, 240)

    def test_segment_buffer_covers_clip(self):
        """Buffer must hold enough segments to cover a full clip."""
        total_buffer = config.SEGMENT_BUFFER_KEEP * config.SEGMENT_DURATION
        needed = config.CLIP_LOOKBACK + config.CLIP_DURATION
        self.assertGreaterEqual(
            total_buffer,
            needed,
            msg=(
                f"SEGMENT_BUFFER_KEEP ({config.SEGMENT_BUFFER_KEEP}) × "
                f"SEGMENT_DURATION ({config.SEGMENT_DURATION}) = {total_buffer}s "
                f"must be ≥ CLIP_LOOKBACK + CLIP_DURATION = {needed}s"
            ),
        )

    def test_fill_mode_valid(self):
        self.assertIn(config.OUTPUT_FILL_MODE.lower(), ("blur", "black"))

    def test_include_streamer_cam_valid(self):
        self.assertIn(config.INCLUDE_STREAMER_CAM.lower(), ("auto", "off"))


if __name__ == "__main__":
    unittest.main()
