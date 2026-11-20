import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import chat_reader
import config
from chat_reader import YouTubeChatReader
from hype_detector import HypeDetector
from recorder import StreamRecorder
from video_processor import (
    normalize_youtube_url,
    probe_video,
    VideoInfo,
    NullChatReader,
    OfflineSegmentProvider,
    VODHypeDetector,
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
