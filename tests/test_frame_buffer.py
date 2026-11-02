"""畫面容量與時間戳防護，使用小型替身避免配置大量記憶體。"""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import frame_buffer
from frame_buffer import FrameBuffer


def image(nbytes=6):
    return SimpleNamespace(nbytes=nbytes)


class FrameBufferTests(unittest.TestCase):
    def test_limit_thins_evenly_and_keeps_the_oldest_frame(self):
        buf = FrameBuffer(16, max_bytes=60)  # 放得下 10 張
        frames = [image() for _ in range(40)]
        with mock.patch("builtins.print") as printed:
            accepted = [buf.push(i / 60, frame) for i, frame in enumerate(frames)]
        self.assertEqual(printed.call_count, 1)
        self.assertTrue(accepted[0] and accepted[10])  # 第一次碰上限時新影格照收
        self.assertIs(buf.frames[0][1], frames[0])  # 延遲起點沒被剪掉
        self.assertLessEqual(buf.total_bytes, 60)
        stamps = [ts for ts, _ in buf.frames]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        # 沒有好幾秒的空檔：最大間隔只是原本的幾倍，不是整段停住
        self.assertLessEqual(max(gaps), 8 / 60 + 1e-9)
        self.assertEqual(buf.total_bytes, sum(f.nbytes for _, f in buf.frames))

    def test_density_recovers_when_usage_drops(self):
        buf = FrameBuffer(1, max_bytes=600)
        with mock.patch("builtins.print"):
            for i in range(30):
                buf.push(i / 60, image(60))
            self.assertGreater(buf._stride, 1)
            # 視窗縮小：之後的影格小很多（全密度也只用 60 bytes），用量掉到四分之一以下就逐步恢復
            for i in range(30, 400):
                buf.push(i / 60, image(1))
        self.assertEqual(buf._stride, 1)

    def test_oversized_frame_is_never_retained(self):
        buf = FrameBuffer(16, max_bytes=5)
        with mock.patch("builtins.print"):
            self.assertFalse(buf.push(1, image(6)))
        self.assertEqual(buf.total_bytes, 0)
        self.assertIsNone(buf.get_at(1))

    def test_timestamp_rollback_clears_after_two_monotonic_seconds(self):
        buf = FrameBuffer(16, max_bytes=24)
        buf.push(100, image())
        with mock.patch.object(frame_buffer.time, "monotonic", side_effect=[10, 11.9, 12]), \
                mock.patch("builtins.print"):
            self.assertFalse(buf.push(90, image()))
            self.assertFalse(buf.push(91, image()))
            self.assertEqual(buf.total_bytes, 6)
            self.assertFalse(buf.push(92, image()))
        self.assertEqual(buf.total_bytes, 0)
        self.assertEqual(len(buf.frames), 0)
        self.assertTrue(buf.push(93, image()))
        self.assertEqual(buf.total_bytes, 6)

    def test_nonfinite_timestamps_never_enter_buffer_and_can_trigger_reset(self):
        buf = FrameBuffer(16, max_bytes=24)
        buf.push(100, image())
        with mock.patch.object(frame_buffer.time, "monotonic", side_effect=[0, 1, 2]), \
                mock.patch("builtins.print"):
            for ts in (float("nan"), float("inf"), -float("inf")):
                self.assertFalse(buf.push(ts, image()))
        self.assertEqual(buf.total_bytes, 0)
        self.assertTrue(buf.push(10, image()))

    def test_valid_timestamp_resets_invalid_streak_and_equal_is_allowed(self):
        buf = FrameBuffer(16, max_bytes=24)
        buf.push(100, image())
        with mock.patch.object(frame_buffer.time, "monotonic", side_effect=[0, 3]):
            self.assertFalse(buf.push(90, image()))
            self.assertTrue(buf.push(100, image()))
            self.assertFalse(buf.push(91, image()))
        self.assertEqual(len(buf.frames), 2)
        self.assertEqual(buf.total_bytes, 12)

    def test_environment_override_and_invalid_values(self):
        with mock.patch.dict(os.environ, {"CAPTION_FRAME_BUFFER_MB": "1.5"}):
            self.assertEqual(frame_buffer._frame_buffer_mb(), 1.5)
        for raw in ("", "0", "-1", "nan", "inf", "oops", "1e308", "1e20"):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {"CAPTION_FRAME_BUFFER_MB": raw}), \
                    mock.patch("builtins.print"):
                with mock.patch.object(frame_buffer, "_default_frame_buffer_mb", return_value=32768.0):
                    self.assertEqual(frame_buffer._frame_buffer_mb(), 32768.0)
        with mock.patch.dict(os.environ, {}, clear=True),                 mock.patch.object(frame_buffer, "_default_frame_buffer_mb", return_value=32768.0):
            self.assertEqual(frame_buffer._frame_buffer_mb(), 32768.0)
        with mock.patch.object(frame_buffer, "FRAME_BUFFER_MAX_MB", 1.5):
            self.assertEqual(FrameBuffer(16).max_bytes, 1572864)
        for cap in (0, -1, 0.5):
            with self.assertRaises(ValueError):
                FrameBuffer(16, max_bytes=cap)


if __name__ == "__main__":
    unittest.main()


class SafetyNetDefaultTests(unittest.TestCase):
    def test_default_is_half_of_physical_ram(self):
        fake = mock.Mock()
        fake.virtual_memory.return_value.total = 64 * 1024 ** 3
        with mock.patch.dict(sys.modules, {"psutil": fake}):
            self.assertEqual(frame_buffer._default_frame_buffer_mb(), 32768.0)

    def test_1080p_at_60_fps_for_16_seconds_never_hits_the_default_cap_on_16_gb(self):
        # 正常大小的畫面不能因為上限被丟掉，不然播放幀數會下降
        frame = 1920 * 1080 * 3
        cap = int(16 * 1024 / 2 * 1024 * 1024)  # 16 GB 電腦的一半
        buf = FrameBuffer(16.0, max_bytes=cap)
        image = mock.Mock(nbytes=frame)
        accepted = sum(buf.push(i / 60, image) for i in range(60 * 30))
        self.assertEqual(accepted, 60 * 30)
