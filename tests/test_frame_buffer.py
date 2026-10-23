"""延遲畫面緩衝與播放狀態測試，不使用真正的 numpy、Tk 或擷取裝置。"""

import math
import queue
import threading
import time
import types
import unittest
from collections import deque
from unittest import mock

from test_perf_logging import load_app_parts


class FrameBufferTests(unittest.TestCase):
    def setUp(self):
        app = load_app_parts("FrameBuffer", deque=deque)
        self.buffer = app.FrameBuffer(max_seconds=10)
        self.first, self.second, self.third = object(), object(), object()

    def test_empty_and_before_oldest_return_none(self):
        self.assertIsNone(self.buffer.get_at(0))
        self.assertIsNone(self.buffer.buffering_seconds(0))
        self.buffer.push(100, self.first)
        self.assertIsNone(self.buffer.get_at(99.9))
        self.assertAlmostEqual(self.buffer.buffering_seconds(99.9), 0.1)

    def test_exact_between_and_after_newest(self):
        for ts, frame in ((100, self.first), (101, self.second), (102, self.third)):
            self.buffer.push(ts, frame)
        self.assertIs(self.buffer.get_at(100), self.first)
        self.assertIs(self.buffer.get_at(101), self.second)
        self.assertIs(self.buffer.get_at(101.9), self.second)
        self.assertIs(self.buffer.get_at(200), self.third)
        self.assertEqual(self.buffer.buffering_seconds(200), 0)

    def test_evicted_frames_change_oldest_boundary(self):
        self.buffer.push(100, self.first)
        self.buffer.push(110, self.second)
        self.assertIs(self.buffer.get_at(100), self.first)
        self.buffer.push(111, self.third)
        self.assertIsNone(self.buffer.get_at(109))
        self.assertEqual(self.buffer.buffering_seconds(109), 1)
        self.assertIs(self.buffer.get_at(110), self.second)

    def test_fifteen_second_startup_countdown(self):
        self.buffer.push(100, self.first)
        for now, seconds in ((100, 15), (103, 12), (114.9, 1)):
            target = now - 15
            self.assertIsNone(self.buffer.get_at(target))
            self.assertEqual(math.ceil(self.buffer.buffering_seconds(target)), seconds)
        self.assertIs(self.buffer.get_at(100), self.first)


class PlayerBufferingTests(unittest.TestCase):
    def setUp(self):
        self.perf = mock.Mock()
        self.app = load_app_parts(
            "FrameBuffer", "PauseState", "PlayerWindow", deque=deque, time=time, math=math,
            queue=queue, tk=types.SimpleNamespace(Tk=object, Frame=object),
            cv2=mock.Mock(INTER_LINEAR=1), Image=mock.Mock(), ImageTk=mock.Mock(),
            CAPTURE_TARGET_FPS=60, DISPLAY_DELAY_SEC=15, _perf_stats=self.perf,
        )
        self.buffer = self.app.FrameBuffer(max_seconds=25)
        self.frame = types.SimpleNamespace(shape=(60, 80, 3), nbytes=14400)
        self.player = self.app.PlayerWindow.__new__(self.app.PlayerWindow)
        self.player.root = mock.Mock()
        self.player.canvas = mock.Mock()
        self.player.canvas.winfo_width.return_value = 640
        self.player.canvas.winfo_height.return_value = 480
        self.player.frame_buffer = self.buffer
        self.player.pause_state = self.app.PauseState()
        self.player.pause_btn = mock.Mock()
        self.player._photo = self.player._photo_size = None
        self.player._last_frame = self.player._image_item = None

    def poll(self, now):
        with mock.patch.object(time, "time", return_value=now):
            self.player._poll_video()

    def toggle_pause(self):
        with mock.patch("builtins.print"):
            self.player._on_toggle_pause()

    def test_buffering_countdown_then_first_frame_and_repeated_frame(self):
        self.buffer.push(100, self.frame)
        self.poll(103)
        self.assertEqual(self.player.canvas.create_text.call_args.kwargs["text"], "緩衝中... 還要 12 秒")
        self.assertIsNone(self.player._last_frame)
        self.app.cv2.resize.assert_not_called()
        self.poll(115)
        self.assertIs(self.player._last_frame, self.frame)
        self.app.ImageTk.PhotoImage.assert_called_once()
        self.poll(115.1)
        self.app.cv2.resize.assert_called_once()
        self.perf.painted.assert_called_once()
        self.assertEqual(self.perf.poll_started.call_count, 3)
        stages = [call.args[0] for call in self.perf.timing.call_args_list]
        self.assertEqual(stages.count("get_at"), 3)
        self.assertEqual(stages.count("resize"), 1)
        self.assertEqual(stages.count("photo"), 1)
        self.assertEqual(stages.count("poll"), 3)

    def test_empty_buffer_and_reset_force_first_frame_to_redraw(self):
        self.buffer.push(100, self.frame)
        self.poll(115)
        with self.buffer.lock:
            self.buffer.frames.clear()
        self.poll(116)
        self.assertEqual(self.player.canvas.create_text.call_args.kwargs["text"], "緩衝中...")
        self.assertIsNone(self.player._last_frame)
        self.assertIsNone(self.player._photo)
        self.assertIsNone(self.player._photo_size)
        self.assertIsNone(self.player._image_item)
        # 同一個物件也要重畫，避免保留上一輪的識別狀態而跳過更新。
        self.buffer.push(102, self.frame)
        self.poll(117)
        self.assertEqual(self.app.ImageTk.PhotoImage.call_count, 2)

    def test_pause_preserves_frame_and_resume_removes_overlay(self):
        self.buffer.push(100, self.frame)
        self.poll(115)
        photo = self.player._photo
        self.toggle_pause()
        with mock.patch.object(self.buffer, "get_at", wraps=self.buffer.get_at) as get_at:
            self.poll(116)
            get_at.assert_not_called()
        self.assertIs(self.player._last_frame, self.frame)
        self.assertIs(self.player._photo, photo)
        self.assertEqual(self.player.canvas.create_text.call_args.kwargs["text"], "⏸ 已暫停")
        self.player.canvas.delete.reset_mock()
        self.toggle_pause()
        self.poll(117)
        self.player.canvas.delete.assert_called_with("pause_overlay")
        self.app.cv2.resize.assert_called_once()

    def test_pause_during_buffering_then_resume(self):
        self.buffer.push(100, self.frame)
        self.poll(101)
        self.toggle_pause()
        self.poll(102)
        self.assertIsNone(self.player._last_frame)
        self.toggle_pause()
        self.poll(103)
        self.assertEqual(self.player.canvas.create_text.call_args.kwargs["text"], "緩衝中... 還要 12 秒")
        self.poll(115)
        self.assertIs(self.player._last_frame, self.frame)

    def test_switch_window_keeps_delayed_timeline_then_pastes_new_frame(self):
        self.buffer.push(100, self.frame)
        self.poll(115)
        chosen = types.SimpleNamespace(title="另一個視窗", visible=True, width=800, height=600, _hWnd=123)
        self.player.screen_capture = mock.Mock()
        with mock.patch.dict("sys.modules", {
            "pygetwindow": types.SimpleNamespace(getAllWindows=lambda: [chosen]),
            "startup_gui": types.SimpleNamespace(pick_single_window=lambda *a, **k: 0),
        }), mock.patch("builtins.print"):
            self.player._on_switch_window()
        self.player.screen_capture.switch_window.assert_called_once_with(123)
        new_frame = types.SimpleNamespace(shape=(60, 80, 3), nbytes=14400)
        self.buffer.push(116, new_frame)
        self.poll(116)
        self.assertIs(self.player._last_frame, self.frame)
        self.poll(131)
        self.assertIs(self.player._last_frame, new_frame)
        self.player._photo.paste.assert_called_once()

    def test_resize_redraws_existing_frame(self):
        self.buffer.push(100, self.frame)
        self.poll(115)
        self.player.canvas.winfo_width.return_value = 320
        self.poll(116)
        self.assertEqual(self.app.ImageTk.PhotoImage.call_count, 2)
        self.assertEqual(self.player._photo_size, (320, 240))

    def test_small_canvas_keeps_state_and_schedules_next_poll(self):
        self.player.canvas.winfo_width.return_value = 1
        self.poll(103)
        self.player.canvas.create_text.assert_not_called()
        self.player.root.after.assert_called_once()


if __name__ == "__main__":
    unittest.main()
