"""screen_capture_worker.py when Windows Graphics Capture cannot be used: the window must fall back to PrintWindow.

windows_capture is swapped in sys.modules per test, so nothing here needs Windows or the real package.
"""

import importlib
import ctypes
import os
import re
import sys
import threading
import types
import unittest
from unittest import mock


def import_worker():
	# The worker only imports cv2/numpy here; real modules or another test's fakes both work.
	for name in ("cv2", "numpy"):
		try:
			importlib.import_module(name)
		except ImportError:
			stand_in = sys.modules[name] = types.ModuleType(name)
			stand_in.ndarray = object  # annotations are evaluated at import before Python 3.14
	return importlib.import_module("screen_capture_worker")


worker = import_worker()
HWND = 0x1234


class OldWindowsCapture:
	"""windows-capture 1.x signature: no window_hwnd parameter."""

	def __init__(self, cursor_capture=True, draw_border=None, secondary_window=None,
			minimum_update_interval=None, dirty_region=None, monitor_index=None, window_name=None):
		pass


class BrokenNativeCapture:
	def __init__(self, **kwargs):
		raise RuntimeError("synthetic native failure")


class FailingStartCapture:
	def __init__(self, **kwargs):
		self.kwargs = kwargs

	def event(self, handler):
		return handler

	def start_free_threaded(self):
		raise RuntimeError("synthetic start failure")


def fake_windows_capture(capture_class):
	module = types.ModuleType("windows_capture")
	module.WindowsCapture = capture_class
	return module


def broken_windows_capture():
	"""Installed package whose native submodule is gone, as a broken install leaves it."""
	module = types.ModuleType("windows_capture")

	def missing_submodule(name):
		if name == "WindowsCapture":
			raise ModuleNotFoundError(
				"No module named 'windows_capture.windows_capture'", name="windows_capture.windows_capture",
			)
		raise AttributeError(name)

	module.__getattr__ = missing_submodule
	return module


def required_windows_capture_version():
	path = os.path.join(os.path.dirname(worker.__file__), "requirements.txt")
	with open(path, encoding="utf-8") as f:
		match = re.search(r"^windows-capture>=([\d.]+)", f.read(), re.M)
	return match.group(1) if match else None


class StopAfter(threading.Event):
	"""Reports stop after a few checks, so a capture loop that never falls back cannot hang the suite."""

	def __init__(self, checks):
		super().__init__()
		self.checks_left = checks

	def is_set(self):
		self.checks_left -= 1
		return self.checks_left < 0 or super().is_set()


class WgcUnavailableTests(unittest.TestCase):
	def setUp(self):
		printer = mock.patch("builtins.print")
		self.printed = printer.start()
		self.addCleanup(printer.stop)

	def printed_text(self):
		return "\n".join(" ".join(str(arg) for arg in call.args) for call in self.printed.call_args_list)

	def run_wgc(self, windows_capture):
		with mock.patch.dict(sys.modules, {"windows_capture": windows_capture}):
			return worker._run_wgc(
				HWND, mock.Mock(name="writer"), types.SimpleNamespace(value=HWND),
				threading.Event(), threading.Event(),
			)

	def test_missing_package_returns_error_with_install_hint(self):
		self.assertEqual(self.run_wgc(None), "error")
		self.assertIn("windows-capture", self.printed_text())
		self.assertIn("pip install -r requirements.txt", self.printed_text())
		# The window stays on PrintWindow for the rest of the session, so installing alone is not enough
		self.assertIn("重新啟動程式", self.printed_text())

	def test_package_without_window_hwnd_returns_error_with_upgrade_hint(self):
		self.assertEqual(self.run_wgc(fake_windows_capture(OldWindowsCapture)), "error")
		self.assertIn("pip install -U windows-capture", self.printed_text())
		self.assertIn("重新啟動程式", self.printed_text())
		# Windows keeps the loaded 1.x .pyd locked, so upgrading while the program runs can fail
		self.assertIn("先關掉程式", self.printed_text())

	def test_upgrade_hint_names_the_minimum_version_in_requirements(self):
		minimum = required_windows_capture_version()
		self.assertIsNotNone(minimum, "requirements.txt has no windows-capture>= line")
		self.run_wgc(fake_windows_capture(OldWindowsCapture))
		self.assertIn(minimum, self.printed_text())

	def test_unloadable_package_returns_error_without_install_hint(self):
		# Without WindowsCapture the import raises a plain ImportError, not ModuleNotFoundError
		self.assertEqual(self.run_wgc(types.ModuleType("windows_capture")), "error")
		self.assertIn("無法使用 Windows Graphics Capture", self.printed_text())
		self.assertIn("cannot import name", self.printed_text())
		self.assertNotIn("pip install", self.printed_text())

	def test_broken_install_returns_error_without_install_hint(self):
		self.assertEqual(self.run_wgc(broken_windows_capture()), "error")
		self.assertIn("無法使用 Windows Graphics Capture", self.printed_text())
		self.assertIn("No module named 'windows_capture.windows_capture'", self.printed_text())
		self.assertNotIn("pip install", self.printed_text())

	def test_other_constructor_failure_returns_error(self):
		self.assertEqual(self.run_wgc(fake_windows_capture(BrokenNativeCapture)), "error")
		self.assertIn("synthetic native failure", self.printed_text())

	def test_start_failure_returns_error(self):
		self.assertEqual(self.run_wgc(fake_windows_capture(FailingStartCapture)), "error")
		self.assertIn("synthetic start failure", self.printed_text())

	def test_capture_process_switches_window_to_printwindow_when_package_is_missing(self):
		hwnd_shared = types.SimpleNamespace(value=HWND)
		printwindow_hwnds = []

		def fake_printwindow(hwnd, writer, hwnd_shared, stop_event, pause_event, target_fps):
			printwindow_hwnds.append(hwnd)
			stop_event.set()
			return "stop"

		# win32 modules hidden too, so the test runner's own priority is not lowered on Windows
		hidden = {"windows_capture": None, "win32api": None, "win32process": None}
		with mock.patch.dict(sys.modules, hidden), \
				mock.patch.object(worker, "_SlotWriter"), \
				mock.patch.object(worker, "_run_printwindow", fake_printwindow):
			worker.run_capture_process(
				hwnd_shared, 1280, 60, "shm", 0, None, None, StopAfter(checks=5), threading.Event(),
				None, None,
			)
		self.assertEqual(printwindow_hwnds, [HWND])


class FakePixels:
	def __init__(self, height, width, channels=4):
		self.shape = (height, width, channels)
		self.nbytes = height * width * channels

	def reshape(self, *args):
		return self


class FakeSlots:
	"""只模擬像素複製介面，可在複製或寫入途中插入另一端的操作。"""

	def __init__(self, capacity=100):
		self.shape = (worker.NUM_SLOTS, capacity)
		self.on_copy = lambda: None
		self.on_write = lambda: None
		self.copies = 0
		self.pixels = [None] * worker.NUM_SLOTS

	def __getitem__(self, key):
		slot, _ = key
		pixels = self.pixels[slot]

		def copy():
			self.copies += 1
			self.on_copy()
			return pixels

		return types.SimpleNamespace(reshape=lambda *shape: types.SimpleNamespace(copy=copy))

	def __setitem__(self, key, pixels):
		self.on_write()
		self.pixels[key[0]] = pixels


class SharedFrameTests(unittest.TestCase):
	def setUp(self):
		self.slots = FakeSlots()
		self.meta = (ctypes.c_double * (worker.NUM_SLOTS * 3))()
		self.seq = (ctypes.c_uint64 * worker.NUM_SLOTS)()
		self.latest = types.SimpleNamespace(value=-1)

	def complete(self, slot, frame_id):
		idx = slot * 3
		self.meta[idx:idx + 3] = [2, 3, 100.5]
		self.seq[slot] = frame_id * 2
		self.latest.value = slot
		self.slots.pixels[slot] = "complete pixels"

	def read(self, last=0):
		return worker.read_latest_frame(self.slots, self.meta, self.seq, self.latest, last)

	def test_odd_sequence_rejects_write_in_progress_before_copy(self):
		self.assertIsNone(self.read())
		self.complete(0, 1)
		self.seq[0] = 3
		self.assertIsNone(self.read())
		self.assertEqual(self.slots.copies, 0)

	def test_overwrite_during_copy_is_rejected_even_if_writer_finishes(self):
		for after in (3, 8):
			with self.subTest(after=after):
				self.complete(0, 1)
				self.slots.on_copy = lambda: self.seq.__setitem__(0, after)
				self.assertIsNone(self.read())
		self.slots.on_copy = lambda: None
		self.assertEqual(self.read(), (4, 100.5, "complete pixels"))

	def test_same_slot_after_full_rotation_is_a_new_frame(self):
		self.complete(0, 1)
		self.assertEqual(self.read()[0], 1)
		self.assertIsNone(self.read(last=1))
		self.complete(0, 4)
		self.assertEqual(self.read(last=1)[0], 4)

	def test_frame_id_above_double_precision_remains_exact(self):
		frame_id = 2**53 + 1
		self.complete(2, frame_id)
		self.assertEqual(self.read(last=frame_id - 1)[0], frame_id)
		self.assertIsNone(self.read(last=frame_id))

	def test_invalid_dimensions_are_not_copied(self):
		self.complete(0, 1)
		for height, width in ((0, 3), (100, 100), (-1, 3)):
			self.meta[0:2] = [height, width]
			self.assertIsNone(self.read())
		self.assertEqual(self.slots.copies, 0)

	def make_writer(self, max_width=1920, target=None):
		writer = worker._SlotWriter.__new__(worker._SlotWriter)
		writer.slots = self.slots
		writer.meta_array = self.meta
		writer.sequences = self.seq
		writer.latest_slot = self.latest
		writer.max_width = max_width
		writer.target_size = target
		writer.slot_nbytes = 1920 * 1920 * 3
		writer.write_slot = writer.frame_id = 0
		return writer

	def fake_cv2(self):
		return types.SimpleNamespace(
			INTER_AREA=3, COLOR_BGRA2RGB=5,
			resize=mock.Mock(side_effect=lambda img, size, **kw: FakePixels(size[1], size[0])),
			cvtColor=mock.Mock(side_effect=lambda img, code: FakePixels(*img.shape[:2], 3)),
		)

	def test_writer_marks_odd_before_pixels_and_even_before_latest(self):
		writer = self.make_writer()
		observed = []
		self.slots.on_write = lambda: observed.append((self.seq[writer.write_slot], self.latest.value))
		with mock.patch.object(worker, "cv2", self.fake_cv2()):
			for _ in range(4):
				writer.write(FakePixels(2, 3))
		self.assertEqual(observed, [(1, -1), (3, 0), (5, 1), (7, 2)])
		self.assertEqual(list(self.seq), [8, 4, 6])
		self.assertEqual(self.read(last=1)[0], 4)

	def test_writer_resizes_before_color_conversion_using_shared_target(self):
		target = mock.MagicMock()
		target.__getitem__.return_value = [1280, 720]
		writer = self.make_writer(target=target)
		cv = self.fake_cv2()
		with mock.patch.object(worker, "cv2", cv):
			writer.write(FakePixels(1080, 1920))
		cv.resize.assert_called_once()
		self.assertEqual(cv.resize.call_args.args[1], (1280, 720))
		self.assertEqual(cv.resize.call_args.kwargs, {"interpolation": cv.INTER_AREA})
		self.assertEqual(cv.cvtColor.call_args.args[0].shape, (720, 1280, 4))
		self.assertEqual(list(self.meta[:2]), [720, 1280])


class WgcPublishTests(unittest.TestCase):
	def test_callback_throttles_before_accessing_pixels(self):
		class Frame:
			reads = 0

			@property
			def frame_buffer(self):
				self.reads += 1
				return "pixels"

		class Capture:
			def __init__(self, **kwargs):
				self.handlers = {}

			def event(self, callback):
				self.handlers[callback.__name__] = callback
				return callback

			def start_free_threaded(self):
				for _ in range(120):
					self.handlers["on_frame_arrived"](frame, control)
				stop.set()
				return control

		stop = threading.Event()
		frame = Frame()
		control = mock.Mock()
		control.is_finished.return_value = True
		writer = mock.Mock()
		with mock.patch.dict(sys.modules, {"windows_capture": fake_windows_capture(Capture)}), \
				mock.patch.object(worker.time, "perf_counter", side_effect=[i / 120 for i in range(120)]), \
				mock.patch.object(worker, "_is_black", return_value=False):
			result = worker._run_wgc(HWND, writer, types.SimpleNamespace(value=HWND),
									 stop, threading.Event(), target_fps=30)
		self.assertEqual(result, "stop")
		self.assertEqual(writer.write.call_count, 30)
		self.assertEqual(frame.reads, 30)


if __name__ == "__main__":
	unittest.main()
