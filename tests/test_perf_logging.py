"""效能統計不需要 GUI、GPU 或真正的 numpy；避免受到整合測試假模組影響。"""

import ast
import csv
import gc
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

from perf_stats import PerfStats


APP_PATH = Path(__file__).resolve().parents[1] / "live_caption_gemini.py"


def load_app_parts(*names, **namespace):
    # 直接執行原始類別，不載入音訊、模型等與這些純邏輯測試無關的依賴。
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace.update(threading=threading, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


class PerfStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # 不註冊到真正 GC，以免測試框架本身的回收造成不可預期的計數。
        callbacks = mock.patch.object(gc, "callbacks", [])
        callbacks.start()
        self.addCleanup(callbacks.stop)
        self.buffer = mock.Mock()
        self.buffer.perf_snapshot.return_value = (3, 600, 1.5)
        with mock.patch("perf_stats.threading.Thread"), mock.patch("builtins.print"):
            self.stats = PerfStats(self.buffer, Path(self.temp.name) / "logs")
        self.addCleanup(self.stats._file.close)
        self.addCleanup(self.stats.close)
        self.memory = mock.patch.object(self.stats, "_process")
        process = self.memory.start()
        self.addCleanup(self.memory.stop)
        process.memory_info.return_value = types.SimpleNamespace(rss=1000, private=2000)

    def test_aggregation_and_csv_columns(self):
        for name in PerfStats.TIMINGS:
            self.stats.timing(name, 2)
            self.stats.timing(name, 6)
        for started in (1.0, 1.02, 1.10, 1.23):
            self.stats.poll_started(started)
        self.stats.painted()
        self.stats.record_gc(3, 0)
        self.stats.record_gc(7, 2)
        self.stats.activity("asr", True)
        self.stats.activity("asr", False)
        self.stats.activity("translation", True)
        with mock.patch("perf_stats.psutil.cpu_percent", return_value=75), \
                mock.patch("perf_stats.psutil.virtual_memory", return_value=types.SimpleNamespace(available=9000)):
            self.stats._writer.writerow(self.stats.snapshot())
        self.stats._file.flush()
        with self.stats.path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            row = next(reader)
            self.assertEqual(reader.fieldnames, list(PerfStats.FIELDS))
        self.assertEqual(int(row["received_frames"]), 2)
        self.assertEqual(int(row["poll_calls"]), 4)
        self.assertEqual(int(row["painted_frames"]), 1)
        self.assertEqual(int(row["buffer_frames"]), 3)
        self.assertEqual(int(row["buffer_bytes"]), 600)
        self.assertEqual(float(row["buffer_span_sec"]), 1.5)
        for name in PerfStats.TIMINGS:
            self.assertEqual(int(row[f"{name}_count"]), 2)
            self.assertEqual(float(row[f"{name}_avg_ms"]), 4)
            self.assertEqual(float(row[f"{name}_max_ms"]), 6)
        self.assertAlmostEqual(float(row["poll_interval_avg_ms"]), 230 / 3)
        self.assertAlmostEqual(float(row["poll_interval_max_ms"]), 130)
        self.assertEqual(int(row["poll_interval_gt50_count"]), 2)
        self.assertEqual(int(row["poll_interval_gt100_count"]), 1)
        self.assertEqual(int(row["gc_count"]), 2)
        self.assertEqual(float(row["gc_total_ms"]), 10)
        self.assertEqual(float(row["gc_max_ms"]), 7)
        self.assertEqual([int(row[f"gc_gen{g}_count"]) for g in range(3)], [1, 0, 1])
        for name, value in (("asr_active", 1), ("translation_active", 1), ("rss_bytes", 1000),
                            ("private_bytes", 2000), ("system_available_bytes", 9000)):
            self.assertEqual(int(row[name]), value)
        self.assertEqual(float(row["system_cpu_percent"]), 75)
        self.assertTrue(row["timestamp"])

    def test_reset_keeps_poll_interval_and_activity_across_seconds(self):
        self.stats.poll_started(1.0)
        self.stats.activity("asr", True)
        self.stats.snapshot()
        self.stats.poll_started(1.2)
        self.stats.activity("asr", False)
        row = self.stats.snapshot()
        self.assertEqual(row["poll_calls"], 1)
        self.assertAlmostEqual(row["poll_interval_max_ms"], 200)
        self.assertEqual(row["asr_active"], 1)
        row = self.stats.snapshot()
        self.assertEqual(row["asr_active"], 0)
        self.assertEqual(row["gc_count"], 0)
        self.assertEqual(row["receive_copy_push_count"], 0)

    def test_gc_callback_measures_pause_and_generation(self):
        with mock.patch("perf_stats.time.perf_counter", side_effect=[10, 10.025]):
            self.stats._gc_callback("start", {"generation": 1})
            self.stats._gc_callback("stop", {"generation": 1})
        row = self.stats.snapshot()
        self.assertAlmostEqual(row["gc_total_ms"], 25)
        self.assertEqual(row["gc_gen1_count"], 1)

    def test_memory_unavailable_stays_empty(self):
        self.stats._process.memory_info.side_effect = OSError("unavailable")
        row = self.stats.snapshot()
        self.assertEqual(row["rss_bytes"], "")
        self.assertEqual(row["private_bytes"], "")

    def test_process_unavailable_stays_empty(self):
        self.stats._process = None
        row = self.stats.snapshot()
        self.assertEqual(row["rss_bytes"], "")
        self.assertEqual(row["private_bytes"], "")

    def test_private_unavailable_keeps_rss(self):
        self.stats._process.memory_info.return_value = types.SimpleNamespace(rss=100)
        row = self.stats.snapshot()
        self.assertEqual(row["rss_bytes"], 100)
        self.assertEqual(row["private_bytes"], "")

    def test_close_flushes_final_row_on_background_thread(self):
        with mock.patch("builtins.print"):
            stats = PerfStats(self.buffer, Path(self.temp.name) / "real")
        stats.timing("receive_copy_push", 4)
        stats.close()
        stats.close()  # finally 與 atexit 重複結束也安全
        self.assertFalse(stats._thread.is_alive())
        self.assertTrue(stats._file.closed)
        self.assertNotIn(stats._callback, gc.callbacks)
        with stats.path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["received_frames"]), 1)

    def test_periodic_writer_runs_off_caller_thread(self):
        written = threading.Event()
        thread_ids = []
        original = PerfStats.snapshot

        def snapshot(stats):
            thread_ids.append(threading.get_ident())
            row = original(stats)
            written.set()
            return row

        with mock.patch.object(PerfStats, "snapshot", snapshot), mock.patch("builtins.print"):
            stats = PerfStats(self.buffer, Path(self.temp.name) / "periodic")
            try:
                self.assertTrue(written.wait(5))
            finally:
                stats.close()
        self.assertTrue(all(tid != threading.get_ident() for tid in thread_ids))
        with stats.path.open(encoding="utf-8-sig", newline="") as stream:
            self.assertGreaterEqual(len(list(csv.DictReader(stream))), 2)


class DisabledPerfTests(unittest.TestCase):
    def test_disabled_has_no_import_file_or_thread(self):
        app = load_app_parts("_start_perf_log")
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("builtins.__import__", side_effect=AssertionError("不應載入額外模組")), \
                mock.patch("builtins.open", side_effect=AssertionError("不應建立檔案")), \
                mock.patch.object(threading.Thread, "start", side_effect=AssertionError("不應建立執行緒")):
            self.assertIsNone(app._start_perf_log(object()))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_only_exact_one_enables_logging(self):
        app = load_app_parts("_start_perf_log")
        for value in ("", "0", "true"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"CAPTION_PERF_LOG": value}):
                self.assertIsNone(app._start_perf_log(None))
        with mock.patch.dict(os.environ, {"CAPTION_PERF_LOG": "1"}), \
                mock.patch("perf_stats.PerfStats", return_value="enabled") as factory:
            self.assertEqual(app._start_perf_log("buffer"), "enabled")
            factory.assert_called_once_with("buffer")

    def test_fresh_interpreter_disabled_probe(self):
        # 新行程只執行真正的啟用入口；排除測試本身已匯入 psutil 的影響。
        script = '''
import ast, os, pathlib, sys, threading
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
node = next(n for n in tree.body if getattr(n, "name", None) == "_start_perf_log")
scope = {"os": os}
exec(compile(ast.Module(body=[node], type_ignores=[]), "probe", "exec"), scope)
os.environ.pop("CAPTION_PERF_LOG", None)
modules, threads = set(sys.modules), set(threading.enumerate())
assert scope["_start_perf_log"](None) is None
assert set(sys.modules) == modules
assert set(threading.enumerate()) == threads
assert not pathlib.Path("logs").exists()
print("disabled: no imports, files or threads")
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", script, str(APP_PATH)], cwd=directory,
                                    capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("disabled: no imports, files or threads", result.stdout)


class FrameBufferPerfTests(unittest.TestCase):
    def test_buffer_snapshot_sums_nbytes_and_retained_time_span(self):
        app = load_app_parts("FrameBuffer", deque=deque)
        buffer = app.FrameBuffer(max_seconds=2)
        self.assertEqual(buffer.perf_snapshot(), (0, 0, 0))
        for ts, size in ((1, 10), (2, 20), (3, 30), (4, 40)):
            buffer.push(ts, types.SimpleNamespace(nbytes=size))
        self.assertEqual(buffer.perf_snapshot(), (3, 90, 2))


if __name__ == "__main__":
    unittest.main()
