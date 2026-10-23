"""選用的播放效能紀錄；只有 CAPTION_PERF_LOG=1 時才載入。"""

import atexit
import csv
import datetime
import gc
import threading
import time
from pathlib import Path

import psutil


class PerfStats:
    """熱路徑只累加數值，由背景執行緒每秒取樣並寫入 CSV。"""

    TIMINGS = ("receive_copy_push", "get_at", "resize", "photo", "poll")
    FIELDS = (
        "timestamp", "interval_sec", "received_frames", "buffer_frames", "buffer_bytes",
        "buffer_span_sec", "poll_calls", "painted_frames",
        *(f"{name}_{suffix}" for name in TIMINGS for suffix in ("count", "avg_ms", "max_ms")),
        "poll_interval_avg_ms", "poll_interval_max_ms", "poll_interval_gt50_count",
        "poll_interval_gt100_count", "gc_count", "gc_total_ms", "gc_max_ms",
        "gc_gen0_count", "gc_gen1_count", "gc_gen2_count", "asr_active", "translation_active",
        "rss_bytes", "private_bytes", "system_cpu_percent", "system_available_bytes",
    )

    def __init__(self, frame_buffer, log_dir="logs"):
        self.frame_buffer = frame_buffer
        # GC callback 可能在累加期間由同一執行緒觸發，因此使用可重入鎖。
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._last_poll = None
        self._last_sample = time.perf_counter()
        self._gc_started = {}
        self._active = {"asr": 0, "translation": 0}
        self._reset()
        try:
            self._process = psutil.Process()
        except (psutil.Error, OSError):
            self._process = None
        try:
            psutil.cpu_percent(None)  # 第一筆只建立 CPU 取樣基準
        except (psutil.Error, OSError):
            pass
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = log_dir / f"perf_caption_{stamp}.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._callback = self._gc_callback
        gc.callbacks.append(self._callback)
        self._thread = threading.Thread(target=self._run, name="caption-perf-log", daemon=True)
        self._thread.start()
        atexit.register(self.close)
        print(f"效能紀錄檔：{self.path.resolve()}")

    def _reset(self):
        self._timings = {name: [0, 0.0, 0.0] for name in self.TIMINGS}
        self._poll_calls = self._painted = 0
        self._interval = [0, 0.0, 0.0, 0, 0]
        self._gc = [0, 0.0, 0.0, 0, 0, 0]
        self._touched = {name: bool(count) for name, count in self._active.items()}

    def timing(self, name, elapsed_ms):
        with self._lock:
            values = self._timings[name]
            values[0] += 1
            values[1] += elapsed_ms
            values[2] = max(values[2], elapsed_ms)

    def poll_started(self, started):
        with self._lock:
            self._poll_calls += 1
            if self._last_poll is not None:
                ms = (started - self._last_poll) * 1000
                values = self._interval
                values[0] += 1
                values[1] += ms
                values[2] = max(values[2], ms)
                values[3] += ms > 50
                values[4] += ms > 100
            self._last_poll = started

    def painted(self):
        with self._lock:
            self._painted += 1

    def activity(self, name, running):
        with self._lock:
            self._active[name] += 1 if running else -1
            if running:
                self._touched[name] = True

    def record_gc(self, elapsed_ms, generation):
        with self._lock:
            values = self._gc
            values[0] += 1
            values[1] += elapsed_ms
            values[2] = max(values[2], elapsed_ms)
            values[3 + generation] += 1

    def _gc_callback(self, phase, info):
        generation = info["generation"]
        if phase == "start":
            self._gc_started[generation] = time.perf_counter()
        elif phase == "stop":
            started = self._gc_started.pop(generation, None)
            if started is not None:
                self.record_gc((time.perf_counter() - started) * 1000, generation)

    def snapshot(self):
        """交換這一秒的累計值；取樣與 CSV 格式化都在背景執行緒。"""
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self._last_sample
            self._last_sample = now
            timings, intervals, collections = self._timings, self._interval, self._gc
            calls, painted, touched = self._poll_calls, self._painted, self._touched
            self._reset()
        count, nbytes, span = self.frame_buffer.perf_snapshot()
        row = dict.fromkeys(self.FIELDS, "")
        row.update(timestamp=datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
                   interval_sec=elapsed, received_frames=timings["receive_copy_push"][0],
                   buffer_frames=count, buffer_bytes=nbytes, buffer_span_sec=span,
                   poll_calls=calls, painted_frames=painted,
                   poll_interval_avg_ms=intervals[1] / intervals[0] if intervals[0] else 0,
                   poll_interval_max_ms=intervals[2], poll_interval_gt50_count=intervals[3],
                   poll_interval_gt100_count=intervals[4], gc_count=collections[0],
                   gc_total_ms=collections[1], gc_max_ms=collections[2],
                   gc_gen0_count=collections[3], gc_gen1_count=collections[4], gc_gen2_count=collections[5],
                   asr_active=int(touched["asr"]), translation_active=int(touched["translation"]))
        for name, (samples, total, maximum) in timings.items():
            row[f"{name}_count"] = samples
            row[f"{name}_avg_ms"] = total / samples if samples else 0
            row[f"{name}_max_ms"] = maximum
        # 各項分開讀取，單一權限不足不會讓其他系統資訊一起消失。
        try:
            memory = self._process.memory_info()
            row["rss_bytes"] = memory.rss
            row["private_bytes"] = getattr(memory, "private", "")
        except (psutil.Error, OSError, AttributeError):
            pass
        try:
            row["system_cpu_percent"] = psutil.cpu_percent(None)
        except (psutil.Error, OSError):
            pass
        try:
            row["system_available_bytes"] = psutil.virtual_memory().available
        except (psutil.Error, OSError):
            pass
        return row

    def _run(self):
        try:
            while not self._stop.wait(1.0):
                self._writer.writerow(self.snapshot())
                self._file.flush()
            # 最後不足一秒的樣本也保留，仍由背景執行緒寫檔。
            self._writer.writerow(self.snapshot())
            self._file.flush()
        except Exception as error:
            print(f"效能紀錄寫入失敗：{error}")
        finally:
            self._file.close()

    def close(self):
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)
        self._stop.set()
        self._thread.join()
        atexit.unregister(self.close)
