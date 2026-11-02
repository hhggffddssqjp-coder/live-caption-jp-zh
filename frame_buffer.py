"""執行緒安全的延遲畫面緩衝，不載入 GUI 或辨識模型。"""

from collections import deque
import math
import os
import threading
import time


# 這個上限只是安全網，不是用來省記憶體的：正常情況（畫面已縮到播放區域大小、
# 每秒最多 60 張）完全碰不到，碰到就會少存影格、播放幀數下降。所以預設給到
# 電腦實體記憶體的一半——1080p、60 fps、16 秒約 6 GB，一般電腦都放得下；
# 只有畫面尺寸異常暴增之類的情況才會擋下來，避免把整台電腦的記憶體吃光。
_FALLBACK_FRAME_BUFFER_MB = 8192.0


def _default_frame_buffer_mb():
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 * 1024) / 2
    except Exception:
        return _FALLBACK_FRAME_BUFFER_MB


def _frame_buffer_mb():
    raw = os.environ.get("CAPTION_FRAME_BUFFER_MB")
    if raw is None:
        return _default_frame_buffer_mb()
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not math.isfinite(value) or value <= 0:
        default = _default_frame_buffer_mb()
        print(f"CAPTION_FRAME_BUFFER_MB={raw!r} 不是有效的正數，改用預設 {default:.0f} MiB（實體記憶體的一半）。")
        return default
    return value


FRAME_BUFFER_MAX_MB = _frame_buffer_mb()  # 以 1024 * 1024 bytes 計算


class FrameBuffer:
    """依時間保存 RGB 陣列；容量不足時拒收新影格，保留既有的播放延遲。"""

    TIMESTAMP_RESET_SEC = 2.0

    def __init__(self, max_seconds: float, max_bytes: int | None = None):
        self.max_seconds = max_seconds
        self.max_bytes = max(1, int(FRAME_BUFFER_MAX_MB * 1024 * 1024)) if max_bytes is None else max_bytes
        if not isinstance(self.max_bytes, int) or self.max_bytes <= 0:
            raise ValueError("畫面緩衝位元組上限必須是正整數")
        self.lock = threading.Lock()
        self.frames = deque()  # [(timestamp, numpy RGB 陣列)]，依時間先後排列
        self.total_bytes = 0
        self._last_ts = None
        self._invalid_since = None
        self._limit_reported = False

    def push(self, ts: float, image):
        with self.lock:
            if not math.isfinite(ts) or (self._last_ts is not None and ts < self._last_ts):
                # 防護計時用單調時鐘，不能依賴正在倒退或失效的擷取時間戳。
                now = time.monotonic()
                if self._invalid_since is None:
                    self._invalid_since = now
                elif now - self._invalid_since >= self.TIMESTAMP_RESET_SEC:
                    self.frames.clear()
                    self.total_bytes = 0
                    self._last_ts = None
                    self._invalid_since = None
                    print("[畫面緩衝] 時間戳持續異常，已清空畫面，等待有效時間重新緩衝。")
                return False
            self._invalid_since = None

            # 即使這張最後因容量不足被跳過，也要讓已超過保存秒數的舊圖自然過期。
            cutoff = ts - self.max_seconds
            while self.frames and self.frames[0][0] < cutoff:
                _, expired = self.frames.popleft()
                self.total_bytes -= expired.nbytes

            nbytes = image.nbytes
            if self.total_bytes + nbytes > self.max_bytes:
                if not self._limit_reported:
                    print("[畫面緩衝] 記憶體已達上限，會跳過部分新影格，保留原本的播放延遲。")
                    self._limit_reported = True
                return False
            self.frames.append((ts, image))
            self.total_bytes += nbytes
            self._last_ts = ts
            return True

    def get_at(self, target_ts: float):
        with self.lock:
            if not self.frames:
                return None
            best = self.frames[0][1]
            for ts, image in self.frames:
                if ts <= target_ts:
                    best = image
                else:
                    break
            return best
