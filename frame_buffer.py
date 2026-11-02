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
    # 換算成位元組後也要是合理的有限整數（例如 1e308 乘上 1024*1024 會變無限大）
    if not math.isfinite(value) or value <= 0 or not math.isfinite(value * 1024 * 1024)             or value * 1024 * 1024 >= 2 ** 62:
        default = _default_frame_buffer_mb()
        print(f"CAPTION_FRAME_BUFFER_MB={raw!r} 不是有效的正數，改用預設 {default:.0f} MiB（實體記憶體的一半）。")
        return default
    return value


FRAME_BUFFER_MAX_MB = _frame_buffer_mb()  # 以 1024 * 1024 bytes 計算


class FrameBuffer:
    """依時間保存 RGB 陣列。容量不足時把保存的影格均勻抽掉一半、之後新影格也隔張收，
    等於降低保存的 fps，但不縮短延遲、也不會出現好幾秒收不到畫面的空檔。"""

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
        self._stride = 1  # 每幾張新影格收一張；碰到上限才會變大
        self._skip = 0

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
            if nbytes > self.max_bytes:
                return False  # 單張就超過上限，不可能存得下
            # 用量降到四分之一以下（例如視窗縮小了）就慢慢恢復原本的密度
            if self._stride > 1 and self.total_bytes * 4 < self.max_bytes:
                self._stride //= 2
            self._skip += 1
            if self._skip < self._stride:
                self._last_ts = ts
                return False
            self._skip = 0
            while self.total_bytes + nbytes > self.max_bytes and len(self.frames) > 1:
                self._thin_out()
            if self.total_bytes + nbytes > self.max_bytes:
                self.frames.clear()
                self.total_bytes = 0
            self.frames.append((ts, image))
            self.total_bytes += nbytes
            self._last_ts = ts
            return True

    def _thin_out(self):
        """均勻抽掉一半（保留最舊那張，延遲起點不變），之後新影格也改成隔張收"""
        kept = deque()
        for index, (ts, image) in enumerate(self.frames):
            if index % 2 == 0:
                kept.append((ts, image))
            else:
                self.total_bytes -= image.nbytes
        self.frames = kept
        self._stride *= 2
        if not self._limit_reported:
            print("[畫面緩衝] 記憶體已達上限，改成少存一些畫面（播放會稍微不那麼順），延遲不變。")
            self._limit_reported = True

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
