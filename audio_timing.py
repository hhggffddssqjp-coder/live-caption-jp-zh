"""
audio_timing.py
延遲播放的聲音要跟畫面、字幕用同一個時鐘（time.time()）走，這裡放跟「時間對齊」
有關的純邏輯：AudioRingBuffer（對齊牆上時鐘的多聲道環形緩衝區）和播放端的
漂移校正。只依賴 numpy，不碰音訊裝置，所以可以直接寫單元測試。
"""

import threading
import time
from collections import deque

import numpy as np


class AudioRingBuffer:
    """執行緒安全的音訊環形緩衝區：即時寫入音訊，播放時讀取「N 秒前」的位置。

    存的是多聲道音訊（形狀 (樣本數, channels)），延遲播放出來才會是立體聲；給 Whisper
    的單聲道 16kHz 版本是另外算的。

    時間軸對齊牆上時鐘：第 k 個樣本代表 start_time + k / sample_rate 那一刻的聲音。
    畫面和字幕都用 time.time()，聲音也要跟同一個時鐘走，不然只要掉一段樣本（主程式
    太忙來不及讀、串流重連的空檔），聲音就會從此永遠比畫面早，而且誤差會累積。
    write() 帶上 captured_end（這批樣本實際錄完的時間）時會檢查：缺樣本就補靜音，
    多出來（擷取裝置的時鐘比較快）就丟掉一點。
    """

    # 誤差超過這麼多才補靜音 / 丟樣本；小於這個的時間抖動不處理
    DRIFT_THRESHOLD_SEC = 0.05
    # 讀取的時間點偶爾會被系統排程延後（只會讓 captured_end 偏晚、不會偏早），
    # 所以取最近幾次的最小值當作真正的誤差，避免一次延遲就誤補一段靜音。
    # 代價是真的缺樣本時，靜音會晚這幾次讀取（約 5 × 32ms）才補上；
    # 視窗越大越不怕誤判，但補靜音的位置離真正的缺口越遠
    DRIFT_WINDOW = 5

    def __init__(self, sample_rate: int, max_seconds: float, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = max(1, int(channels))
        self.max_samples = max(1, int(sample_rate * max_seconds))
        self.lock = threading.Lock()
        self.buffer = np.zeros((self.max_samples, self.channels), dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0
        # 第一次寫入時才決定時間軸的起點；在那之前沒有任何聲音
        self.start_time: float | None = None
        self._recent_drift = deque(maxlen=self.DRIFT_WINDOW)

    def _as_frames(self, samples: np.ndarray) -> np.ndarray:
        """把寫進來的資料整理成 (樣本數, self.channels)；聲道數不同時降混 / 複製"""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.shape[1] == self.channels:
            return samples
        if self.channels == 1:
            return samples.mean(axis=1, keepdims=True)
        if samples.shape[1] == 1:
            return np.repeat(samples, self.channels, axis=1)
        return samples[:, : self.channels] if samples.shape[1] > self.channels else np.pad(
            samples, ((0, 0), (0, self.channels - samples.shape[1])), mode="edge"
        )

    def rebase_clock(self, now: float | None = None) -> None:
        """讓時間軸的「現在」直接對到 now，不補靜音（暫停後繼續時用，維持原本的暫停行為）"""
        with self.lock:
            if self.start_time is None:
                return
            now = time.time() if now is None else now
            self.start_time = now - self.total_written / self.sample_rate
            self._recent_drift.clear()

    def write(self, samples: np.ndarray, captured_end: float | None = None):
        """寫入一批樣本。captured_end 是這批樣本錄完的牆上時間；只有確定是「即時」
        讀到的資料才傳（讀取被卡住、在追積壓的資料時傳 None），用來對齊時鐘。"""
        frames = self._as_frames(samples)
        with self.lock:
            n = len(frames)
            if n == 0:
                return
            if self.start_time is None:
                self.start_time = (time.time() if captured_end is None else captured_end) - n / self.sample_rate
            elif captured_end is not None:
                expected_end = self.start_time + (self.total_written + n) / self.sample_rate
                self._recent_drift.append(captured_end - expected_end)
                drift = min(self._recent_drift)
                if drift > self.DRIFT_THRESHOLD_SEC:
                    # 少了一段樣本：補等長的靜音，讓後面的聲音回到正確的時間點
                    self._store(np.zeros((int(drift * self.sample_rate), self.channels), dtype=np.float32))
                    self._recent_drift.clear()
                elif captured_end - expected_end < -self.DRIFT_THRESHOLD_SEC:
                    # 樣本比牆上時鐘多（擷取裝置時鐘偏快）：丟掉這批開頭多出來的部分
                    extra = min(n, int(-(captured_end - expected_end) * self.sample_rate))
                    frames = frames[extra:]
                    self._recent_drift.clear()
            self._store(frames)

    def _store(self, samples: np.ndarray):
        """把 (樣本數, channels) 的資料接在最後面寫進環形緩衝區（呼叫端要先拿到 lock）"""
        n = len(samples)
        if n == 0:
            return

        # 寫入位置永遠用「絕對樣本編號 % 緩衝區大小」計算，不能自己單獨累加 write_pos，
        # 不然單次寫入量超過緩衝區容量時（理論上不會發生，但求穩固）位置會對不齊。
        store = samples if n <= self.max_samples else samples[-self.max_samples:]
        store_n = len(store)

        self.total_written += n
        store_start_abs = self.total_written - store_n
        start_pos = store_start_abs % self.max_samples

        end = start_pos + store_n
        if end <= self.max_samples:
            self.buffer[start_pos:end] = store
        else:
            first = self.max_samples - start_pos
            self.buffer[start_pos:] = store[:first]
            self.buffer[:end - self.max_samples] = store[first:]
        self.write_pos = end % self.max_samples

    def read_at(self, target_time: float, n: int) -> np.ndarray:
        """讀出 target_time 開始的 n 個樣本，形狀 (n, channels)；沒有資料的部分是靜音"""
        out = np.zeros((n, self.channels), dtype=np.float32)
        with self.lock:
            if self.start_time is None:
                return out
            elapsed = target_time - self.start_time
            target_sample = int(round(elapsed * self.sample_rate))
            available_start = max(0, self.total_written - self.max_samples)

            offset = 0
            if target_sample < available_start:
                # 要的時間點已經被新資料蓋掉了：前面補靜音，後面接最舊還在的資料
                offset = min(n, available_start - target_sample)
                target_sample = available_start
            if target_sample >= self.total_written:
                return out

            end_sample = min(target_sample + (n - offset), self.total_written)
            count = end_sample - target_sample
            start_pos = target_sample % self.max_samples

            if start_pos + count <= self.max_samples:
                out[offset:offset + count] = self.buffer[start_pos:start_pos + count]
            else:
                first = self.max_samples - start_pos
                out[offset:offset + first] = self.buffer[start_pos:]
                out[offset + first:offset + count] = self.buffer[:count - first]
            return out

    def latest_time(self) -> float:
        """回傳「目前實際已經寫到哪個時間點」，給延遲播放器起算用。

        時間軸對齊牆上時鐘之後，這個值會很接近 time.time()；還沒寫入任何資料時
        直接回傳 time.time()。
        """
        with self.lock:
            if self.start_time is None:
                return time.time()
            return self.start_time + self.total_written / self.sample_rate


class PlaybackDriftTracker:
    """延遲播放器的漂移校正：播放裝置的時鐘跟電腦的時鐘有細微差異（ppm 等級），
    看長節目時聲音會慢慢跑到畫面前面或後面。

    每隔 interval 秒比較一次「現在耳朵聽到的是哪個時間點的聲音」和「應該聽到的
    時間點（現在 - 延遲秒數）」。第一次量到的差當作基準（輸出串流本身的固定延遲），
    之後跟基準相比，偏差超過 threshold 秒就回傳要修正多少秒，由播放器把讀取位置
    往前跳或往回退這麼多（一次很短的接縫，久久才發生一次）。
    """

    def __init__(self, threshold: float = 0.08, interval: float = 5.0):
        self.threshold = threshold
        self.interval = interval
        self.reset()

    def reset(self) -> None:
        """重新量基準（剛開始播放、暫停後繼續時）"""
        self._baseline: float | None = None
        self._last_check = 0.0

    def correction(self, heard_time: float, now: float, delay: float) -> float:
        """回傳讀取位置要加上的秒數；不用修正時回傳 0。

        heard_time：現在耳朵聽到的聲音在 AudioRingBuffer 時間軸上的時間點。
        """
        error = heard_time - (now - delay)
        if self._baseline is None:
            self._baseline = error
            self._last_check = now
            return 0.0
        if now - self._last_check < self.interval:
            return 0.0
        self._last_check = now
        drift = error - self._baseline
        if abs(drift) <= self.threshold:
            return 0.0
        # 聽到的時間點落後（drift < 0）就往前跳；超前就往回退
        return -drift
