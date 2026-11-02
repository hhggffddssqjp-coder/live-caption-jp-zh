"""畫面尺寸與排程的小工具，不依賴 GUI 或擷取裝置。"""


def fit_frame_size(width, height, max_width, target_width=0, target_height=0):
    """等比縮進限制範圍；0 表示該方向不限，只縮小、不放大。"""
    if width <= 0 or height <= 0:
        raise ValueError("畫面寬高必須大於 0")
    ratios = [(1, 1)]
    for limit, source in ((max_width, width), (target_width, width), (target_height, height)):
        if limit > 0:
            ratios.append((limit, source))
    numerator, denominator = min(ratios, key=lambda pair: pair[0] / pair[1])
    # 用整數運算向下取整，避免浮點誤差多縮一個像素，奇數尺寸也不用補成偶數。
    return max(1, width * numerator // denominator), max(1, height * numerator // denominator)


class DebouncedSize:
    """最後一次尺寸變化滿 300 ms 才發布；排程函式可用 Tk after 或假時鐘。"""

    def __init__(self, schedule, cancel, publish, delay_ms=300):
        self.schedule = schedule
        self.cancel = cancel
        self.publish = publish
        self.delay_ms = delay_ms
        self._job = None
        self._pending = None
        self._published = None

    def update(self, width, height):
        size = (int(width), int(height))
        # Tk 佈局完成前常回報 1x1，最小化也不應把後續影格縮成一個點。
        if min(size) <= 10:
            if self._job is not None:
                self.cancel(self._job)
            self._job = self._pending = None
            return
        if size == self._pending:
            return
        if self._job is not None:
            self.cancel(self._job)
        self._job = self._pending = None
        if size == self._published:
            return
        self._pending = size
        self._job = self.schedule(self.delay_ms, self._flush)

    def _flush(self):
        size = self._pending
        self._job = self._pending = None
        self._published = size
        self.publish(*size)
