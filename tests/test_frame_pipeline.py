"""畫面縮放與去抖動測試，不需要 GUI。"""

import unittest

from frame_pipeline import DebouncedSize, fit_frame_size


class FrameSizeTests(unittest.TestCase):
    def test_aspect_ratios_and_limits(self):
        cases = [
            ((1920, 1080, 1920, 1280, 720), (1280, 720)),
            ((1920, 1080, 1920, 1000, 1000), (1000, 562)),
            ((1080, 1920, 1920, 1280, 720), (405, 720)),
            ((1200, 1200, 1920, 1280, 720), (720, 720)),
            ((1920, 1080, 1280, 0, 0), (1280, 720)),
            ((640, 360, 1920, 1280, 720), (640, 360)),
            ((1921, 1081, 1920, 1279, 719), (1277, 719)),
            ((101, 51, 100, 0, 0), (100, 50)),
            ((1920, 1080, 1920, 0, 720), (1280, 720)),
            ((1920, 1080, 1920, 1280, 0), (1280, 720)),
            ((1, 1000, 1920, 100, 1), (1, 1)),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(fit_frame_size(*args), expected)

    def test_invalid_source(self):
        with self.assertRaises(ValueError):
            fit_frame_size(0, 100, 1920)


class FakeScheduler:
    def __init__(self):
        self.now = 0
        self.serial = 0
        self.jobs = {}

    def after(self, delay, callback):
        self.serial += 1
        self.jobs[self.serial] = (self.now + delay, callback)
        return self.serial

    def cancel(self, job):
        self.jobs.pop(job)

    def advance(self, milliseconds):
        self.now += milliseconds
        for job, (due, callback) in list(self.jobs.items()):
            if due <= self.now:
                del self.jobs[job]
                callback()


class DebouncedSizeTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeScheduler()
        self.sizes = []
        self.size = DebouncedSize(self.clock.after, self.clock.cancel,
                                  lambda w, h: self.sizes.append((w, h)))

    def test_only_last_size_published_after_quiet_period(self):
        self.size.update(640, 360)
        self.clock.advance(200)
        self.size.update(1280, 720)
        self.clock.advance(299)
        self.assertEqual(self.sizes, [])
        self.clock.advance(1)
        self.assertEqual(self.sizes, [(1280, 720)])

    def test_duplicate_event_does_not_postpone_publication(self):
        self.size.update(640, 360)
        self.clock.advance(200)
        self.size.update(640, 360)
        self.clock.advance(100)
        self.size.update(640, 360)
        self.assertEqual(self.sizes, [(640, 360)])
        self.assertEqual(self.clock.jobs, {})

    def test_minimize_cancels_pending_without_publishing_tiny_size(self):
        self.size.update(640, 360)
        self.size.update(1, 1)
        self.clock.advance(500)
        self.assertEqual(self.sizes, [])
        self.size.update(1280, 720)
        self.clock.advance(300)
        self.assertEqual(self.sizes, [(1280, 720)])

    def test_return_to_published_size_cancels_change(self):
        self.size.update(640, 360)
        self.clock.advance(300)
        self.size.update(800, 600)
        self.size.update(640, 360)
        self.clock.advance(300)
        self.assertEqual(self.sizes, [(640, 360)])


if __name__ == "__main__":
    unittest.main()
