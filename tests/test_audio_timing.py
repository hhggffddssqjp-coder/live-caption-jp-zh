"""Unit tests for audio_timing (real numpy, fake clock; no audio device needed)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from audio_timing import AudioRingBuffer, PlaybackDriftTracker  # noqa: E402

SR = 1000          # small sample rate keeps the numbers readable
BLOCK = 32         # 32 ms blocks, like the real capture loop


def block(value, n=BLOCK, channels=2):
	return np.full((n, channels), value, dtype=np.float32)


class RingLayoutTests(unittest.TestCase):
	def test_stereo_round_trip(self):
		ring = AudioRingBuffer(SR, 2.0, channels=2)
		data = np.stack([np.arange(100, dtype=np.float32), -np.arange(100, dtype=np.float32)], axis=1)
		ring.write(data, captured_end=10.0)
		out = ring.read_at(ring.start_time, 100)
		self.assertEqual(out.shape, (100, 2))
		np.testing.assert_array_equal(out, data)

	def test_channel_count_is_adapted(self):
		stereo = AudioRingBuffer(SR, 1.0, channels=2)
		stereo.write(np.ones(10, dtype=np.float32), captured_end=1.0)  # mono in -> both channels
		np.testing.assert_array_equal(stereo.read_at(stereo.start_time, 10), np.ones((10, 2)))
		mono = AudioRingBuffer(SR, 1.0, channels=1)
		mono.write(np.stack([np.ones(10), np.zeros(10)], axis=1), captured_end=1.0)  # stereo in -> averaged
		np.testing.assert_allclose(mono.read_at(mono.start_time, 10), np.full((10, 1), 0.5))

	def test_nothing_written_yet_reads_silence(self):
		ring = AudioRingBuffer(SR, 1.0, channels=2)
		self.assertTrue(np.all(ring.read_at(0.0, 50) == 0))
		self.assertAlmostEqual(ring.latest_time(), __import__("time").time(), delta=1.0)

	def test_reading_before_the_oldest_kept_sample_pads_with_silence(self):
		ring = AudioRingBuffer(SR, 0.1, channels=1)  # keeps 100 samples
		ring.write(np.arange(1, 201, dtype=np.float32), captured_end=0.2)
		out = ring.read_at(ring.start_time + 0.05, 100)[:, 0]  # samples 50..149; 50..99 are gone
		self.assertTrue(np.all(out[:50] == 0))
		np.testing.assert_array_equal(out[50:], np.arange(101, 151, dtype=np.float32))


class RingClockTests(unittest.TestCase):
	def write_steady(self, ring, start_end, count, value=1.0):
		end = start_end
		for _ in range(count):
			end += BLOCK / SR
			ring.write(block(value), captured_end=end)
		return end

	def test_first_write_anchors_the_timeline_to_the_wall_clock(self):
		ring = AudioRingBuffer(SR, 5.0, channels=2)
		ring.write(block(1.0), captured_end=100.0)
		self.assertAlmostEqual(ring.start_time, 100.0 - BLOCK / SR)
		self.assertAlmostEqual(ring.latest_time(), 100.0)

	def test_dropped_samples_are_replaced_with_silence(self):
		ring = AudioRingBuffer(SR, 10.0, channels=2)
		end = self.write_steady(ring, 100.0, 30)
		gap_start = ring.latest_time()
		# 300 ms of audio never arrived (overflow / reconnect); later blocks are on time again
		end = self.write_steady(ring, end + 0.3, 30, value=2.0)
		self.assertAlmostEqual(ring.latest_time(), end, delta=AudioRingBuffer.DRIFT_THRESHOLD_SEC)
		# about 300 ms of silence was inserted, within a few reads of the gap
		window = AudioRingBuffer.DRIFT_WINDOW * BLOCK / SR
		around_gap = ring.read_at(gap_start, int((0.3 + window + 0.05) * SR))[:, 0]
		self.assertAlmostEqual(int(np.sum(around_gap == 0)), 300, delta=BLOCK)
		self.assertTrue(np.all(ring.read_at(end - 0.2, 100) == 2.0), "audio after the gap keeps its real time")

	def test_a_single_late_timestamp_does_not_insert_silence(self):
		ring = AudioRingBuffer(SR, 10.0, channels=2)
		end = self.write_steady(ring, 100.0, 30)
		before = ring.total_written
		ring.write(block(1.0), captured_end=end + BLOCK / SR + 0.2)  # this read was just scheduled late
		end = self.write_steady(ring, end + BLOCK / SR, 5)
		self.assertEqual(ring.total_written, before + 6 * BLOCK, "no silence for a one-off delay")

	def test_backlog_reads_without_a_timestamp_are_not_treated_as_gaps(self):
		ring = AudioRingBuffer(SR, 10.0, channels=2)
		end = self.write_steady(ring, 100.0, 30)
		before = ring.total_written
		for _ in range(10):  # stalled, then drained the backlog: caller passes captured_end=None
			ring.write(block(1.0))
		end += 10 * BLOCK / SR
		self.write_steady(ring, end, 25)
		self.assertEqual(ring.total_written, before + 35 * BLOCK)

	def test_capture_clock_running_fast_drops_the_extra_samples(self):
		ring = AudioRingBuffer(SR, 10.0, channels=2)
		end = self.write_steady(ring, 100.0, 30)
		# 100 ms more samples than wall time passed
		ring.write(block(1.0, n=BLOCK + 100), captured_end=end + BLOCK / SR)
		self.assertAlmostEqual(ring.latest_time(), end + BLOCK / SR, delta=AudioRingBuffer.DRIFT_THRESHOLD_SEC)

	def test_rebase_moves_now_without_inserting_silence(self):
		ring = AudioRingBuffer(SR, 10.0, channels=2)
		self.write_steady(ring, 100.0, 30)
		before = ring.total_written
		ring.rebase_clock(now=500.0)  # e.g. resuming after a pause
		self.assertAlmostEqual(ring.latest_time(), 500.0)
		self.write_steady(ring, 500.0, 30)
		self.assertEqual(ring.total_written, before + 30 * BLOCK)


class PlaybackDriftTrackerTests(unittest.TestCase):
	def test_first_measurement_is_the_baseline(self):
		tracker = PlaybackDriftTracker(threshold=0.08, interval=5.0)
		self.assertEqual(tracker.correction(heard_time=94.7, now=100.0, delay=5.0), 0.0)

	def test_checks_only_every_interval(self):
		tracker = PlaybackDriftTracker(threshold=0.08, interval=5.0)
		tracker.correction(95.0, 100.0, 5.0)
		self.assertEqual(tracker.correction(95.0 + 1.0, 101.0 + 0.5, 5.0), 0.0)  # 1.5 s later: too soon

	def test_small_drift_is_ignored(self):
		tracker = PlaybackDriftTracker(threshold=0.08, interval=5.0)
		tracker.correction(95.0, 100.0, 5.0)
		self.assertEqual(tracker.correction(100.0 - 0.05, 105.0, 5.0), 0.0)

	def test_audio_falling_behind_skips_forward(self):
		tracker = PlaybackDriftTracker(threshold=0.08, interval=5.0)
		tracker.correction(95.0, 100.0, 5.0)
		self.assertAlmostEqual(tracker.correction(100.0 - 0.2, 105.0, 5.0), 0.2)

	def test_audio_running_ahead_steps_back(self):
		tracker = PlaybackDriftTracker(threshold=0.08, interval=5.0)
		tracker.correction(95.0, 100.0, 5.0)
		self.assertAlmostEqual(tracker.correction(100.0 + 0.15, 105.0, 5.0), -0.15)

	def test_reset_measures_a_new_baseline(self):
		tracker = PlaybackDriftTracker(threshold=0.08, interval=5.0)
		tracker.correction(95.0, 100.0, 5.0)
		tracker.reset()
		self.assertEqual(tracker.correction(150.0, 105.0, 5.0), 0.0)


if __name__ == "__main__":
	unittest.main()
