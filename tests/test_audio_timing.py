"""Unit tests for audio_timing (real numpy, fake clock; no audio device needed)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from audio_timing import AudioRingBuffer, EnergySpeechGate, PlaybackDriftTracker  # noqa: E402

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


class RingGapTests(unittest.TestCase):
	def test_gap_larger_than_capacity_keeps_silence_and_new_audio(self):
		ring = AudioRingBuffer(SR, 0.1, channels=2)
		ring.write(block(1, n=27), captured_end=0.027)
		ring.write(block(2, n=13), captured_end=0.340)  # 缺口 300 個樣本
		self.assertEqual(ring.total_written, 340)
		self.assertEqual(ring.write_pos, 40)
		self.assertEqual(ring.start_time, 0.0)
		self.assertAlmostEqual(ring.latest_time(), 0.340)
		expected = np.concatenate([block(0, n=87), block(2, n=13)])
		np.testing.assert_array_equal(ring.read_at(0.240, 100), expected)
		self.assertTrue(np.all(ring.read_at(0.0, 27) == 0))

	def test_hours_long_gap_never_allocates_more_than_capacity(self):
		sample_rate = 48000
		ring = AudioRingBuffer(sample_rate, 0.1, channels=2)
		ring.write(block(1, n=240), captured_end=0.005)
		gap_samples = 6 * 3600 * sample_rate
		real_zeros = np.zeros
		allocations = []

		def bounded_zeros(shape, *args, **kwargs):
			# 先擋住過大的配置，舊版失敗時也不會真的配置數 GB。
			self.assertLessEqual(shape[0], ring.max_samples)
			allocations.append(shape)
			return real_zeros(shape, *args, **kwargs)

		with mock.patch("audio_timing.np.zeros", side_effect=bounded_zeros):
			ring.write(block(2, n=240), captured_end=21600.010)
		self.assertTrue(allocations)
		self.assertEqual(ring.total_written, gap_samples + 480)
		self.assertEqual(ring.write_pos, ring.total_written % ring.max_samples)
		self.assertAlmostEqual(ring.latest_time(), 21600.010)
		expected = np.concatenate([block(0, n=4560), block(2, n=240)])
		np.testing.assert_array_equal(ring.read_at(21599.910, 4800), expected)

	def test_wrapped_reads_match_full_gap_storage(self):
		# 以舊版「整段補零再 _store」作為參照，涵蓋缺口與新資料跨過容量邊界。
		for gap in (64, 128, 333):
			for channels in (1, 2):
				for new_count in (19, 160):
					with self.subTest(gap=gap, channels=channels, new_count=new_count):
						ring = AudioRingBuffer(1024, 0.125, channels=channels)
						reference = AudioRingBuffer(1024, 0.125, channels=channels)
						initial = np.arange(211 * channels, dtype=np.float32).reshape(211, channels)
						new_audio = np.full((new_count, channels), -1, dtype=np.float32)
						for buffer in (ring, reference):
							buffer.write(initial, captured_end=211 / 1024)
						ring.write(new_audio, captured_end=(211 + gap + new_count) / 1024)
						reference._store(np.zeros((gap, channels), dtype=np.float32))
						reference._store(new_audio)
						self.assertEqual(ring.total_written, reference.total_written)
						self.assertEqual(ring.write_pos, reference.write_pos)
						np.testing.assert_array_equal(ring.buffer, reference.buffer)
						for start in (0, ring.total_written - 148, ring.total_written - 50, ring.total_written):
							np.testing.assert_array_equal(
								ring.read_at(start / 1024, 180), reference.read_at(start / 1024, 180),
							)
						# 再繞一圈後，略過的缺口前段也不能影響後續資料。
						for buffer in (ring, reference):
							buffer.write(initial)
						np.testing.assert_array_equal(ring.buffer, reference.buffer)
						np.testing.assert_array_equal(
							ring.read_at(ring.latest_time() - 0.125, 128),
							reference.read_at(reference.latest_time() - 0.125, 128),
						)


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


class EnergySpeechGateTests(unittest.TestCase):
	SR = 16000
	CHUNK = 512  # 32 ms, the same window size the capture loop feeds

	def chunk_at_rms(self, rms_int16):
		# a constant signal has an RMS equal to its amplitude
		return np.full(self.CHUNK, rms_int16 / 32768.0, dtype=np.float32)

	def feed(self, gate, pattern):
		"""pattern: list of (rms, chunks); returns every non-None event with its chunk index."""
		events, index = [], 0
		for rms, count in pattern:
			for _ in range(count):
				event = gate(self.chunk_at_rms(rms))
				if event is not None:
					events.append((index, event))
				index += 1
		return events

	def test_rms_is_measured_in_16_bit_units(self):
		self.assertAlmostEqual(EnergySpeechGate.rms_int16(self.chunk_at_rms(60)), 60, places=2)

	def test_quiet_whisper_above_the_threshold_starts_and_silence_ends_a_sentence(self):
		gate = EnergySpeechGate(rms_threshold=60, silence_end_ms=700, sample_rate=self.SR)
		# 0.3 s of near silence, 2 s of whisper at RMS 120, then 1 s of near silence
		events = self.feed(gate, [(10, 10), (120, 62), (10, 31)])
		self.assertEqual(events[0], (10, {"start": 10 * self.CHUNK}))
		# 700 ms of silence = 22 chunks of 32 ms; the end lands on the 22nd quiet chunk
		self.assertEqual(len(events), 2)
		end_index, end_event = events[1]
		self.assertIn("end", end_event)
		self.assertEqual(end_index, 10 + 62 + 21)

	def test_short_pauses_inside_a_whisper_do_not_split_the_sentence(self):
		gate = EnergySpeechGate(rms_threshold=60, silence_end_ms=700, sample_rate=self.SR)
		events = self.feed(gate, [(120, 20), (10, 10), (120, 20), (10, 25)])  # 320 ms dip in the middle
		self.assertEqual([list(e)[0] for _, e in events], ["start", "end"])

	def test_below_the_threshold_is_never_speech(self):
		gate = EnergySpeechGate(rms_threshold=60, silence_end_ms=700, sample_rate=self.SR)
		self.assertEqual(self.feed(gate, [(55, 100)]), [])

	def test_return_seconds_and_reset(self):
		gate = EnergySpeechGate(rms_threshold=60, silence_end_ms=700, sample_rate=self.SR)
		self.feed(gate, [(10, 5)])
		event = gate(self.chunk_at_rms(200), return_seconds=True)
		self.assertAlmostEqual(event["start"], 5 * self.CHUNK / self.SR)
		gate.reset_states()
		self.assertEqual(gate(self.chunk_at_rms(200)), {"start": 0})


if __name__ == "__main__":
	unittest.main()
