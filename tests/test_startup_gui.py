"""Pure-logic tests for startup_gui (no window is opened)."""

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if "tkinter" not in sys.modules:
	sys.modules["tkinter"] = types.ModuleType("tkinter")

from startup_gui import audio_routing_warning, physical_device_name  # noqa: E402

FXSOUND_LOOPBACK = "FxSound Speakers (FxSound Audio Enhancer) [Loopback]"
FXSOUND = "FxSound Speakers (FxSound Audio Enhancer)"
CABLE_LOOPBACK = "CABLE Input (VB-Audio Virtual Cable) [Loopback]"
CABLE = "CABLE Input (VB-Audio Virtual Cable)"
HEADPHONES = "Realtek HD Audio 2nd output (Realtek(R) Audio)"


class AudioRoutingWarningTests(unittest.TestCase):
	def warn(self, input_name, output_name, input_is_default=False, output_is_default=False):
		return audio_routing_warning(
			input_name, output_name, input_is_default, output_is_default, default_output_name=FXSOUND_LOOPBACK,
		)

	def test_loopback_suffix_is_removed_before_comparing(self):
		self.assertEqual(physical_device_name(FXSOUND_LOOPBACK), FXSOUND)
		self.assertEqual(physical_device_name(f"  {FXSOUND}  "), FXSOUND)

	def test_same_device_with_and_without_loopback_suffix_warns(self):
		self.assertIn("同一個裝置", self.warn(FXSOUND_LOOPBACK, FXSOUND))

	def test_both_system_default_warns(self):
		self.assertIn("系統預設", self.warn("系統預設…", "系統預設…", True, True))

	def test_default_input_matching_the_chosen_output_warns(self):
		self.assertIn("同一個裝置", self.warn("系統預設…", FXSOUND, input_is_default=True))

	def test_default_output_matching_the_chosen_input_warns(self):
		self.assertIn("同一個裝置", self.warn(FXSOUND_LOOPBACK, "系統預設…", output_is_default=True))

	def test_playing_into_cable_warns(self):
		self.assertIn("虛擬音訊線", self.warn(FXSOUND_LOOPBACK, CABLE))

	def test_recommended_setup_has_no_warning(self):
		self.assertEqual(self.warn(CABLE_LOOPBACK, HEADPHONES), "")
		self.assertEqual(self.warn(CABLE_LOOPBACK, "系統預設…", output_is_default=True), "")


if __name__ == "__main__":
	unittest.main()
