"""入口在全新直譯器中模擬 Windows spawn，不使用其他測試的相依套件替身。"""

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "live_caption_gemini.py"


class LightEntryTests(unittest.TestCase):
	def run_python(self, source):
		result = subprocess.run(
			[sys.executable, "-c", source, str(ENTRY)], cwd=ROOT,
			capture_output=True, text=True, timeout=30,
		)
		self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

	def test_spawn_reexecution_does_not_import_heavy_dependencies(self):
		self.run_python('''
import runpy
import sys

runpy.run_path(sys.argv[1], run_name="__mp_main__")
for name in ("torch", "faster_whisper", "numpy", "tkinter", "caption_app"):
    assert name not in sys.modules, name
''')

	def test_import_alias_preserves_private_names_and_patch_globals(self):
		self.run_python('''
import importlib
import sys
import types
from unittest import mock

app = types.ModuleType("caption_app")
exec("translation_problem = lambda: 'original'\\ndef _check(): return translation_problem()", app.__dict__)
sys.modules["caption_app"] = app
entry = importlib.import_module("live_caption_gemini")
assert entry is app
assert entry._check() == "original"
with mock.patch("live_caption_gemini.translation_problem", return_value="patched"):
    assert app._check() == "patched"
assert entry._check() == "original"
''')

	def test_cli_forwards_arguments_and_exit_status(self):
		for args, exit_code in (([], 0), (["--rebuild", "recording.wav"], 1), (["--invalid"], 2)):
			with self.subTest(args=args):
				self.run_python('''
import runpy
import sys
import types

entry = sys.argv[1]
expected_args = %r
app = types.ModuleType("caption_app")
calls = []
app._run_cli = lambda argv: calls.append(argv) or %r
sys.modules["caption_app"] = app
sys.argv = [entry] + expected_args
try:
    runpy.run_path(entry, run_name="__main__")
except SystemExit as error:
    assert error.code == %r, error.code
else:
    raise AssertionError("CLI did not exit")
assert calls == [expected_args], calls
''' % (args, exit_code, exit_code))


if __name__ == "__main__":
	unittest.main()
