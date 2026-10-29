"""自我檢查的離線測試：不用顯卡、音效卡或第三方套件。"""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import check_setup as check


class NvidiaTests(unittest.TestCase):
    def test_parse_cards_and_cuda(self):
        cards = check.parse_nvidia_smi('"NVIDIA 顯卡, 測試", 581.00, 16384 MiB\nNVIDIA 小卡, 580.00, 4096 MiB\n無法辨識\n')
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0].name, "NVIDIA 顯卡, 測試")
        self.assertEqual(cards[1].memory_mb, 4096)
        self.assertEqual(check.parse_cuda_version("| Driver Version: 581.00 CUDA Version: 13.0 |"), (13, 0))
        self.assertIsNone(check.parse_cuda_version("CUDA Version: N/A"))
        # 新版驅動的表頭
        self.assertEqual(check.parse_cuda_version("| NVIDIA-SMI 617.14   KMD Version: 617.14   CUDA UMD Version: 13.4 |"), (13, 4))

    @mock.patch("check_setup.subprocess.run")
    def test_commands_are_lists_with_timeout(self, run):
        run.side_effect = [SimpleNamespace(returncode=0, stdout="NVIDIA 測試卡, 580, 8192 MiB"),
                           SimpleNamespace(returncode=0, stdout="CUDA Version: 12.8")]
        cards, cuda = check.detect_nvidia()
        self.assertEqual(cuda, (12, 8))
        self.assertEqual(cards[0].memory_mb, 8192)
        self.assertIn("--format=csv,noheader", run.call_args_list[0].args[0])
        for call in run.call_args_list:
            self.assertIsInstance(call.args[0], list)
            self.assertGreater(call.kwargs["timeout"], 0)
            self.assertNotIn("shell", call.kwargs)

    def test_missing_driver_timeout_and_failed_command(self):
        for failure in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 15)):
            with self.subTest(failure=type(failure).__name__), mock.patch("check_setup.subprocess.run", side_effect=failure):
                self.assertEqual(check.detect_nvidia(), ([], None))
        with mock.patch("check_setup.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="CUDA Version: 13.0")):
            self.assertEqual(check.detect_nvidia(), ([], None))


class RequirementsTests(unittest.TestCase):
    def test_simple_and_unknown_requirements(self):
        self.assertEqual(check.parse_requirements("""
            # 套件註解
            windows-capture >= 2.0.0 # 最低版本
            Pillow
            numpy>=1.0,<3
            google-genai[extra]==1.0
            -r other.txt
        """), [("windows-capture", "2.0.0"), ("Pillow", None), ("numpy", None), ("google-genai", None)])

    def test_versions(self):
        for actual, minimum, expected in [("2.10", "2.9", True), ("2.0", "2.0.0", True),
                                           ("2.0.0rc1", "2.0.0", False), ("2.0.0.dev1", "2.0.0", False),
                                           ("2.0.0.post1", "2.0.0", True), ("2.8.0+cu130", "2.0.0", True),
                                           ("2026.10.19", "2026.10.19", True), ("2.42", "2.43", False)]:
            with self.subTest(actual=actual, minimum=minimum):
                self.assertEqual(check.version_at_least(actual, minimum), expected)
        with self.assertRaises(ValueError):
            check.version_at_least("未知", "2.0")

    def test_missing_old_and_unknown_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "requirements.txt"
            path.write_text("missing\nold>=2.0\nok>=1.0\nodd>=1.0\ncomplex>=5,<8", encoding="utf-8")
            def version(name):
                if name == "missing":
                    raise check.metadata.PackageNotFoundError(name)
                return {"old": "1.9", "ok": "1.2", "odd": "未知", "complex": "0.1"}[name]
            with mock.patch("check_setup.metadata.version", side_effect=version):
                results = check.check_requirements(path)
            self.assertEqual([r.status for r in results], ["❌", "❌", "✅", "⚠️", "✅"])
            self.assertEqual(check.check_requirements(Path(temp) / "missing")[0].status, "❌")


class PrivacyTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {"USERPROFILE": r"C:\Users\測試使用者", "HOME": "/home/tester"})
    def test_mask_paths_and_summary(self):
        text = check.mask_paths(r'C:\Users\測試使用者\下載 C:\Users\Other\Desktop C:/Users/Another/test /home/tester/a /Users/someone/b')
        self.assertNotIn("測試使用者", text)
        for name in ("Other", "Another", "tester", "someone"):
            self.assertNotIn(name, text)
        self.assertIn("%USERPROFILE%", text)
        summary = check.generate_summary([check.Result("❌", "測試", r"C:\Users\Other\a", "重跑 install.bat")],
                                         "Windows 11", "3.12，64 位元", [check.GPU("測試顯卡", "580", 8192)])
        self.assertIn("Windows 11", summary)
        self.assertIn("3.12", summary)
        self.assertIn("測試顯卡", summary)
        self.assertIn("怎麼補", summary)
        self.assertNotIn("Other", summary)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_key_presence_only(self):
        secret = "AIza" + "非常機密的測試值"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(check.has_gemini_key(root))
            (root / "gemini_api_key.txt").write_text(" \n", encoding="utf-8")
            self.assertFalse(check.has_gemini_key(root))
            (root / "gemini_api_key.txt").write_text(secret, encoding="utf-8")
            self.assertTrue(check.has_gemini_key(root))
            summary = check.generate_summary([check.Result("✅", "Gemini 金鑰", "有找到")], "Windows", "3.12", [])
            self.assertNotIn(secret, summary)
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
                self.assertTrue(check.has_gemini_key(root / "不存在"))


class DeviceTests(unittest.TestCase):
    @mock.patch("check_setup.importlib.import_module", side_effect=ImportError)
    def test_import_failures_do_not_escape(self, imports):
        self.assertFalse(check.probe_torch().installed)
        self.assertEqual(check.detect_cable(), ([], False))
        self.assertIsNone(check.file_version("測試.exe"))

    def test_torch_cpu_and_cuda(self):
        for cuda in (True, False):
            module = SimpleNamespace(__version__="2.10.0+cu130" if cuda else "2.10.0+cpu",
                                     cuda=SimpleNamespace(is_available=lambda: cuda),
                                     version=SimpleNamespace(cuda="13.0" if cuda else None))
            with mock.patch("check_setup.importlib.import_module", return_value=module):
                info = check.probe_torch()
            self.assertEqual(info.cuda, cuda)
            self.assertEqual(check.needs_cublas12(info), cuda)

    def test_fresh_torch_uses_a_new_process(self):
        with mock.patch("check_setup.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout='{"installed": true, "version": "2+cpu", "cuda": false, "cuda_build": ""}')) as run:
            self.assertTrue(check.probe_torch(fresh=True).installed)
            self.assertIn("--torch-info", run.call_args.args[0])

    def test_only_cable_inputs_count_and_audio_is_closed(self):
        audio = mock.Mock()
        audio.get_device_count.return_value = 2
        audio.get_device_info_by_index.side_effect = [
            {"name": "CABLE 輸出", "maxInputChannels": 0},
            {"name": "cable 輸入", "maxInputChannels": 2}]
        with mock.patch("check_setup.importlib.import_module", return_value=SimpleNamespace(PyAudio=lambda: audio)):
            self.assertEqual(check.detect_cable(), (["cable 輸入"], True))
        audio.terminate.assert_called_once()

    def test_fallback_to_sounddevice(self):
        module = SimpleNamespace(query_devices=lambda: [{"name": "CABLE", "max_input_channels": 1}])
        with mock.patch("check_setup.importlib.import_module", side_effect=[ImportError(), module]):
            self.assertEqual(check.detect_cable(), (["CABLE"], True))

    def test_successful_enumeration_is_remembered(self):
        audio = mock.Mock()
        audio.get_device_count.return_value = 0
        with mock.patch("check_setup.importlib.import_module", side_effect=[SimpleNamespace(PyAudio=lambda: audio), ImportError()]):
            self.assertEqual(check.detect_cable(), ([], True))

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_cublas_in_package_or_path(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as python_dir,                 mock.patch("check_setup.sys.executable", str(Path(python_dir) / "python.exe")),                 mock.patch("check_setup.sys.base_prefix", python_dir),                 mock.patch.dict(os.environ, {"PATH": ""}):
            path = Path(temp)
            self.assertFalse(check.find_cublas12(path))
            # 放在 python.exe 旁邊也算
            (Path(python_dir) / "cublas64_12.dll").write_bytes(b"test")
            self.assertTrue(check.find_cublas12(path / "不存在"))
            (Path(python_dir) / "cublas64_12.dll").unlink()
            (path / "cublas64_12.dll").write_bytes(b"test")
            self.assertTrue(check.find_cublas12(path))
            with mock.patch.dict(os.environ, {"PATH": str(path)}):
                self.assertTrue(check.find_cublas12(path / "不存在"))

    def test_file_version(self):
        api = SimpleNamespace(GetFileVersionInfo=lambda *_: {"FileVersionMS": (2 << 16) | 43, "FileVersionLS": 0})
        with mock.patch("check_setup.importlib.import_module", return_value=api):
            self.assertEqual(check.file_version("測試.exe"), "2.43.0.0")

    def test_nirsoft_version(self):
        # 網站上的 2.53 在檔案版本裡是 2.5.3.0
        self.assertEqual(check.nirsoft_version("2.5.3.0"), "2.53")
        self.assertTrue(check.version_at_least(check.nirsoft_version("2.5.3.0"), "2.43"))
        self.assertFalse(check.version_at_least(check.nirsoft_version("2.4.2.0"), "2.43"))
        self.assertIsNone(check.nirsoft_version(None))


class FullCheckTests(unittest.TestCase):
    def run_fake_check(self, root, cards, info):
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("check_setup.detect_nvidia", return_value=(cards, (13, 0))))
            stack.enter_context(mock.patch("check_setup.probe_torch", return_value=info))
            stack.enter_context(mock.patch("check_setup.platform.system", return_value="Windows"))
            stack.enter_context(mock.patch("check_setup.platform.machine", return_value="AMD64"))
            stack.enter_context(mock.patch("check_setup.struct.calcsize", return_value=8))
            stack.enter_context(mock.patch("check_setup.metadata.version", side_effect=check.metadata.PackageNotFoundError))
            stack.enter_context(mock.patch("check_setup.find_cublas12", return_value=False))
            # 放最後，避免影響 mock.patch 本身用 importlib 尋找目標。
            stack.enter_context(mock.patch("check_setup.importlib.import_module", side_effect=ImportError))
            stack.enter_context(contextlib.redirect_stdout(output))
            code = check.run_checks(root)
        return code, output.getvalue()

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_missing_modules_low_memory_cpu_and_private_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "requirements.txt").write_text("numpy>=1.0", encoding="utf-8")
            secret = "AIza" + "機密值"
            (root / "gemini_api_key.txt").write_text(secret, encoding="utf-8")
            code, output = self.run_fake_check(root, [check.GPU("測試顯卡", "580", 4096)], check.TorchInfo(True, "2+cpu"))
            self.assertNotEqual(code, 0)
            self.assertIn("❌ PyTorch", output)
            self.assertIn("⚠️ NVIDIA", output)
            self.assertIn("❌ faster_whisper", output)
            self.assertIn("❌ ctranslate2", output)
            self.assertIn("總結", output)
            self.assertNotIn(secret, output)
            self.assertNotIn(str(root), output)

    def test_cu13_missing_dll_is_error(self):
        with tempfile.TemporaryDirectory() as temp:
            code, output = self.run_fake_check(Path(temp), [], check.TorchInfo(True, "2+cu130", True, "13.0"))
        self.assertEqual(code, 1)
        self.assertIn("❌ CUDA 12 cuBLAS", output)

    def test_warning_only_returns_zero(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            root = Path(temp)
            (root / "requirements.txt").write_text("numpy", encoding="utf-8")
            stack.enter_context(mock.patch("check_setup.metadata.version", return_value="2.0"))
            stack.enter_context(mock.patch("check_setup.detect_nvidia", return_value=([], None)))
            stack.enter_context(mock.patch("check_setup.probe_torch", return_value=check.TorchInfo(True, "2+cpu")))
            stack.enter_context(mock.patch("check_setup.detect_cable", return_value=([], True)))
            stack.enter_context(mock.patch("check_setup.platform.system", return_value="Windows"))
            stack.enter_context(mock.patch("check_setup.platform.machine", return_value="AMD64"))
            stack.enter_context(mock.patch("check_setup.struct.calcsize", return_value=8))
            stack.enter_context(mock.patch("check_setup.importlib.import_module", return_value=SimpleNamespace()))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(check.run_checks(root), 0)


if __name__ == "__main__":
    unittest.main()
