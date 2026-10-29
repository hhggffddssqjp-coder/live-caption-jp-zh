"""安裝器的離線測試；所有 pip、網路和開網頁操作都用假資料。"""

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import check_setup as check
import install


class TorchChoiceTests(unittest.TestCase):
    def test_cuda_boundaries(self):
        cases = [(None, "cpu"), ((11, 8), "cpu"), ((12, 5), "cpu"), ((12, 6), "cu126"),
                 ((12, 7), "cu126"), ((12, 8), "cu128"), ((12, 9), "cu128"),
                 ((13, 0), "cu130"), ((13, 4), "cu130"), ((14, 0), "cu130")]
        for cuda, suffix in cases:
            with self.subTest(cuda=cuda):
                self.assertEqual(install.choose_torch_index(cuda), "https://download.pytorch.org/whl/" + suffix)

    def test_completed_torch_detection(self):
        gpu_index = install.choose_torch_index((13, 0))
        self.assertTrue(install.torch_is_ready(check.TorchInfo(True, "2+cu128", True, "12.8"), gpu_index))
        cpu = check.TorchInfo(True, "2+cpu", False)
        self.assertTrue(install.torch_is_ready(cpu, install.CPU_INDEX))
        self.assertFalse(install.torch_is_ready(cpu, gpu_index))
        self.assertFalse(install.torch_is_ready(check.TorchInfo(), install.CPU_INDEX))
        self.assertFalse(install.torch_is_ready(check.TorchInfo(True, "2+cu130", False, "13.0"), install.CPU_INDEX))

    @mock.patch("install.run_pip")
    @mock.patch("install.check.probe_torch", return_value=check.TorchInfo(True, "2+cu128", True, "12.8"))
    @mock.patch("install.check.detect_nvidia", return_value=([check.GPU("測試卡", "580", 8192)], (13, 0)))
    def test_existing_cuda_skips_pip(self, detect, probe, pip):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.install_torch())
        pip.assert_not_called()
        probe.assert_called_once_with(fresh=True)

    @mock.patch("install.run_pip", return_value=True)
    @mock.patch("install.check.probe_torch")
    @mock.patch("install.check.detect_nvidia", return_value=([check.GPU("測試卡", "580", 8192)], (12, 8)))
    def test_cpu_is_replaced_with_cuda_and_rechecked(self, detect, probe, pip):
        probe.side_effect = [check.TorchInfo(True, "2+cpu"), check.TorchInfo(True, "2+cu128", True, "12.8")]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.install_torch())
        self.assertIn("--force-reinstall", pip.call_args.args)
        self.assertEqual(pip.call_args.args[-1], install.choose_torch_index((12, 8)))
        self.assertEqual(probe.call_count, 2)


class SkipTests(unittest.TestCase):
    def test_requirements_ready_checks_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "requirements.txt"
            path.write_text("numpy>=2.0", encoding="utf-8")
            with mock.patch("check_setup.metadata.version", return_value="2.1"):
                self.assertTrue(install.requirements_are_ready(path))
            with mock.patch("check_setup.metadata.version", return_value="1.9"):
                self.assertFalse(install.requirements_are_ready(path))
            with mock.patch("check_setup.metadata.version", side_effect=check.metadata.PackageNotFoundError):
                self.assertFalse(install.requirements_are_ready(path))

    @mock.patch("install.run_pip")
    @mock.patch("install.requirements_are_ready", return_value=True)
    def test_completed_requirements_skip(self, ready, pip):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.install_requirements())
        pip.assert_not_called()

    @mock.patch("install.urllib.request.urlopen")
    @mock.patch("install.ask_yes_no")
    def test_existing_soundvolumeview_skips_download_and_question(self, ask, opener):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tools").mkdir()
            (root / "tools" / "SoundVolumeView.exe").write_bytes(b"test")
            with mock.patch("install.ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(install.install_soundvolumeview())
        ask.assert_not_called()
        opener.assert_not_called()

    @mock.patch("install.ask_yes_no")
    @mock.patch("install.check.detect_cable", return_value=(["CABLE"], True))
    def test_existing_cable_skips_question(self, cable, ask):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.setup_cable())
        ask.assert_not_called()

    @mock.patch("install.getpass.getpass")
    @mock.patch("install.offer_page")
    @mock.patch("install.check.has_gemini_key", return_value=True)
    def test_existing_key_skips_input(self, key, page, read):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.setup_key())
        page.assert_not_called()
        read.assert_not_called()


class CublasTests(unittest.TestCase):
    def test_copies_only_dlls_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, dest = root / "source", root / "destination"
            source.mkdir()
            dest.mkdir()
            for name in ("cublas64_12.dll", "cublasLt64_12.dll", "other.dll"):
                (source / name).write_bytes(b"new")
            (dest / "cublas64_12.dll").write_bytes(b"existing")
            self.assertFalse(install.cublas_copy_complete(dest))
            install.copy_cublas_dlls(source, dest)
            self.assertTrue(install.cublas_copy_complete(dest))
            self.assertEqual((dest / "cublas64_12.dll").read_bytes(), b"existing")
            self.assertFalse((dest / "other.dll").exists())

    def test_incomplete_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cublas64_12.dll").write_bytes(b"test")
            with self.assertRaises(FileNotFoundError):
                install.copy_cublas_dlls(root, root)

    @mock.patch("install.run_pip")
    @mock.patch("install.cublas_copy_complete", return_value=True)
    @mock.patch("install.check.package_dir", return_value=Path("ctranslate2"))
    @mock.patch("install.check.probe_torch", return_value=check.TorchInfo(True, "2+cu130", True, "13.0"))
    def test_completed_dlls_skip_pip(self, probe, package, complete, pip):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.install_cublas())
        pip.assert_not_called()

    @mock.patch("install.run_pip")
    @mock.patch("install.check.probe_torch", return_value=check.TorchInfo(True, "2+cu128", True, "12.8"))
    def test_cuda12_skips_extra_dlls(self, probe, pip):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.install_cublas())
        pip.assert_not_called()


class ZipTests(unittest.TestCase):
    def make_zip(self, root, entries):
        archive = root / "test.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, data in entries:
                bundle.writestr(name, data)
        return archive

    def test_only_exe_is_extracted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = self.make_zip(root, [("SoundVolumeView.exe", b"exe"), ("readme.txt", b"text"),
                                           ("nested/other.exe", b"other")])
            destination = install.extract_soundvolumeview(archive, root / "tools")
            self.assertEqual(destination.read_bytes(), b"exe")
            self.assertEqual([p.name for p in (root / "tools").iterdir()], ["SoundVolumeView.exe"])

    def test_rejects_path_traversal_before_writing_anything(self):
        for name in ("../SoundVolumeView.exe", "sub/../../bad", r"..\bad.exe", "/bad.exe",
                     "C:/bad.exe", r"C:\bad.exe", "//host/share/bad.exe", "file:stream"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = self.make_zip(root, [("SoundVolumeView.exe", b"exe"), (name, b"bad")])
                with self.assertRaises(ValueError):
                    install.extract_soundvolumeview(archive, root / "tools")
                self.assertFalse((root / "tools").exists())

    def test_rejects_missing_duplicate_and_symlink(self):
        for entries in ([('readme.txt', b"text")],
                        [("a/SoundVolumeView.exe", b"a"), ("b/SoundVolumeView.exe", b"b")]):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with self.assertRaises(ValueError):
                    install.extract_soundvolumeview(self.make_zip(root, entries), root / "tools")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "test.zip"
            info = zipfile.ZipInfo("SoundVolumeView.exe")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(info, "../target")
            with self.assertRaises(ValueError):
                install.extract_soundvolumeview(archive, root / "tools")

    @mock.patch("install.ask_yes_no", return_value=False)
    @mock.patch("install.urllib.request.urlopen")
    def test_declining_download_never_connects(self, opener, ask):
        with tempfile.TemporaryDirectory() as temp, mock.patch("install.ROOT", Path(temp)), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(install.install_soundvolumeview())
        opener.assert_not_called()

    @mock.patch("install.ask_yes_no", return_value=True)
    @mock.patch("install.urllib.request.urlopen", side_effect=TimeoutError)
    def test_download_timeout_is_handled(self, opener, ask):
        with tempfile.TemporaryDirectory() as temp, mock.patch("install.ROOT", Path(temp)), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(install.install_soundvolumeview())
        self.assertGreater(opener.call_args.kwargs["timeout"], 0)

    def test_downloaded_zip_is_removed_on_success_and_failure(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as bundle:
            bundle.writestr("SoundVolumeView.exe", b"exe")
        for body, success in ((data.getvalue(), True), (b"not-a-zip", False)):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                real_tempdir = tempfile.TemporaryDirectory
                with mock.patch("install.ROOT", root), mock.patch("install.ask_yes_no", return_value=True), \
                        mock.patch("install.urllib.request.urlopen", return_value=io.BytesIO(body)), \
                        mock.patch("install.tempfile.TemporaryDirectory", side_effect=lambda **kw: real_tempdir(dir=root, **kw)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(install.install_soundvolumeview(), success)
                self.assertFalse(list(root.glob("caption-install-*")))
                self.assertEqual((root / "tools" / "SoundVolumeView.exe").is_file(), success)


class KeyAndFlowTests(unittest.TestCase):
    @mock.patch("install.check.has_gemini_key", return_value=False)
    @mock.patch("install.offer_page")
    def test_key_trimmed_not_echoed_and_blank_skips(self, page, present):
        for key in ("  機密測試值  ", "   "):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                output = io.StringIO()
                with mock.patch("install.ROOT", root), mock.patch("install.getpass.getpass", return_value=key), contextlib.redirect_stdout(output):
                    self.assertTrue(install.setup_key())
                if key.strip():
                    self.assertEqual((root / "gemini_api_key.txt").read_text(encoding="utf-8"), key.strip())
                    self.assertNotIn(key.strip(), output.getvalue())
                else:
                    self.assertFalse((root / "gemini_api_key.txt").exists())

    @mock.patch("install.check.has_gemini_key", return_value=False)
    @mock.patch("install.offer_page")
    @mock.patch("install.getpass.getpass", side_effect=install.getpass.GetPassWarning)
    def test_unsafe_getpass_fallback_is_handled(self, read, page, present):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(install.setup_key())

    def test_failure_does_not_block_independent_steps(self):
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("install.sys.platform", "win32"))
            stack.enter_context(mock.patch("install.check.struct.calcsize", return_value=8))
            stack.enter_context(mock.patch("install.upgrade_pip", side_effect=PermissionError))
            actions = [stack.enter_context(mock.patch("install." + name, return_value=True)) for name in
                       ("install_torch", "install_requirements", "install_cublas", "install_soundvolumeview", "setup_cable", "setup_key")]
            final = stack.enter_context(mock.patch("install.check.run_checks", return_value=1))
            stack.enter_context(contextlib.redirect_stdout(output))
            self.assertEqual(install.main(), 1)
        for action in actions:
            action.assert_called_once()
        final.assert_called_once()
        self.assertIn("第 8/8 步", output.getvalue())

    @mock.patch("install.subprocess.run", return_value=SimpleNamespace(returncode=0))
    def test_pip_uses_list_and_live_output(self, run):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.run_pip("install", "--upgrade", "pip"))
        self.assertEqual(run.call_args.args[0][1:], ["-m", "pip", "install", "--upgrade", "pip"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertNotIn("capture_output", run.call_args.kwargs)
        self.assertNotIn("stdout", run.call_args.kwargs)

    def test_batch_file_is_ascii_and_keeps_exit_code(self):
        content = (install.ROOT / "install.bat").read_bytes().decode("ascii")
        self.assertIn('chcp 65001 >nul', content)
        self.assertIn('cd /d "%~dp0"', content)
        # 先找 python：啟動捷徑和 README 都用 python 執行，套件要裝在同一個 Python
        self.assertLess(content.index("python -c"), content.index("py -3 -c"))
        self.assertIn("Add python.exe to PATH", content)
        self.assertIn("exit /b %INSTALL_RESULT%", content)


if __name__ == "__main__":
    unittest.main()
