"""檢查安裝狀況；不連網，也不顯示金鑰內容。"""

import contextlib
import csv
import importlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parent
CABLE_HELP = ("到 https://vb-audio.com/Cable/ 下載並解壓縮，對 VBCABLE_Setup_x64.exe "
              "按右鍵 → 以系統管理員身分執行 → Install Driver → 重開機。")
SVV_HELP = ("重跑 install.bat，或到 https://www.nirsoft.net/utils/sound_volume_view.html "
            "下載 64 位元版 2.43 以上，把 SoundVolumeView.exe 放進 tools 資料夾。")


@dataclass
class Result:
    status: str
    item: str
    detail: str
    fix: str = ""


@dataclass
class GPU:
    name: str
    driver: str
    memory_mb: float


@dataclass
class TorchInfo:
    installed: bool = False
    version: str = ""
    cuda: bool = False
    cuda_build: str = ""


def mask_paths(text):
    """就算摘要被整段轉貼，也不留下使用者資料夾的完整路徑。"""
    text = str(text)
    for key in ("USERPROFILE", "HOME"):
        home = os.environ.get(key, "")
        if home and len(home) > 3:
            for spelling in (home, home.replace("\\", "/")):
                text = re.sub(re.escape(spelling), lambda _: "%USERPROFILE%", text,
                              flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]:[\\/]Users[\\/][^\\/\r\n\"'<>]+", "%USERPROFILE%", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"/(?:home|Users)/[^/\s\"'<>]+", "%USERPROFILE%", text)
    return re.sub(r"AIza[\w-]+", "[金鑰已隱藏]", text)


def parse_cuda_version(output):
    # 新版驅動（例如 617）的 nvidia-smi 改印成「CUDA UMD Version: 13.4」
    match = re.search(r"CUDA (?:UMD )?Version\s*:\s*(\d+)\.(\d+)", output)
    return tuple(map(int, match.groups())) if match else None


def parse_nvidia_smi(output):
    cards = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 3:
            continue
        match = re.search(r"[\d.]+", row[2])
        if match:
            try:
                cards.append(GPU(row[0].strip(), row[1].strip(), float(match[0])))
            except ValueError:
                pass
    return cards


def detect_nvidia():
    """驅動不在、查詢逾時或輸出不完整時，仍繼續檢查其他項目。"""
    outputs = []
    for args in (["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                 ["nvidia-smi"]):
        try:
            run = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=15)
            outputs.append(run.stdout if run.returncode == 0 else "")
        except (OSError, subprocess.SubprocessError):
            outputs.append("")
    return parse_nvidia_smi(outputs[0]), parse_cuda_version(outputs[1])


def probe_torch(fresh=False):
    if fresh:
        # pip 可能剛換過 torch；用新行程避免沿用記憶體裡的舊版。
        try:
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--torch-info"],
                                 capture_output=True, text=True, encoding="utf-8", timeout=60)
            return TorchInfo(**json.loads(run.stdout)) if run.returncode == 0 else TorchInfo()
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            return TorchInfo()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            torch = importlib.import_module("torch")
            return TorchInfo(True, str(torch.__version__), bool(torch.cuda.is_available()),
                             str(torch.version.cuda or ""))
    except Exception:
        return TorchInfo()


def needs_cublas12(torch):
    return "+cu13" in torch.version or torch.cuda_build.startswith("13.")


def package_dir(name):
    """用安裝紀錄找資料夾，不必先載入可能缺 DLL 的套件。"""
    try:
        path = Path(metadata.distribution(name).locate_file(name.replace("-", "_")))
        return path if path.is_dir() else None
    except Exception:
        return None


def find_cublas12(directory=None):
    directory = directory if directory is not None else package_dir("ctranslate2")
    places = ([directory] if directory else [])
    # python.exe 所在的資料夾一定在 DLL 搜尋路徑上，有些人把 DLL 放在那裡
    places += [Path(sys.executable).parent, Path(sys.base_prefix)]
    places += [Path(p.strip('"')) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for place in places:
        try:
            if (place / "cublas64_12.dll").is_file():
                return True
        except OSError:
            # PATH 可能含有目前帳號無權讀取的資料夾，繼續找下一個。
            continue
    return False


def parse_requirements(text):
    """只解讀套件名稱和單一 >=；其他寫法保守地只查有沒有裝。"""
    items = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)", line)
        if not match:
            continue
        simple = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\s*>=\s*(\d+(?:\.\d+)*(?:(?:a|b|rc|dev|post)\d*)?)", line)
        items.append((match[1], simple[1] if simple else None))
    return items


def version_key(version):
    """比較常見版本號，忽略 +cu130 等本機標記；正式版排在預覽版之後。"""
    match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:[.-]?(dev|a|b|rc|post)(\d*))?(?:\+[^\s]+)?", version)
    if not match:
        raise ValueError("看不懂這個版本號")
    release = tuple(map(int, match[1].split(".")))
    release = release + (0,) * max(0, 8 - len(release))
    rank = {"dev": -4, "a": -3, "b": -2, "rc": -1, None: 0, "post": 1}[match[2]]
    return release, rank, int(match[3] or 0)


def version_at_least(installed, minimum):
    return version_key(installed) >= version_key(minimum)


def check_requirements(path):
    results = []
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return [Result("❌", "requirements.txt", "讀不到套件清單", "重新下載並完整解壓縮程式，再執行 install.bat。")]
    for name, minimum in parse_requirements(text):
        fix = "重跑 install.bat，或執行 python -m pip install -r requirements.txt。"
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            results.append(Result("❌", name, "還沒安裝", fix))
            continue
        except Exception:
            results.append(Result("❌", name, "讀不到安裝版本", fix))
            continue
        try:
            enough = minimum is None or version_at_least(installed, minimum)
        except ValueError:
            results.append(Result("⚠️", name, "已安裝，但無法比較版本", fix))
            continue
        results.append(Result("✅" if enough else "❌", name,
                              f"{installed}" + (f"（需要 >= {minimum}）" if minimum else ""),
                              "" if enough else fix))
    return results


def has_gemini_key(root=ROOT):
    # 只回傳有沒有；內容不放進結果、不回顯，也不做網路驗證。
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return True
    try:
        return bool((Path(root) / "gemini_api_key.txt").read_bytes().strip())
    except OSError:
        return False


def detect_cable():
    """回傳找到的輸入裝置名稱，以及有沒有成功列出裝置。"""
    enumerated = False
    for name in ("pyaudiowpatch", "sounddevice"):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                module = importlib.import_module(name)
                if name == "pyaudiowpatch":
                    audio = module.PyAudio()
                    try:
                        devices = [audio.get_device_info_by_index(i) for i in range(audio.get_device_count())]
                    finally:
                        audio.terminate()
                    inputs = [d["name"] for d in devices if d.get("maxInputChannels", 0) > 0]
                else:
                    inputs = [d["name"] for d in module.query_devices() if d.get("max_input_channels", 0) > 0]
            enumerated = True
            cable = [n for n in inputs if "CABLE" in n.upper()]
            if cable:
                return cable, True
        except Exception:
            continue
        # 另一套音訊函式庫可能列得到不同的裝置，繼續試。
    return [], enumerated


def file_version(path):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            api = importlib.import_module("win32api")
            info = api.GetFileVersionInfo(str(path), "\\")
        high, low = info["FileVersionMS"], info["FileVersionLS"]
        return f"{high >> 16}.{high & 65535}.{low >> 16}.{low & 65535}"
    except Exception:
        return None


def nirsoft_version(file_ver):
    """NirSoft 網站寫的「2.53」在檔案版本裡是 2.5.3.0，換回網站上的寫法才能和 2.43 比"""
    try:
        major, minor, build = (int(part) for part in file_ver.split(".")[:3])
        return f"{major}.{minor}{build}"
    except Exception:
        return None


def generate_summary(results, windows, python, cards):
    lines = ["=== 可以整段複製貼給 AI 助手（開始） ===", "即時中日對照字幕：安裝自我檢查",
             f"系統：{windows}", f"Python：{python}",
             "顯卡：" + ("；".join(f"{c.name}，驅動 {c.driver}，{c.memory_mb:g} MiB" for c in cards) or "沒有偵測到 NVIDIA 顯卡")]
    for result in results:
        lines.append(f"{result.status} {result.item}：{result.detail}")
        if result.fix:
            lines.append(f"  怎麼補：{result.fix}")
    lines += ["請依上述結果提供下一步；不要要求提供 API 金鑰。", "=== 可以整段複製貼給 AI 助手（結束） ==="]
    return mask_paths("\n".join(lines))


def run_checks(root=ROOT):
    root = Path(root)
    results = []
    windows = f"{platform.system()} {platform.release()}（{platform.version()}）"
    os64 = platform.machine().lower() in ("amd64", "x86_64", "arm64", "aarch64") or bool(os.environ.get("PROCESSOR_ARCHITEW6432"))
    system_ok = platform.system() == "Windows" and os64
    results.append(Result("✅" if system_ok else "❌", "Windows 64 位元", windows,
                          "" if system_ok else "請改用 Windows 10/11 的 64 位元電腦。"))
    python = f"{platform.python_version()}，{struct.calcsize('P') * 8} 位元"
    python_ok = sys.version_info >= (3, 10) and struct.calcsize("P") == 8
    results.append(Result("✅" if python_ok else "❌", "Python", python,
                          "" if python_ok else "到 https://www.python.org/downloads/windows/ 安裝 64 位元 Python 3.10 以上，勾選 Add python.exe to PATH。"))
    cards, cuda = detect_nvidia()
    if not cards:
        results.append(Result("⚠️", "NVIDIA 顯卡", "沒有偵測到；會改用 CPU，語音辨識會非常慢",
                              "有 NVIDIA 顯卡的話，先更新官方驅動，再重跑 install.bat；沒有也能用 CPU 試跑。"))
    for i, card in enumerate(cards, 1):
        low = card.memory_mb < 8192
        results.append(Result("⚠️" if low else "✅", f"NVIDIA 顯卡 {i}",
                              f"{card.name}，驅動 {card.driver}，顯存 {card.memory_mb:g} MiB；驅動支援 CUDA " + (".".join(map(str, cuda)) if cuda else "未知"),
                              "顯存不到 8 GB，可能記憶體不足；可改用較小的 Whisper 模型，例如設定 WHISPER_MODEL=small。" if low else ""))
    torch = probe_torch()
    if not torch.installed:
        results.append(Result("❌", "PyTorch", "沒有安裝，或無法載入", "重跑 install.bat。"))
    else:
        bad = bool(cards) and not torch.cuda
        results.append(Result("❌" if bad else ("✅" if torch.cuda else "⚠️"), "PyTorch",
                              f"{torch.version}；CUDA 可用：{'是' if torch.cuda else '否（CPU 會非常慢）'}",
                              "重跑 install.bat；有 NVIDIA 卡仍無法使用 CUDA 時，先更新顯卡驅動。" if bad else ("" if torch.cuda else "想加速可使用支援的 NVIDIA 顯卡。")))
    for module_name in ("faster_whisper", "ctranslate2"):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                importlib.import_module(module_name)
            results.append(Result("✅", module_name, "可以載入"))
        except Exception:
            results.append(Result("❌", module_name, "沒有安裝，或無法載入", "重跑 install.bat；若提到 cuBLAS，照 README「安裝」第 1 步補裝 DLL。"))
    if needs_cublas12(torch):
        found = find_cublas12()
        results.append(Result("✅" if found else "❌", "CUDA 12 cuBLAS", "有找到 cublas64_12.dll" if found else "缺少 cublas64_12.dll",
                              "" if found else "重跑 install.bat；或照 README「安裝」第 1 步：python -m pip install nvidia-cublas-cu12，再把 cublas*64_12.dll 複製到 ctranslate2 套件資料夾。"))
    results.extend(check_requirements(root / "requirements.txt"))
    key = has_gemini_key(root)
    results.append(Result("✅" if key else "⚠️", "Gemini 金鑰", "有找到" if key else "沒找到",
                          "" if key else "重跑 install.bat，或到 https://aistudio.google.com/apikey 申請後存進 gemini_api_key.txt；只用本機模型可以忽略。"))
    cable, enumerated = detect_cable()
    results.append(Result("✅" if cable else "⚠️", "虛擬音訊線", "有找到 CABLE 輸入裝置" if cable else ("沒找到 CABLE 輸入裝置" if enumerated else "無法列出音訊裝置（缺套件或音效裝置未就緒）"), "" if cable else CABLE_HELP))
    exe = root / "tools" / "SoundVolumeView.exe"
    version = nirsoft_version(file_version(exe)) if exe.is_file() else None
    old = version is not None and not version_at_least(version, "2.43")
    results.append(Result("⚠️" if not exe.is_file() or old else "✅", "SoundVolumeView",
                          (f"已找到，版本 {version}" if version else "已找到（無法讀取版本，請確認為 2.43 以上）") if exe.is_file() else "沒找到 tools\\SoundVolumeView.exe",
                          SVV_HELP if not exe.is_file() or old else ""))
    for result in results:
        print(mask_paths(f"{result.status} {result.item}：{result.detail}"))
        if result.fix:
            print(mask_paths(f"  怎麼補：{result.fix}"))
    errors = sum(r.status == "❌" for r in results)
    warnings = sum(r.status == "⚠️" for r in results)
    print(f"\n總結：{errors} 項需要修正，{warnings} 項提醒。" + ("請依上面的「怎麼補」處理，再檢查一次。" if errors else "必要檢查通過，請留意提醒。"))
    print("\n" + generate_summary(results, windows, python, cards))
    return 1 if errors else 0


def configure_output():
    # 雙擊批次檔時，讓中文與狀態符號用 UTF-8 顯示。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    configure_output()
    if sys.argv[1:] == ["--torch-info"]:
        print(json.dumps(vars(probe_torch())))
    else:
        sys.exit(run_checks())
