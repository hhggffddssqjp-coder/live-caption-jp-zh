"""跟著畫面完成安裝；可重跑，不會自動安裝音訊驅動。"""

import getpass
from importlib import metadata
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import warnings
import zipfile

import check_setup as check


ROOT = Path(__file__).resolve().parent
# 從高到低比對「驅動支援的 CUDA」，不是電腦是否裝了 CUDA Toolkit。
# RTX 50 系列需要 CUDA 12.8 以上；驅動太舊時請先更新驅動。
CUDA_INDEXES = (
    ((13, 0), "https://download.pytorch.org/whl/cu130"),
    ((12, 8), "https://download.pytorch.org/whl/cu128"),
    ((12, 6), "https://download.pytorch.org/whl/cu126"),
)
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
SVV_URL = "https://www.nirsoft.net/utils/soundvolumeview-x64.zip"


def choose_torch_index(cuda):
    for minimum, url in CUDA_INDEXES:
        if cuda is not None and cuda >= minimum:
            return url
    return CPU_INDEX


def torch_is_ready(info, index):
    if not info.installed:
        return False
    if info.cuda:
        return True
    return index == CPU_INDEX and ("+cpu" in info.version or not info.cuda_build)


def requirements_are_ready(path):
    results = check.check_requirements(path)
    return bool(results) and all(r.status == "✅" for r in results)


def run_pip(*arguments):
    command = [sys.executable, "-m", "pip", *arguments]
    print("即將執行：" + check.mask_paths(subprocess.list2cmdline(command)), flush=True)
    # 不攔截輸出，下載進度和 pip 的錯誤會立刻顯示。
    try:
        result = subprocess.run(command, cwd=ROOT)
    except OSError:
        print("無法啟動 pip。請確認 Python 安裝完整，再重跑 install.bat。")
        return False
    if result.returncode:
        print("這次套件安裝沒有成功。請看上方 pip 的錯誤，確認網路與磁碟空間後重跑 install.bat。")
        return False
    return True


def ask_yes_no(prompt):
    try:
        return input(prompt + " (Y/n)：").strip().lower() in ("", "y", "yes")
    except (EOFError, OSError):
        print("目前無法輸入，先略過；之後可重跑 install.bat。")
        return False


def offer_page(url):
    if ask_yes_no("要開啟下載／申請網頁嗎？"):
        try:
            os.startfile(url)
        except (AttributeError, OSError):
            print("無法自動開網頁，請自己用瀏覽器開啟：" + url)


def upgrade_pip():
    print("pip 會自己檢查最新版；已是最新版時會略過下載和安裝。")
    return run_pip("install", "--upgrade", "pip")


def install_torch():
    cards, cuda = check.detect_nvidia()
    index = choose_torch_index(cuda if cards else None)
    if torch_is_ready(check.probe_torch(fresh=True), index):
        print("已裝好可用的 PyTorch，略過。")
        if index == CPU_INDEX:
            print("提醒：如果目前用 CPU，語音辨識會非常慢；有 NVIDIA 卡請更新驅動後重跑。")
        return True
    if index == CPU_INDEX:
        print("⚠️ 沒有偵測到支援 CUDA 12.6 以上的 NVIDIA 驅動，這次改裝 CPU 版，語音辨識會非常慢。")
        print("有 NVIDIA 顯卡的話，建議先更新官方驅動，再重跑 install.bat；RTX 50 系列需要 CUDA 12.8 以上。")
    else:
        print("依顯卡驅動選擇：" + index.rsplit("/", 1)[-1])
    # CPU 版和 CUDA 版的版本號可能一樣；強制替換才能確實換掉舊版。
    if not run_pip("install", "--upgrade", "--force-reinstall", "torch", "--index-url", index):
        return False
    info = check.probe_torch(fresh=True)
    if not torch_is_ready(info, index):
        print("PyTorch 已安裝，但仍無法使用預期的運算方式。請更新 NVIDIA 驅動，再重跑 install.bat。")
        return False
    return True


def install_requirements():
    path = ROOT / "requirements.txt"
    if requirements_are_ready(path):
        print("套件都已安裝，而且版本符合 requirements.txt，略過。")
        return True
    return run_pip("install", "-r", str(path))


def cublas_copy_complete(destination):
    return destination is not None and all((Path(destination) / name).is_file()
                                          for name in ("cublas64_12.dll", "cublasLt64_12.dll"))


def copy_cublas_dlls(source, destination):
    """只複製 README 指定的 CUDA 12 cuBLAS DLL。"""
    files = list(Path(source).glob("cublas*64_12.dll"))
    names = {p.name.lower() for p in files}
    if not {"cublas64_12.dll", "cublaslt64_12.dll"}.issubset(names):
        raise FileNotFoundError("CUDA 12 cuBLAS 的 DLL 不齊全")
    for path in files:
        target = Path(destination) / path.name
        if not target.is_file():
            shutil.copy2(path, target)


def cublas_source():
    try:
        return Path(metadata.distribution("nvidia-cublas-cu12").locate_file("nvidia/cublas/bin"))
    except metadata.PackageNotFoundError:
        return None


def install_cublas():
    info = check.probe_torch(fresh=True)
    if not info.installed:
        print("目前無法載入 PyTorch，先略過相依的 DLL 步驟；修好 PyTorch 後重跑 install.bat。")
        return False
    if not check.needs_cublas12(info):
        print("目前不是 CUDA 13 版 PyTorch，不需要這個補裝步驟，略過。")
        return True
    destination = check.package_dir("ctranslate2")
    if destination is None:
        print("還沒裝好 ctranslate2，無法複製 DLL。修好套件安裝後重跑 install.bat。")
        return False
    if cublas_copy_complete(destination):
        print("ctranslate2 資料夾已經有完整的 CUDA 12 cuBLAS DLL，略過。")
        return True
    source = cublas_source()
    if source is None or not cublas_copy_complete(source):
        if not run_pip("install", "nvidia-cublas-cu12"):
            return False
        source = cublas_source()
    if source is None:
        print("找不到 CUDA 12 cuBLAS 套件，請照 README「安裝」第 1 步手動補裝。")
        return False
    copy_cublas_dlls(source, destination)
    print("已把 CUDA 12 cuBLAS DLL 複製到 ctranslate2 資料夾。")
    return cublas_copy_complete(destination)


def extract_soundvolumeview(archive, tools):
    """先檢查整包路徑，再只取指定檔案，不執行整包解壓縮。"""
    with zipfile.ZipFile(archive) as bundle:
        candidates = []
        for member in bundle.infolist():
            name = member.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if (path.is_absolute() or PureWindowsPath(name).drive or ".." in path.parts
                    or ":" in name or "\x00" in name
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("壓縮檔含有不安全的路徑，已取消解壓縮")
            if path.name.lower() == "soundvolumeview.exe" and not member.is_dir():
                candidates.append(member)
        if len(candidates) != 1 or not 0 < candidates[0].file_size <= 20 * 1024 * 1024:
            raise ValueError("壓縮檔沒有唯一且大小合理的 SoundVolumeView.exe")
        content = bundle.read(candidates[0])
    destination = Path(tools) / "SoundVolumeView.exe"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def install_soundvolumeview():
    if (ROOT / "tools" / "SoundVolumeView.exe").is_file():
        print("已找到 tools\\SoundVolumeView.exe，略過下載；最後檢查會確認版本。")
        return True
    if not ask_yes_no("要自動從 NirSoft 官網下載 SoundVolumeView 嗎？"):
        print(check.SVV_HELP)
        return False
    try:
        # 暫存 ZIP 不論成功、失敗都會刪掉，只保留指定的執行檔。
        with tempfile.TemporaryDirectory(prefix="caption-install-") as temp:
            archive = Path(temp) / "soundvolumeview.zip"
            with urllib.request.urlopen(SVV_URL, timeout=30) as response, archive.open("wb") as output:
                total = 0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 20 * 1024 * 1024:
                        raise ValueError("下載檔案大小異常")
                    output.write(chunk)
            extract_soundvolumeview(archive, ROOT / "tools")
        print("已放好 tools\\SoundVolumeView.exe。")
        return True
    except Exception:
        print("下載或解壓縮失敗，可能是網路、網站回應或資料夾寫入權限的問題。")
        print(check.SVV_HELP)
        return False


def setup_cable():
    cable, _ = check.detect_cable()
    if cable:
        print("已找到 CABLE 輸入裝置，略過。")
        return True
    print("虛擬音訊線讓程式只錄瀏覽器的聲音，避免原音和延遲聲音疊在一起或形成回音。")
    print(check.CABLE_HELP)
    print("這裡只會開網頁，請自行安裝驅動；如果剛裝完，請重開機後再檢查。")
    offer_page("https://vb-audio.com/Cable/")
    return False


def setup_key():
    if check.has_gemini_key(ROOT):
        print("有找到 Gemini 金鑰，略過；不顯示內容、不做網路驗證。")
        return True
    print("到 https://aistudio.google.com/apikey 可免費申請 Gemini API 金鑰，API 有免費額度限制。")
    print("只用本機模型可以略過；請不要把金鑰貼給別人或 AI 助手。")
    offer_page("https://aistudio.google.com/apikey")
    try:
        with warnings.catch_warnings():
            # 若終端機無法隱藏輸入，就略過，避免 getpass 退回會回顯的模式。
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("請貼上金鑰（不顯示內容；直接按 Enter 略過）：").strip()
    except (EOFError, OSError, getpass.GetPassWarning):
        print("目前無法安全輸入金鑰，先略過。之後可自行存進程式資料夾的 gemini_api_key.txt。")
        return False
    if not key:
        print("已略過金鑰設定；使用本機模型不需要金鑰。")
        return True
    (ROOT / "gemini_api_key.txt").write_text(key, encoding="utf-8")
    print("金鑰已存好，不會顯示內容或做網路驗證。")
    return True


def main():
    check.configure_output()
    if sys.version_info < (3, 10) or check.struct.calcsize("P") != 8 or sys.platform != "win32":
        print("請使用 Windows 10/11 和 64 位元 Python 3.10 以上。")
        print("到 https://www.python.org/downloads/windows/ 安裝，記得勾選 Add python.exe to PATH，再重跑 install.bat。")
        return 1
    print("安裝會下載 Python 套件，可能需要幾分鐘；隨時可以關閉，之後重跑 install.bat 接著處理。")
    steps = [("更新 pip", upgrade_pip), ("選擇並安裝 PyTorch", install_torch),
             ("安裝 requirements.txt 裡的套件", install_requirements),
             ("補上 CUDA 13 所需的 CUDA 12 cuBLAS DLL", install_cublas),
             ("準備 SoundVolumeView", install_soundvolumeview),
             ("確認虛擬音訊線 VB-Cable", setup_cable), ("設定 Gemini 金鑰", setup_key)]
    pending = []
    for number, (title, action) in enumerate(steps, 1):
        print(f"\n第 {number}/8 步：{title}", flush=True)
        try:
            if not action():
                pending.append(title)
        except Exception as error:
            reason = "資料夾沒有寫入權限" if isinstance(error, PermissionError) else "檔案或套件尚未準備好"
            print(f"這一步沒有完成：{reason}。請確認資料夾可寫入，並參考 README「安裝」；修好後重跑 install.bat。")
            pending.append(title)
    print("\n第 8/8 步：執行完整自我檢查")
    exit_code = check.run_checks(ROOT)
    print("\n安裝流程已走完。" + ("仍需處理或確認：" + "、".join(pending) if pending else "各安裝步驟已完成。"))
    print("VB-Cable 剛裝完一定要重開機；重開機後可用 python check_setup.py 再確認。")
    print("檢查沒有 ❌ 後，在程式資料夾執行：python live_caption_gemini.py")
    print("有問題時，把上面的自我檢查摘要整段貼給 AI 助手；不要貼金鑰。")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
