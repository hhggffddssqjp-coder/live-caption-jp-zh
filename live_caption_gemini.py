"""輕量啟動入口；Windows spawn 重跑此檔時不載入字幕程式的重型相依套件。"""

import sys


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # 打包成執行檔時，先處理子行程再載入主程式
    from caption_app import _run_cli

    sys.exit(_run_cli(sys.argv[1:]))
elif __name__ != "__mp_main__":
    # 一般 import 沿用同一個模組物件，包含私有名稱與屬性 patch 的行為。
    # 不能只複製名稱，否則函式的全域變數仍指向另一份命名空間。
    import caption_app

    sys.modules[__name__] = caption_app
