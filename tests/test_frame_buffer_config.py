"""檢查畫面與聲音的保存秒數，不載入辨識模型或開啟裝置。"""

import ast
from pathlib import Path
import unittest


class FrameBufferConfigTests(unittest.TestCase):
    def test_video_margin_does_not_change_audio_retention(self):
        source = Path(__file__).resolve().parents[1] / "caption_app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        constants = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(constants["BUFFER_MARGIN_SEC"], 8.0)
        self.assertEqual(constants["FRAME_BUFFER_MARGIN_SEC"], 1.0)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "FrameBuffer"]
        self.assertEqual(len(calls), 1)
        seconds = next(kw.value for kw in calls[0].keywords if kw.arg == "max_seconds")
        self.assertEqual(ast.unparse(seconds), "DISPLAY_DELAY_SEC + FRAME_BUFFER_MARGIN_SEC")
        audio = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "AudioRingBuffer")
        self.assertEqual(next(kw.value.id for kw in audio.keywords if kw.arg == "max_seconds"), "buffer_seconds")
        assignment = next(node for node in main.body if isinstance(node, ast.Assign)
                          and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "buffer_seconds")
        self.assertEqual(ast.unparse(assignment.value), "DISPLAY_DELAY_SEC + BUFFER_MARGIN_SEC")


if __name__ == "__main__":
    unittest.main()
