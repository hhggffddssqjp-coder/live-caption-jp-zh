"""Integration tests for the translation backends inside the real application modules.

Windows, audio, GPU and GUI modules are replaced in ``sys.modules`` before import; the
translation code under test (prompts, batching, retries, output files) is the real code.
"""

import importlib
import json
import os
import queue
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

from local_llm import MODE_INSTRUCT, MODE_RIVA, LocalLLMError
from local_translation import LocalTranslationEngine


class FakeSegment:
	def __init__(self, index, text=None):
		self.start = index + 0.25
		self.end = index + 0.75
		self.text = f" {text if text is not None else f'日文{index + 1}です。'} "


class ScriptedClient:
	def __init__(self, handler, mode=MODE_INSTRUCT):
		self.handler = handler
		self.mode = mode
		self.server_info = {}
		self.model = None
		self.base_url = "http://127.0.0.1:8766"
		self.timeout = 120.0
		self.max_tokens = 256
		self.calls = []

	def ensure_ready(self, *_, **__):
		pass

	def chat(self, system_prompt, user_prompt, *, max_tokens=None, timeout=None, temperature=None):
		self.calls.append((system_prompt, user_prompt))
		reply = self.handler(system_prompt, user_prompt)
		if isinstance(reply, BaseException):
			raise reply
		return reply


def bilingual_reply(contents, tag, keep=None):
	"""A well-formed rebuild reply for the batch in ``contents``: one "日文\\n中文" block per
	line. ``keep`` limits how many lines are answered, to simulate an incomplete reply."""
	batch = contents.split("請處理這一批句子：\n", 1)[1].splitlines()
	batch = batch if keep is None else batch[:keep]
	return "\n\n".join(f"{ja}\n{tag}譯文{k}" for k, ja in enumerate(batch))


class _GeminiCalls(list):
	"""generate_content calls; per-request timeouts are kept apart in .timeouts_ms so the
	exact-match assertions on prompts and parameters stay unchanged."""

	def __init__(self):
		super().__init__()
		self.timeouts_ms = []


def install_fake_gemini(reply):
	"""Install a google.genai stand-in; returns the list of generate_content calls."""
	calls = _GeminiCalls()

	class HttpOptions:
		def __init__(self, timeout=None):
			self.timeout = timeout

	class GenerateContentConfig:
		def __init__(self, **kwargs):
			self.kwargs = kwargs

	class Models:
		def generate_content(self, model, contents, config):
			kwargs = dict(config.kwargs)
			http_options = kwargs.pop("http_options", None)
			calls.timeouts_ms.append(getattr(http_options, "timeout", None))
			calls.append({"model": model, "contents": contents, **kwargs})
			result = reply(contents, len(calls))
			if isinstance(result, BaseException):
				raise result
			if isinstance(result, tuple):  # (text, finish_reason)
				text, finish_reason = result
				return types.SimpleNamespace(text=text, candidates=[types.SimpleNamespace(finish_reason=finish_reason)])
			return types.SimpleNamespace(text=result)

		def list(self):
			return []

	class Client:
		def __init__(self, api_key):
			self.api_key = api_key
			self.models = Models()

	google = types.ModuleType("google")
	genai = types.ModuleType("google.genai")
	genai_types = types.ModuleType("google.genai.types")
	genai_errors = types.ModuleType("google.genai.errors")
	genai.Client = Client
	genai_types.GenerateContentConfig = GenerateContentConfig
	genai_types.HttpOptions = HttpOptions
	genai.types, genai.errors, google.genai = genai_types, genai_errors, genai
	sys.modules.update({"google": google, "google.genai": genai, "google.genai.types": genai_types, "google.genai.errors": genai_errors})
	return calls


def import_application():
	numpy = types.ModuleType("numpy")
	numpy.ndarray = object
	numpy.float32 = float
	torch = types.ModuleType("torch")
	torch.cuda = types.SimpleNamespace(is_available=lambda: False)
	scipy = types.ModuleType("scipy")
	scipy_signal = types.ModuleType("scipy.signal")
	scipy_signal.resample_poly = mock.Mock(name="resample_poly")
	pil = types.ModuleType("PIL")
	pil.Image = mock.Mock(name="Image")
	pil.ImageTk = mock.Mock(name="ImageTk")
	faster_whisper = types.ModuleType("faster_whisper")
	faster_whisper.WhisperModel = mock.Mock(name="WhisperModel")
	silero_vad = types.ModuleType("silero_vad")
	silero_vad.load_silero_vad = mock.Mock(name="load_silero_vad")
	silero_vad.VADIterator = mock.Mock(name="VADIterator")
	tkinter = mock.MagicMock(name="tkinter")
	sys.modules.update({
		"numpy": numpy, "torch": torch, "scipy": scipy,
		"scipy.signal": scipy_signal, "cv2": types.ModuleType("cv2"),
		"PIL": pil, "faster_whisper": faster_whisper,
		"silero_vad": silero_vad, "pyaudiowpatch": types.ModuleType("pyaudiowpatch"),
		"sounddevice": types.ModuleType("sounddevice"),
		"tkinter": tkinter, "tkinter.scrolledtext": tkinter.scrolledtext,
	})
	sys.modules.pop("live_caption_gemini", None)
	return importlib.import_module("live_caption_gemini")


MASTER_LIVE_SYSTEM_PROMPT = (
	"你是專業的日文-繁體中文口譯，正在幫忙即時翻譯一個日文廣播/網路節目的口語對話。"
	"內容常有語助詞、停頓、話講到一半重講、省略主詞這類真人講話的狀況，"
	"請翻成自然通順、口語化的繁體中文，像是真人在說話，不要翻得死板生硬、也不要逐字直譯。"
	"只要輸出翻譯結果本身，不要加任何說明、引號、備註、拼音或原文。"
)


class TranslationBackendTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.app = import_application()

	def setUp(self):
		self.environment = mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
		self.environment.start()
		self.addCleanup(self.environment.stop)
		self.app.genai = self.app.genai_types = self.app.genai_errors = None
		self.sleeps = []
		sleeper = mock.patch.object(self.app.time, "sleep", side_effect=self.sleeps.append)
		sleeper.start()
		self.addCleanup(sleeper.stop)
		printer = mock.patch("builtins.print")
		self.printed = printer.start()
		self.addCleanup(printer.stop)

	def printed_text(self):
		return "\n".join(" ".join(str(arg) for arg in call.args) for call in self.printed.call_args_list)

	def gemini_translator(self, reply):
		calls = install_fake_gemini(reply)
		return self.app.Translator("gemini"), calls

	def local_translator(self, handler, mode=MODE_INSTRUCT):
		warmups = ("請翻譯這一句：\nこんにちは。", "Translate this into Traditional Chinese: こんにちは。")
		client = ScriptedClient(lambda system, user: "你好。" if user in warmups else handler(system, user), mode=mode)

		def factory(**kwargs):
			return LocalTranslationEngine(
				client, live_timeout=15, batch_lines=20, startup_wait=0,
				sleep=self.sleeps.append, log=lambda *_: None, **kwargs,
			)

		# Hermetic regardless of what's on the developer's real machine (LOCAL_LLM_* env,
		# a real local_llm_server.json): never let auto-start actually run in this helper.
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", return_value=None):
			translator = self.app.Translator("local")
		client.calls.clear()
		return translator, client

	def run_reconstruction(self, translator, lines):
		asr = mock.Mock()
		asr.transcribe.return_value = ([FakeSegment(i, text) for i, text in enumerate(lines)], None)
		temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(temp_dir.cleanup)
		wav_path = Path(temp_dir.name) / "transcript_20260926_010203_audio.wav"
		wav_path.write_bytes(b"RIFF")
		out_path = self.app.rebuild_transcript_from_full_audio(wav_path, asr, translator, {"田中": "田中"}, "節目")
		return out_path, wav_path



class GeminiPathTests(TranslationBackendTestCase):
	def test_default_backend_is_gemini_with_master_prompts_and_config(self):
		calls = install_fake_gemini(lambda contents, n: "  請多指教。  ")
		translator = self.app.Translator()
		self.assertEqual(translator.backend, "gemini")
		self.assertEqual(translator.translate_with_context("田中です。", "よろしく。", {"田中": "田中"}), "請多指教。")
		self.assertEqual(calls, [{
			"model": self.app.GEMINI_MODEL,
			"contents": "前一句話（僅供參考上下文，不用翻譯）：\n田中です。\n\n請翻譯這一句：\nよろしく。",
			"system_instruction": MASTER_LIVE_SYSTEM_PROMPT + "\n" + self.app.glossary_hint({"田中": "田中"}),
			"temperature": 0.3,
		}])

	def test_gemini_rate_limit_skips_sentence_without_sleeping(self):
		translator, _ = self.gemini_translator(lambda contents, n: RuntimeError("429 RESOURCE_EXHAUSTED"))
		self.assertEqual(translator.translate_with_context("", "よろしく。"), "")
		self.assertEqual(self.sleeps, [])
		self.assertIn("免費額度", self.printed_text())

	def test_gemini_timeout_skips_the_sentence_without_raising(self):
		translator, calls = self.gemini_translator(lambda contents, n: TimeoutError("The read operation timed out"))
		self.assertEqual(translator.translate_with_context("", "よろしく。"), "")
		self.assertEqual(self.sleeps, [])
		self.assertIn("沒有回應", self.printed_text())
		self.assertEqual(calls.timeouts_ms, [int(self.app.GEMINI_LIVE_TIMEOUT_SEC * 1000)])

	def test_live_captions_and_rebuild_use_their_own_timeouts(self):
		translator, calls = self.gemini_translator(
			lambda contents, n: "中文" if n == 1 else bilingual_reply(contents, "x")
		)
		translator.translate_with_context("", "よろしく。")
		self.run_reconstruction(translator, ["日文です。"])
		self.assertEqual(calls.timeouts_ms, [15000, 120000])

	def test_gemini_echoing_japanese_is_retried_once_without_context(self):
		source = "スタッフさんがペコリとしてくださっております"
		translator, calls = self.gemini_translator(lambda contents, n: source if n == 1 else "工作人員向我們鞠躬了")
		self.assertEqual(translator.translate_with_context("こんばんは", source), "工作人員向我們鞠躬了")
		self.assertEqual(len(calls), 2)
		self.assertEqual(calls[1]["contents"], "請把這一句翻成繁體中文：\n" + source)

	def test_gemini_keeps_first_reply_when_retry_is_still_japanese(self):
		source = "スタッフさんがペコリとしてくださっております"
		translator, calls = self.gemini_translator(lambda contents, n: source)
		self.assertEqual(translator.translate_with_context("", source), source)
		self.assertEqual(len(calls), 2)

	def test_gemini_chinese_reply_is_not_retried(self):
		translator, calls = self.gemini_translator(lambda contents, n: "大家晚安")
		self.assertEqual(translator.translate_with_context("", "こんばんは"), "大家晚安")
		self.assertEqual(len(calls), 1)

	def test_missing_gemini_sdk_explains_how_to_fix_it(self):
		with mock.patch.dict(sys.modules, {"google": None, "google.genai": None}):
			with self.assertRaisesRegex(RuntimeError, "pip install google-genai"):
				self.app.Translator("gemini")

	def test_missing_gemini_key_still_fails_clearly(self):
		install_fake_gemini(lambda contents, n: "")
		os.environ.pop("GEMINI_API_KEY")
		with mock.patch.object(self.app, "GEMINI_API_KEY_FILE", Path("/nonexistent/gemini_api_key.txt")):
			with self.assertRaisesRegex(RuntimeError, "Gemini API 金鑰"):
				self.app.Translator("gemini")

	def test_gemini_reconstruction_keeps_master_batches_retries_and_pacing(self):
		lines = [f"日文{i}です。" for i in range(61)]
		translator, calls = self.gemini_translator(
			lambda contents, n: "" if n == 1 else bilingual_reply(contents, f"批次{n}")
		)
		out_path, wav_path = self.run_reconstruction(translator, lines)
		self.assertEqual([len(call["contents"].split("請處理這一批句子：\n", 1)[1].splitlines()) for call in calls], [60, 60, 1])
		self.assertEqual(calls[0]["contents"], "請處理這一批句子：\n" + "\n".join(lines[:60]))
		self.assertTrue(calls[2]["contents"].startswith("（前面幾句當上下文參考，不用重複翻譯：\n日文58です。\n日文59です。\n\n"))
		self.assertEqual(self.sleeps, [20, 4.5])
		text = out_path.read_text(encoding="utf-8")
		self.assertIn("翻譯模型看過完整上下文潤過的版本", text)
		self.assertIn("日文59です。\n批次2譯文59", text)
		self.assertTrue(text.endswith("日文60です。\n批次3譯文0\n"))
		self.assertFalse(wav_path.exists())

	def test_gemini_reconstruction_keeps_wav_when_a_batch_fails_all_retries(self):
		lines = [f"日文{i}です。" for i in range(61)]
		translator, calls = self.gemini_translator(
			lambda contents, n: "" if n <= 3 else bilingual_reply(contents, f"批次{n}")
		)
		out_path, wav_path = self.run_reconstruction(translator, lines)
		self.assertEqual(len(calls), 4)
		text = out_path.read_text(encoding="utf-8")
		self.assertIn("有 1 批整理失敗", text)
		self.assertTrue(text.endswith("日文60です。\n批次4譯文0\n"))
		self.assertTrue(wav_path.exists())
		self.assertIn("已保留完整錄音", self.printed_text())

	def test_gemini_reply_covering_only_half_the_batch_keeps_the_wav(self):
		lines = [f"日文{i}です。" for i in range(20)]
		translator, calls = self.gemini_translator(lambda contents, n: bilingual_reply(contents, "半", keep=10))
		out_path, wav_path = self.run_reconstruction(translator, lines)
		self.assertEqual(len(calls), 3)  # incomplete replies are retried like empty ones
		self.assertEqual(self.sleeps, [20, 20, 20])
		text = out_path.read_text(encoding="utf-8")
		self.assertIn("有 1 批整理失敗", text)
		self.assertIn("日文9です。\n半譯文9", text)  # the partial result is still kept in the transcript
		self.assertTrue(wav_path.exists())

	def test_gemini_reply_cut_off_by_the_output_limit_keeps_the_wav(self):
		lines = [f"日文{i}です。" for i in range(4)]
		translator, calls = self.gemini_translator(
			lambda contents, n: (bilingual_reply(contents, "截"), types.SimpleNamespace(name="MAX_TOKENS"))
		)
		out_path, wav_path = self.run_reconstruction(translator, lines)
		self.assertEqual(len(calls), 3)
		self.assertIn("截斷", self.printed_text())
		self.assertTrue(wav_path.exists())

	def test_complete_gemini_reply_still_deletes_the_wav(self):
		lines = [f"日文{i}です。" for i in range(4)]
		translator, calls = self.gemini_translator(
			lambda contents, n: (bilingual_reply(contents, "全"), types.SimpleNamespace(name="STOP"))
		)
		out_path, wav_path = self.run_reconstruction(translator, lines)
		self.assertEqual(len(calls), 1)
		self.assertFalse(wav_path.exists())

	def test_gemini_reply_missing_a_quarter_passes_the_batch_check_but_keeps_the_wav(self):
		# 45 of 60 lines is enough for the per-batch retry check (75%), so it is not retried,
		# but deleting the only recording needs (almost) every line
		lines = [f"日文{i}です。" for i in range(60)]
		translator, calls = self.gemini_translator(lambda contents, n: bilingual_reply(contents, "缺", keep=45))
		out_path, wav_path = self.run_reconstruction(translator, lines)
		self.assertEqual(len(calls), 1)
		self.assertTrue(wav_path.exists())
		self.assertIn("少了 15 句", out_path.read_text(encoding="utf-8"))

	def test_rebuild_coverage_counts_missing_placeholder_and_problem_lines(self):
		lines = [f"日文{i}です。" for i in range(5)]
		parts = [
			"日文0です。\n譯文0\n\n日文1です。\n（翻譯失敗）",
			"日文2です。\nこれは翻訳されていないです",  # still Japanese: a translation problem
		]
		self.assertEqual(self.app._rebuild_coverage(parts, lines, None), (2, 1, 1))

	def test_join_thread_reports_whether_the_thread_really_finished(self):
		import threading
		import time
		self.assertTrue(self.app._join_thread(threading.Thread(target=lambda: None), 0.1))  # never started
		done = threading.Thread(target=lambda: None)
		done.start()
		self.assertTrue(self.app._join_thread(done, 1.0))
		release = threading.Event()
		stuck = threading.Thread(target=release.wait, daemon=True)
		stuck.start()
		started = time.perf_counter()
		self.assertFalse(self.app._join_thread(stuck, 0.2))  # timed out: must not pretend it ended
		self.assertLess(time.perf_counter() - started, 1.0)
		release.set()
		stuck.join(timeout=1)

	def test_worker_and_capture_do_not_shadow_thread_internals(self):
		# Python 3.10-3.12 call Thread._stop() while joining; an Event stored there breaks join()
		self.assertFalse(hasattr(self.app.Worker, "_stop") and not callable(getattr(self.app.Worker, "_stop")))
		source = Path(self.app.__file__).read_text(encoding="utf-8")
		self.assertNotIn("self._stop = threading.Event()", source)

	def test_rebuild_is_complete_allows_a_few_merged_lines_but_no_placeholders(self):
		self.assertTrue(self.app._rebuild_is_complete(missing=3, placeholders=0, problems=0, total=60))
		self.assertFalse(self.app._rebuild_is_complete(missing=4, placeholders=0, problems=0, total=60))
		self.assertFalse(self.app._rebuild_is_complete(missing=0, placeholders=1, problems=0, total=60))
		self.assertFalse(self.app._rebuild_is_complete(missing=0, placeholders=0, problems=4, total=60))
		self.assertTrue(self.app._rebuild_is_complete(missing=1, placeholders=0, problems=1, total=5))



class SparseRebuildSpeechTests(TranslationBackendTestCase):
	"""辨識到的語音幾乎是空白時，不能刪掉唯一的錄音。"""

	def test_short_recording_does_not_trigger(self):
		for duration in (0.0, 60.0, 119.99):
			with self.subTest(duration=duration):
				self.assertFalse(self.app._rebuild_speech_too_sparse(0.0, duration))

	def test_minimum_recording_duration_is_inclusive(self):
		self.assertTrue(self.app._rebuild_speech_too_sparse(0.0, 120.0))

	def test_speech_ratio_must_be_strictly_below_ten_percent(self):
		self.assertTrue(self.app._rebuild_speech_too_sparse(11.99, 120.0))
		self.assertFalse(self.app._rebuild_speech_too_sparse(12.0, 120.0))
		self.assertFalse(self.app._rebuild_speech_too_sparse(12.01, 120.0))

	def test_unknown_recording_duration_does_not_trigger(self):
		for duration in (None, float("nan"), float("inf")):
			with self.subTest(duration=duration):
				self.assertFalse(self.app._rebuild_speech_too_sparse(0.0, duration))

	def run_wav_reconstruction(self, segments, duration=180.0, asmr=False, read_error=None):
		# 使用真正的 WAV 標頭，不依賴被換成假模組的 numpy、torch 或 Google 套件。
		temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(temp_dir.cleanup)
		wav_path = Path(temp_dir.name) / "transcript_20261023_010203_audio.wav"
		with wave.open(str(wav_path), "wb") as recording:
			recording.setnchannels(1)
			recording.setsampwidth(2)
			recording.setframerate(16000)
			recording.writeframes(b"\0\0" * int(duration * 16000))
		asr = mock.Mock()
		asr.transcribe.return_value = (iter(segments), None)
		translator, calls = self.gemini_translator(lambda contents, n: bilingual_reply(contents, "中文"))
		with mock.patch.object(self.app, "ASMR_MODE", asmr):
			if read_error is None:
				out_path = self.app.rebuild_transcript_from_full_audio(wav_path, asr, translator, None)
			else:
				with mock.patch.object(self.app.wave, "open", side_effect=read_error):
					out_path = self.app.rebuild_transcript_from_full_audio(wav_path, asr, translator, None)
		self.assertEqual(asr.transcribe.call_args.kwargs["vad_filter"], not asmr)
		return out_path, wav_path, calls

	def test_sparse_speech_keeps_recording_and_explains_how_to_retry(self):
		for asmr in (False, True):
			with self.subTest(asmr=asmr):
				out_path, wav_path, _ = self.run_wav_reconstruction([FakeSegment(0)], asmr=asmr)
				self.assertTrue(wav_path.exists())
				text = out_path.read_text(encoding="utf-8")
				self.assertIn("辨識出的語音只有 0.0 分鐘，錄音有 3.0 分鐘", text)
				self.assertIn("可以用氣音模式重新處理", text)
				self.assertIn("--rebuild", text)
				self.assertNotIn(str(wav_path.parent), text)
				self.assertIn("可以用氣音模式重新處理", self.printed_text())
				self.assertIn(self.app._rebuild_command(wav_path), self.printed_text())

	def test_normal_speech_deletes_recording_even_with_long_pauses(self):
		# 只講了整段約 13%，仍算正常；加總兩句才會超過門檻。
		segments = [FakeSegment(0), FakeSegment(1)]
		segments[0].start, segments[0].end = 30.0, 42.0
		segments[1].start, segments[1].end = 130.0, 142.0
		out_path, wav_path, _ = self.run_wav_reconstruction(segments)
		self.assertTrue(out_path.exists())
		self.assertFalse(wav_path.exists())
		self.assertNotIn("這份整理稿可能有缺漏", out_path.read_text(encoding="utf-8"))

	def test_filtered_segments_do_not_count_as_recognized_speech(self):
		segments = [FakeSegment(0), FakeSegment(1, ""), FakeSegment(2, next(iter(self.app._HALLUCINATION_DENYLIST)))]
		for segment in segments[1:]:
			segment.start, segment.end = 10.0, 100.0
		out_path, wav_path, _ = self.run_wav_reconstruction(segments)
		self.assertTrue(wav_path.exists())
		self.assertIn("辨識出的語音只有 0.0 分鐘", out_path.read_text(encoding="utf-8"))

	def test_no_recognized_lines_keeps_recording_without_translation(self):
		out_path, wav_path, calls = self.run_wav_reconstruction([FakeSegment(0, "")])
		self.assertIsNone(out_path)
		self.assertTrue(wav_path.exists())
		self.assertEqual(len(calls), 0)

	def test_unreadable_duration_uses_existing_completeness_checks(self):
		for error in (OSError("無法讀取"), EOFError(), wave.Error("不完整的 WAV")):
			with self.subTest(error=type(error).__name__):
				out_path, wav_path, _ = self.run_wav_reconstruction([FakeSegment(0)], read_error=error)
				self.assertTrue(out_path.exists())
				self.assertFalse(wav_path.exists())


class PromptParityTests(TranslationBackendTestCase):
	"""The local backend must send exactly what the Gemini backend sends."""

	def test_live_caption_prompts_are_identical(self):
		translator, calls = self.gemini_translator(lambda contents, n: "請多指教。")
		local, client = self.local_translator(lambda system, user: "請多指教。")
		glossary = {"羊宮妃那": "羊宮妃那", "ゆみやひな": "羊宮妃那"}
		for prev_ja in ("", "田中です。"):
			translator.translate_with_context(prev_ja, "よろしくお願いします。", glossary)
			local.translate_with_context(prev_ja, "よろしくお願いします。", glossary)
		self.assertEqual([(call["system_instruction"], call["contents"]) for call in calls], client.calls)

	def test_reconstruction_prompts_are_identical(self):
		lines = [f"日文{i}です。" for i in range(12)]
		reply = "\n\n".join(f"{line}\n中文{i}。" for i, line in enumerate(lines))
		translator, calls = self.gemini_translator(lambda contents, n: reply)
		self.run_reconstruction(translator, lines)
		local, client = self.local_translator(lambda system, user: reply)
		self.run_reconstruction(local, lines)
		self.assertEqual([(call["system_instruction"], call["contents"]) for call in calls], client.calls)



class LocalBackendTests(TranslationBackendTestCase):
	def test_local_backend_does_not_import_the_google_sdk(self):
		with mock.patch.object(self.app, "_import_gemini_sdk", side_effect=AssertionError("Gemini SDK imported")):
			translator, _ = self.local_translator(lambda system, user: "你好。")
		self.assertEqual(translator.backend, "local")
		self.assertIsNone(self.app.genai)

	def test_environment_selects_backend_and_rejects_unknown_names(self):
		os.environ["TRANSLATION_BACKEND"] = " LOCAL "
		self.assertEqual(self.app.resolve_translation_backend(os.environ["TRANSLATION_BACKEND"]), "local")
		with self.assertRaisesRegex(RuntimeError, "gemini 或 local"):
			self.app.resolve_translation_backend("ollama")
		self.assertEqual(self.app.resolve_translation_backend(None), "gemini")

	def test_unreachable_local_service_fails_with_guidance(self):
		client = ScriptedClient(lambda s, u: "")
		client.ensure_ready = mock.Mock(side_effect=LocalLLMError("本機模型連線失敗", retryable=True, kind="connection"))
		factory = lambda **kwargs: LocalTranslationEngine(client, startup_wait=0, log=lambda *_: None, **kwargs)
		# Patched so this test's outcome never depends on a developer's own untracked
		# local_llm_server.json / LOCAL_LLM_* env vars actually present on this machine.
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", return_value=None):
			with self.assertRaisesRegex(RuntimeError, "llama-server[\\s\\S]*Gemini API"):
				self.app.Translator("local")

	def test_unreachable_local_service_without_model_path_gets_autostart_hint(self):
		client = ScriptedClient(lambda s, u: "")
		client.ensure_ready = mock.Mock(side_effect=LocalLLMError("本機模型連線失敗", retryable=True, kind="connection"))
		factory = lambda **kwargs: LocalTranslationEngine(client, startup_wait=0, log=lambda *_: None, **kwargs)
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", return_value=None) as spy:
			with self.assertRaisesRegex(RuntimeError, "llama-server[\\s\\S]*Gemini API[\\s\\S]*LOCAL_LLM_MODEL_PATH"):
				self.app.Translator("local")
		spy.assert_called_once_with(client)

	def test_configured_but_broken_local_service_reports_the_specific_reason_without_generic_wrapper(self):
		from llama_server import LlamaServerError

		client = ScriptedClient(lambda s, u: "")
		factory = lambda **kwargs: LocalTranslationEngine(client, startup_wait=0, log=lambda *_: None, **kwargs)
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", side_effect=LlamaServerError("找不到模型檔：C:\\models\\missing.gguf")):
			with self.assertRaises(RuntimeError) as caught:
				self.app.Translator("local")
		message = str(caught.exception)
		self.assertIn("找不到模型檔", message)
		self.assertNotIn("請先啟動 llama-server 並載入模型", message)

	def test_m3_spawned_server_gets_a_different_message_when_prepare_still_fails(self):
		# The app itself just auto-started the server (ensure_llama_server returned a
		# handle); telling the user to "start it yourself" / "configure auto-start" would
		# be nonsensical when auto-start already ran and the failure is something else.
		client = ScriptedClient(lambda s, u: LocalLLMError("暖機失敗", kind="empty"))
		factory = lambda **kwargs: LocalTranslationEngine(client, startup_wait=0, log=lambda *_: None, **kwargs)
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", return_value=mock.Mock()):
			with self.assertRaises(RuntimeError) as caught:
				self.app.Translator("local")
		message = str(caught.exception)
		self.assertIn("已自動啟動 llama-server", message)
		self.assertIn("暖機失敗", message)
		self.assertNotIn("請先啟動 llama-server 並載入模型", message)
		self.assertNotIn("LOCAL_LLM_MODEL_PATH", message)

	def test_local_live_caption_repairs_bad_reply(self):
		replies = iter(["今日はいい天気ですね。", "今天天氣真好呢。"])
		translator, client = self.local_translator(lambda system, user: next(replies))
		self.assertEqual(translator.translate_with_context("田中です。", "今日はいい天気ですね。"), "今天天氣真好呢。")
		self.assertEqual(client.calls[1][1], "請翻譯：\n今日はいい天気ですね。")

	def test_local_reconstruction_splits_truncated_batches_and_keeps_every_line(self):
		lines = [f"日文{i}です。" for i in range(24)]

		def handler(system, user):
			batch = user.split("請處理這一批句子：\n", 1)[1].splitlines()
			if len(batch) > 10:
				return LocalLLMError("截斷", kind="truncated")
			return "\n\n".join(f"{line}\n中文{line[2:-3]}。" for line in batch)

		translator, client = self.local_translator(handler)
		out_path, wav_path = self.run_reconstruction(translator, lines)
		text = out_path.read_text(encoding="utf-8")
		for i in range(24):
			self.assertIn(f"日文{i}です。\n中文{i}。", text)
		self.assertIn("翻譯模型看過完整上下文潤過的版本", text)
		self.assertFalse(wav_path.exists())
		self.assertEqual(self.sleeps, [])

	def test_local_reconstruction_keeps_recording_when_server_dies(self):
		translator, _ = self.local_translator(lambda system, user: LocalLLMError("連線失敗", retryable=True, kind="connection"))
		out_path, wav_path = self.run_reconstruction(translator, ["日文です。"])
		self.assertIsNone(out_path)
		self.assertTrue(wav_path.exists())
		self.assertIn("已保留完整錄音", self.printed_text())

	def test_riva_reconstruction_is_labelled_as_sentence_by_sentence(self):
		translator, client = self.local_translator(lambda system, user: "中文。", mode=MODE_RIVA)
		out_path, _ = self.run_reconstruction(translator, ["日文です。", "田中です。"])
		text = out_path.read_text(encoding="utf-8")
		self.assertIn("本機翻譯專用模型逐句翻譯", text)
		self.assertIn("日文です。\n中文。\n\n田中です。\n中文。", text)
		self.assertEqual(client.calls[1], ("", "Translate this into Traditional Chinese: __GLOSSARY_0__です。"))



class WhisperModelConfigTests(TranslationBackendTestCase):
	def test_cuda_default_is_large_v3(self):
		self.assertEqual(self.app.resolve_whisper_model_size(True, environ={}), "large-v3")

	def test_cpu_default_is_small(self):
		self.assertEqual(self.app.resolve_whisper_model_size(False, environ={}), "small")

	def test_whisper_model_env_overrides_both_cuda_and_cpu_defaults(self):
		self.assertEqual(self.app.resolve_whisper_model_size(True, environ={"WHISPER_MODEL": "medium"}), "medium")
		self.assertEqual(self.app.resolve_whisper_model_size(False, environ={"WHISPER_MODEL": " large-v3-turbo "}), "large-v3-turbo")

	def test_blank_env_value_falls_back_to_the_cuda_cpu_default(self):
		self.assertEqual(self.app.resolve_whisper_model_size(True, environ={"WHISPER_MODEL": "   "}), "large-v3")


class GlossaryFileTests(TranslationBackendTestCase):
	def test_glossary_hint_skips_values_that_are_not_strings(self):
		valid = {"陽宮ひな": "羊宮妃那", "HOOOOPE": "HOOOOPE", "鈴木": ""}
		for value in (["ゆみやひな", "陽宮ひな"], 1, True, {"正確寫法": "こもれびじかん"}, None):
			with self.subTest(value=value):
				hint = self.app.glossary_hint({**valid, "こもれびじかん": value})
				self.assertEqual(hint, self.app.glossary_hint(valid))
				self.assertIn("- 羊宮妃那（聽寫可能寫成：陽宮ひな）", hint)
				self.assertIn("- 鈴木\n", hint + "\n")
				self.assertEqual(self.app.glossary_hint({"こもれびじかん": value}), "")
		self.assertEqual(self.app.glossary_hint({"": ""}), "")

	def test_load_glossary_skips_and_reports_values_that_are_not_strings(self):
		valid = {"ようみやひな": "羊宮妃那", "ホープ": "HOOOOPE", "鈴木": ""}
		invalid = {"羊宮妃那": ["ゆみやひな", "陽宮ひな"], "HOOOOPE": 1, "こもれびじかん": True, "番組": {"正確寫法": "ばんぐみ"}, "田中": None}
		with tempfile.TemporaryDirectory() as temp_dir:
			path = Path(temp_dir) / "glossary.json"
			path.write_text(json.dumps({**valid, **invalid}, ensure_ascii=False, indent=2), encoding="utf-8")
			with mock.patch.object(self.app, "GLOSSARY_PATH", path):
				glossary = self.app.load_glossary()
		self.assertEqual(glossary, valid)
		lines = self.printed_text().splitlines()
		for key, value in invalid.items():
			with self.subTest(key=key):
				self.assertTrue(any(key in line and json.dumps(value, ensure_ascii=False) in line and '"聽錯寫法": "正確寫法"' in line for line in lines), lines)
		for key in valid:
			self.assertNotIn(key, self.printed_text())

	def test_live_caption_still_translates_when_a_glossary_value_is_not_a_string(self):
		# Worker.run calls translate_with_context with no try around it, so a raise here would
		# stop the captions for the rest of the session.
		translator, calls = self.gemini_translator(lambda contents, n: "羊宮妃那來了。")
		glossary = {"陽宮ひな": "羊宮妃那", "羊宮妃那": ["ゆみやひな", "陽宮ひな"], "HOOOOPE": 1}
		self.assertEqual(translator.translate_with_context("", "陽宮ひなが来ました。", glossary), "羊宮妃那來了。")
		self.assertEqual(len(calls), 1)
		self.assertIn("- 羊宮妃那（聽寫可能寫成：陽宮ひな）", calls[0]["system_instruction"])

	def test_load_glossary_recreates_the_default_file_when_missing(self):
		# glossary.json is now gitignored (a user's own list should not conflict with git
		# pull); a fresh clone must still work by falling back to _DEFAULT_GLOSSARY.
		with tempfile.TemporaryDirectory() as temp_dir:
			missing_path = Path(temp_dir) / "glossary.json"
			with mock.patch.object(self.app, "GLOSSARY_PATH", missing_path):
				glossary = self.app.load_glossary()
			self.assertEqual(glossary, self.app._DEFAULT_GLOSSARY)
			self.assertTrue(missing_path.exists())
			self.assertEqual(json.loads(missing_path.read_text(encoding="utf-8")), self.app._DEFAULT_GLOSSARY)

	def test_glossary_hint_groups_misheard_spellings_under_the_correct_name(self):
		hint = self.app.glossary_hint({"羊宮妃那": "羊宮妃那", "ゆみやひな": "羊宮妃那", "陽宮ひな": "羊宮妃那", "HOOOOPE": "HOOOOPE"})
		self.assertIn("- 羊宮妃那（聽寫可能寫成：ゆみやひな、陽宮ひな）", hint)
		self.assertIn("- HOOOOPE\n", hint + "\n")
		self.assertNotIn("完全保留原文", hint)

	def test_glossary_is_not_used_as_a_whisper_prompt(self):
		translator, _ = self.gemini_translator(lambda contents, n: "中文")
		asr = mock.Mock()
		asr.transcribe.return_value = ([FakeSegment(0, "田中です。")], None)
		with tempfile.TemporaryDirectory() as temp_dir:
			wav_path = Path(temp_dir) / "transcript_20260926_010203_audio.wav"
			wav_path.write_bytes(b"RIFF")
			self.app.rebuild_transcript_from_full_audio(wav_path, asr, translator, {"田中": "田中"}, "節目")
		self.assertIsNone(asr.transcribe.call_args.kwargs.get("initial_prompt"))


SOUND_ROWS = [
	{"Name": "喇叭", "Type": "Device", "Direction": "Render", "Default": "Render",
	 "Item ID": "{0.0.0.00000000}.{aaa}", "Command-Line Friendly ID": "FxSound Audio Enhancer\\Device\\喇叭\\Render"},
	{"Name": "Realtek HD Audio 2nd output", "Type": "Device", "Direction": "Render", "Default": "",
	 "Item ID": "{0.0.0.00000000}.{bbb}", "Command-Line Friendly ID": "Realtek(R) Audio\\Device\\Realtek HD Audio 2nd output\\Render"},
	{"Name": "CABLE Input", "Type": "Device", "Direction": "Render", "Default": "",
	 "Item ID": "{0.0.0.00000000}.{ccc}", "Command-Line Friendly ID": "VB-Audio Virtual Cable\\Device\\CABLE Input\\Render"},
	# 程式工作階段那一列的 Name 是顯示名稱（Brave），不是 brave.exe
	{"Name": "Brave", "Type": "Application", "Direction": "Render", "Default": "",
	 "Device State": "Active",  # SoundVolumeView 2.43 起，程式工作階段那幾列也會填這個欄位
	 "Item ID": "{0.0.0.00000000}.{bbb}|\\Device\\HarddiskVolume3\\Program Files\\Brave\\brave.exe%b{0}|1%b42",
	 "Process Path": "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"},
]

# 程式以前用過的裝置上會留著 Inactive／Expired 的工作階段；裝置沿用 SOUND_ROWS 前三列（喇叭是系統預設）
SESSION_ROWS = SOUND_ROWS[:3] + [
	{"Name": "Firefox", "Type": "Application", "Direction": "Render", "Default": "", "Device State": "Inactive",
	 "Item ID": "{0.0.0.00000000}.{bbb}|\\Device\\HarddiskVolume3\\Program Files\\Mozilla Firefox\\firefox.exe%b{0}|1%b7",
	 "Process Path": "C:\\Program Files\\Mozilla Firefox\\firefox.exe"},
	{"Name": "Firefox", "Type": "Application", "Direction": "Render", "Default": "", "Device State": "Active",
	 "Item ID": "{0.0.0.00000000}.{aaa}|\\Device\\HarddiskVolume3\\Program Files\\Mozilla Firefox\\firefox.exe%b{0}|1%b7",
	 "Process Path": "C:\\Program Files\\Mozilla Firefox\\firefox.exe"},
	{"Name": "Microsoft Edge", "Type": "Application", "Direction": "Render", "Default": "", "Device State": "Inactive",
	 "Item ID": "{0.0.0.00000000}.{aaa}|\\Device\\HarddiskVolume3\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe%b{0}|1%b9",
	 "Process Path": "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"},
	{"Name": "Microsoft Edge", "Type": "Application", "Direction": "Render", "Default": "", "Device State": "Expired",
	 "Item ID": "{0.0.0.00000000}.{bbb}|\\Device\\HarddiskVolume3\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe%b{0}|1%b9",
	 "Process Path": "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"},
	{"Name": "Google Chrome", "Type": "Application", "Direction": "Render", "Default": "", "Device State": "Active",
	 "Item ID": "{0.0.0.00000000}.{ccc}|\\Device\\HarddiskVolume3\\Program Files\\Google\\Chrome\\Application\\chrome.exe%b{0}|1%b5",
	 "Process Path": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"},
]


class _FakeAudio:
	"""Stands in for a numpy float array: numpy is stubbed in this test module."""

	def __init__(self, samples):
		self.samples = samples

	def __mul__(self, _):
		return self

	def astype(self, _):
		return self

	def tobytes(self):
		return b"\x00\x00" * self.samples


class RecordingTests(TranslationBackendTestCase):
	def make_capture(self, record_path):
		capture = object.__new__(self.app.AudioCapture)  # skip PyAudio / VAD setup
		capture._record_path = record_path
		capture._wav_writer = None
		capture._recording = False
		capture._wav_lock = __import__("threading").Lock()
		return capture

	def make_running_capture(self, open_results):
		"""An AudioCapture wired to fakes: _open_stream returns/raises open_results in order."""
		import threading
		capture = self.make_capture(None)
		capture._stop_event = threading.Event()
		capture._stream = None
		capture.pa = mock.Mock()
		capture.on_error = mock.Mock()
		capture.stream_sr = 48000
		capture.channels = 2
		capture.device = {"name": "CABLE Input (VB-Audio Virtual Cable) [Loopback]"}
		capture._preferred_name = capture.device["name"]
		capture.audio_ring = None
		capture.pause_state = None
		capture.vad_iterator = mock.Mock()
		capture._open_stream = mock.Mock(side_effect=open_results)
		return capture

	def test_run_always_cleans_up_and_reports_when_the_loop_crashes(self):
		capture = self.make_running_capture([])
		stream = mock.Mock()
		capture._stream = stream
		capture._capture_loop = mock.Mock(side_effect=RuntimeError("boom"))
		with mock.patch("builtins.print"):
			capture.run()
		stream.close.assert_called_once_with()
		capture.pa.terminate.assert_called_once_with()
		self.assertIn("音訊擷取停止了", capture.on_error.call_args.args[0])

	def test_first_open_failure_is_retried_and_a_bad_block_does_not_end_the_thread(self):
		stream = mock.Mock()
		reads = []

		def read(frames, exception_on_overflow=False):
			reads.append(frames)
			if len(reads) >= 2:
				capture._stop_event.set()
			return b"\x00\x00\x00\x00" * 2 * frames

		stream.read.side_effect = read
		capture = self.make_running_capture([OSError("device busy"), stream])
		# numpy is stubbed in this module; this stand-in only supports what the loop needs, and
		# its frombuffer raises: that is the "processing one block failed" case, after which the
		# loop must keep reading
		fake_np = types.SimpleNamespace(
			empty=lambda *a, **k: [], float32="float32",
			frombuffer=mock.Mock(side_effect=ValueError("bad block")),
		)
		with mock.patch.object(self.app.time, "sleep"), mock.patch("builtins.print"), \
				mock.patch.object(self.app, "np", fake_np):
			capture.run()
		self.assertEqual(capture._open_stream.call_count, 2)       # retried after the first failure
		self.assertEqual(len(reads), 2)                            # kept reading after the bad block
		messages = [call.args[0] for call in capture.on_error.call_args_list]
		self.assertTrue(any("開啟音訊串流失敗" in m for m in messages))
		self.assertTrue(any("處理音訊時發生錯誤" in m for m in messages))
		stream.close.assert_called_once_with()                     # cleaned up by run()'s finally

	def test_stop_then_start_again_appends_instead_of_overwriting(self):
		import wave
		fake_np = types.SimpleNamespace(clip=lambda x, *_: x, int16="int16")
		with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(self.app, "np", fake_np):
			path = Path(temp_dir) / "transcript_audio.wav"
			capture = self.make_capture(path)
			self.assertTrue(capture.start_recording())
			capture._write_recording(_FakeAudio(16000))
			self.assertTrue(capture.stop_recording())
			capture._write_recording(_FakeAudio(8000))  # not recording: must be skipped
			self.assertTrue(capture.start_recording())
			capture._write_recording(_FakeAudio(4000))
			capture._close_recording()
			with wave.open(str(path), "rb") as wav:
				self.assertEqual(wav.getnframes(), 16000 + 4000)

	def test_recording_flags(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			capture = self.make_capture(Path(temp_dir) / "a.wav")
			self.assertFalse(capture.stop_recording())
			self.assertTrue(capture.start_recording())
			self.assertFalse(capture.start_recording())
			self.assertTrue(capture.is_recording())
			capture.stop_recording()
			self.assertFalse(capture.is_recording())
			capture._close_recording()
		self.assertFalse(self.make_capture(None).start_recording())


class AudioRoutingTests(TranslationBackendTestCase):
	def test_app_device_is_found_by_process_path_not_display_name(self):
		self.assertEqual(self.app.get_app_output_device("brave.exe", SOUND_ROWS),
			"Realtek(R) Audio\\Device\\Realtek HD Audio 2nd output\\Render")

	def test_app_without_a_session_returns_none(self):
		self.assertIsNone(self.app.get_app_output_device("chrome.exe", SOUND_ROWS))

	def test_default_and_named_devices(self):
		self.assertEqual(self.app.get_default_output_device(SOUND_ROWS), "FxSound Audio Enhancer\\Device\\喇叭\\Render")
		self.assertEqual(self.app.find_output_device("CABLE Input", SOUND_ROWS), "VB-Audio Virtual Cable\\Device\\CABLE Input\\Render")
		self.assertIsNone(self.app.find_output_device("不存在的裝置", SOUND_ROWS))

	def test_restorer_switches_back_only_once(self):
		self.patch_sound_volume_view()
		with mock.patch.object(self.app, "set_app_output_device", return_value=True) as set_device:
			restore = self.app.AudioRoutingRestorer("brave.exe", "Realtek(R) Audio\\Device\\耳機\\Render")
			restore()
			restore()  # finally + atexit + console handler may all fire
		set_device.assert_called_once_with("brave.exe", "Realtek(R) Audio\\Device\\耳機\\Render", timeout=10.0)

	def test_restorer_retries_on_the_next_trigger_after_a_failure(self):
		self.patch_sound_volume_view()
		with mock.patch.object(self.app, "set_app_output_device", side_effect=[False, True]) as set_device:
			restore = self.app.AudioRoutingRestorer("brave.exe", None)
			restore()  # e.g. the console handler timed out
			restore()  # finally / atexit tries again
			restore()  # already succeeded: nothing more
		self.assertEqual(set_device.call_count, 2)

	def test_restorer_gives_up_after_max_attempts(self):
		self.patch_sound_volume_view()
		with mock.patch.object(self.app, "set_app_output_device", return_value=False) as set_device:
			restore = self.app.AudioRoutingRestorer("brave.exe", None)
			for _ in range(5):
				restore()
		self.assertEqual(set_device.call_count, self.app.AudioRoutingRestorer.MAX_ATTEMPTS)

	def test_non_zero_exit_code_counts_as_a_failed_switch(self):
		self.patch_sound_volume_view()
		with mock.patch("subprocess.run", return_value=types.SimpleNamespace(returncode=3)):
			self.assertFalse(self.app.set_app_output_device("brave.exe", "DefaultRenderDevice"))
		with mock.patch("subprocess.run", return_value=types.SimpleNamespace(returncode=0)) as run:
			self.assertTrue(self.app.set_app_output_device("brave.exe", "DefaultRenderDevice", timeout=2.0))
		self.assertEqual(run.call_args.kwargs["timeout"], 2.0)

	def test_restorer_without_a_remembered_device_follows_the_system_default(self):
		# goes through restore_app_output_device: back to "follow the system default", not pinned
		self.patch_sound_volume_view()
		with mock.patch.object(self.app, "set_app_output_device", return_value=True) as set_device:
			self.app.AudioRoutingRestorer("brave.exe", None)()
		set_device.assert_called_once_with("brave.exe", "DefaultRenderDevice", timeout=10.0)

	def test_console_close_event_runs_the_restorer(self):
		handlers = []
		win32api = types.SimpleNamespace(SetConsoleCtrlHandler=lambda handler, add: handlers.append(handler))
		win32con = types.SimpleNamespace(
			CTRL_C_EVENT=0, CTRL_BREAK_EVENT=1, CTRL_CLOSE_EVENT=2, CTRL_LOGOFF_EVENT=5, CTRL_SHUTDOWN_EVENT=6,
		)
		callback = mock.Mock()
		with mock.patch.dict(sys.modules, {"win32api": win32api, "win32con": win32con}):
			self.app._install_console_close_handler(callback)
		self.assertFalse(handlers[0](2))  # CTRL_CLOSE_EVENT: run it, but let Windows continue
		# short per-call timeout so both SoundVolumeView calls fit in Windows' ~5 s close window
		callback.assert_called_once_with(timeout=self.app.CONSOLE_CLOSE_TIMEOUT_SEC)
		self.assertLessEqual(2 * self.app.CONSOLE_CLOSE_TIMEOUT_SEC, 5)

	def test_inactive_session_listed_first_is_not_the_current_device(self):
		self.assertEqual(self.app.get_app_output_device("firefox.exe", SESSION_ROWS),
			"FxSound Audio Enhancer\\Device\\喇叭\\Render")

	def test_app_with_only_inactive_sessions_returns_none(self):
		self.assertIsNone(self.app.get_app_output_device("msedge.exe", SESSION_ROWS))

	def test_session_without_device_state_returns_none(self):
		# SoundVolumeView 2.43 以前，程式工作階段那幾列的 Device State 是空的
		old_rows = [{**row, "Device State": ""} if row["Type"] == "Application" else row for row in SOUND_ROWS]
		self.assertIsNone(self.app.get_app_output_device("brave.exe", old_rows))

	def test_remember_skips_a_session_on_the_system_default_device(self):
		cable = "VB-Audio Virtual Cable\\Device\\CABLE Input\\Render"
		self.assertIsNone(self.app.remember_app_output_device("firefox.exe", cable, SESSION_ROWS))

	def test_remember_skips_a_session_left_on_cable(self):
		cable = "VB-Audio Virtual Cable\\Device\\CABLE Input\\Render"
		self.assertIsNone(self.app.remember_app_output_device("chrome.exe", cable, SESSION_ROWS))

	def test_remember_keeps_a_session_on_another_device(self):
		cable = "VB-Audio Virtual Cable\\Device\\CABLE Input\\Render"
		self.assertEqual(self.app.remember_app_output_device("brave.exe", cable, SOUND_ROWS),
			"Realtek(R) Audio\\Device\\Realtek HD Audio 2nd output\\Render")

	def patch_sound_volume_view(self, exists=True):
		# 只是讓 exists() 成立的空檔案；用到的測試都把 set_app_output_device 換掉了，不會真的去執行它
		temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(temp_dir.cleanup)
		path = Path(temp_dir.name) / "SoundVolumeView.exe"
		if exists:
			path.write_bytes(b"")
		patcher = mock.patch.object(self.app, "SOUND_VOLUME_VIEW_PATH", path)
		patcher.start()
		self.addCleanup(patcher.stop)

	def test_restore_without_a_remembered_device_follows_the_system_default(self):
		self.patch_sound_volume_view()
		with mock.patch.object(self.app, "set_app_output_device", return_value=True) as set_device:
			self.assertTrue(self.app.restore_app_output_device("brave.exe", None))
		set_device.assert_called_once_with("brave.exe", "DefaultRenderDevice", timeout=10.0)
		self.assertIn("跟著系統預設的播放裝置", self.printed_text())
		self.assertNotIn("沒辦法自動切回", self.printed_text())

	def test_restore_switches_back_to_the_remembered_device(self):
		self.patch_sound_volume_view()
		device = "Realtek(R) Audio\\Device\\Realtek HD Audio 2nd output\\Render"
		with mock.patch.object(self.app, "set_app_output_device", return_value=True) as set_device:
			self.app.restore_app_output_device("brave.exe", device)
		set_device.assert_called_once_with("brave.exe", device, timeout=10.0)
		self.assertIn(f"已把 brave.exe 的輸出裝置切回「{device}」。", self.printed_text())
		self.assertNotIn("沒辦法自動切回", self.printed_text())

	def test_restore_failure_asks_to_switch_back_by_hand(self):
		self.patch_sound_volume_view()
		with mock.patch.object(self.app, "set_app_output_device", return_value=False):
			self.app.restore_app_output_device("brave.exe", None)
		self.assertIn("沒辦法自動切回 brave.exe 的輸出裝置，可能需要自己在音量混音器裡切回來。", self.printed_text())
		self.assertNotIn("已把", self.printed_text())

	def test_restore_without_sound_volume_view_asks_to_switch_back_by_hand(self):
		self.patch_sound_volume_view(exists=False)
		with mock.patch.object(self.app, "set_app_output_device") as set_device:
			self.app.restore_app_output_device("brave.exe", None)
		self.assertIn("請自己在 Windows 音量混音器裡把 brave.exe 的輸出裝置從「CABLE Input」改回原本的設定。", self.printed_text())
		set_device.assert_not_called()

	def test_remember_skips_a_session_left_on_another_cable(self):
		# 沒有名稱剛好是 CABLE Input 的裝置時，main() 傳進來的只是「CABLE Input」這個字串
		rows = SOUND_ROWS[:2] + [
			{"Name": "CABLE In 16ch", "Type": "Device", "Direction": "Render", "Default": "",
			 "Item ID": "{0.0.0.00000000}.{ddd}", "Command-Line Friendly ID": "VB-Audio Virtual Cable\\Device\\CABLE In 16ch\\Render"},
			{"Name": "Brave", "Type": "Application", "Direction": "Render", "Default": "", "Device State": "Active",
			 "Item ID": "{0.0.0.00000000}.{ddd}|\\Device\\HarddiskVolume3\\Program Files\\Brave\\brave.exe%b{0}|1%b42",
			 "Process Path": "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"},
		]
		self.assertIsNone(self.app.remember_app_output_device("brave.exe", "CABLE Input", rows))

	def test_remember_without_an_active_session_returns_none(self):
		cable = "VB-Audio Virtual Cable\\Device\\CABLE Input\\Render"
		self.assertIsNone(self.app.remember_app_output_device("msedge.exe", cable, SESSION_ROWS))


class WorkflowWiringTests(TranslationBackendTestCase):
	def test_worker_reuses_injected_translator_without_constructing_another(self):
		injected = mock.Mock()
		with mock.patch.object(self.app, "Translator") as translator_class, \
				mock.patch.object(self.app, "load_glossary", return_value={}), \
				tempfile.TemporaryDirectory() as temp_dir, \
				mock.patch.object(self.app, "TRANSCRIPT_DIR", Path(temp_dir)):
			worker = self.app.Worker(queue.Queue(), queue.Queue(), queue.Queue(), translator=injected)
		translator_class.assert_not_called()
		self.assertIs(worker.translator, injected)

	def test_worker_keeps_confident_speech_even_with_high_no_speech_prob(self):
		# large-v3 often gives clear speech over background music a no_speech_prob of 0.6~0.9;
		# faster-whisper's own rule (no_speech_prob AND low avg_logprob) already filters silence,
		# so the worker must not drop a segment just because no_speech_prob is high.
		import time
		segment = FakeSegment(0, "していく番組です")
		segment.no_speech_prob = 0.85
		segment.avg_logprob = -0.16
		translator = mock.Mock()
		translator.translate_with_context.return_value = "這是會這樣做下去的節目。"
		utterances, captions = queue.Queue(), queue.Queue()
		with mock.patch.object(self.app, "load_glossary", return_value={}), \
				tempfile.TemporaryDirectory() as temp_dir, \
				mock.patch.object(self.app, "TRANSCRIPT_DIR", Path(temp_dir)):
			worker = self.app.Worker(utterances, captions, queue.Queue(), translator=translator)
			worker.asr = mock.Mock()
			worker.asr.transcribe.return_value = ([segment], None)
			worker.start()
			utterances.put(([0.0] * 16000, time.time()))  # numpy is stubbed here; asr is mocked anyway
			try:
				_, ja, zh = captions.get(timeout=5)
			finally:
				worker.stop()
				worker.join(timeout=5)
		self.assertEqual(ja, "していく番組です")
		self.assertEqual(zh, "這是會這樣做下去的節目。")

	def test_worker_keeps_running_after_an_error_and_shows_a_warning(self):
		import time
		segment = FakeSegment(0, "よろしく")
		segment.no_speech_prob = 0.1
		segment.avg_logprob = -0.1
		translator = mock.Mock()
		translator.translate_with_context.return_value = "請多指教"
		utterances, captions = queue.Queue(), queue.Queue()
		with mock.patch.object(self.app, "load_glossary", return_value={}), \
				tempfile.TemporaryDirectory() as temp_dir, \
				mock.patch.object(self.app, "TRANSCRIPT_DIR", Path(temp_dir)), \
				mock.patch("builtins.print") as printed:
			worker = self.app.Worker(utterances, captions, queue.Queue(), translator=translator)
			worker.asr = mock.Mock()
			worker.asr.transcribe.side_effect = [
				RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"),
				([segment], None),
			]
			worker.start()
			utterances.put(([0.0] * 16000, time.time()))
			utterances.put(([0.0] * 16000, time.time()))
			try:
				warning = captions.get(timeout=5)
				caption = captions.get(timeout=5)
			finally:
				worker.stop()
				worker.join(timeout=5)
		self.assertFalse(worker.is_alive())
		self.assertIn("失敗", warning[1])
		self.assertIn("cuBLAS", warning[2])
		self.assertEqual(caption[1:], ("よろしく", "請多指教"))
		self.assertIn("cuBLAS", " ".join(str(c.args) for c in printed.call_args_list))



class KeptRecordingCommandTests(TranslationBackendTestCase):
	"""Every message that keeps the recording must say how to re-process it; the console gives the exact command."""

	def assert_rebuild_command(self, text, wav_path):
		import re

		match = re.search(r'python "([^"]+)" --rebuild "([^"]+)"', text)
		self.assertIsNotNone(match, text)
		script, wav = match.groups()
		# Absolute paths, so the command works whichever folder it is pasted into.
		self.assertTrue(os.path.isabs(script) and os.path.isabs(wav), match.group(0))
		self.assertTrue(os.path.samefile(script, self.app.__file__), script)
		self.assertTrue(os.path.samefile(wav, wav_path), wav)

	def test_rebuild_command_makes_a_relative_recording_path_absolute(self):
		relative = Path("transcripts/transcript_20260926_010203_audio.wav")
		command = self.app._rebuild_command(relative)
		self.assertTrue(command.endswith(f' --rebuild "{os.path.abspath(relative)}"'), command)

	def test_partial_failure_transcript_gives_the_rebuild_command_without_local_paths(self):
		translator, _ = self.gemini_translator(lambda contents, n: "" if n <= 3 else f"批次{n}")
		out_path, wav_path = self.run_reconstruction(translator, [f"日文{i}です。" for i in range(61)])
		text = out_path.read_text(encoding="utf-8")
		self.assertIn("--rebuild", text)
		self.assertIn(wav_path.name, text)
		# The transcript is meant to be read and shared; absolute paths would expose the user's folder.
		self.assertNotIn(str(wav_path.parent), text)
		self.assertNotIn(os.path.abspath(self.app.__file__), text)

	def test_partial_failure_console_message_gives_the_rebuild_command(self):
		translator, _ = self.gemini_translator(lambda contents, n: "" if n <= 3 else f"批次{n}")
		_, wav_path = self.run_reconstruction(translator, [f"日文{i}です。" for i in range(61)])
		self.assert_rebuild_command(self.printed_text(), wav_path)

	def test_empty_translation_console_message_gives_the_rebuild_command(self):
		translator, _ = self.gemini_translator(lambda contents, n: "")
		out_path, wav_path = self.run_reconstruction(translator, ["日文です。"])
		self.assertIsNone(out_path)
		self.assert_rebuild_command(self.printed_text(), wav_path)

	def test_local_service_died_console_message_gives_the_rebuild_command(self):
		translator, _ = self.local_translator(lambda system, user: LocalLLMError("連線失敗", retryable=True, kind="connection"))
		out_path, wav_path = self.run_reconstruction(translator, ["日文です。"])
		self.assertIsNone(out_path)
		self.assert_rebuild_command(self.printed_text(), wav_path)



class SavedRecordingRebuildTests(TranslationBackendTestCase):
	"""rebuild_saved_recording re-processes a kept recording with the backend main() would pick."""

	def saved_recording(self, name="transcript_20260926_010203_audio.wav", title="節目", settings=None):
		temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(temp_dir.cleanup)
		folder = Path(temp_dir.name)
		wav_path = folder / name
		wav_path.write_bytes(b"RIFF")
		if title is not None:
			(folder / "transcript_20260926_010203.txt").write_text(f"#TITLE: {title}\n\n", encoding="utf-8")
		settings_path = folder / "last_settings.json"
		if settings is not None:
			settings_path.write_text(json.dumps(settings), encoding="utf-8")
		patcher = mock.patch.object(self.app, "SETTINGS_PATH", settings_path)
		patcher.start()
		self.addCleanup(patcher.stop)
		return wav_path

	def rebuild_saved(self, wav_path, translator_error=None, whisper_error=None, rebuild_error=None):
		"""Run rebuild_saved_recording with the models and the rebuild step itself mocked out."""
		with mock.patch.object(self.app, "Translator", side_effect=translator_error) as translator_class, \
				mock.patch.object(self.app, "WhisperModel", side_effect=whisper_error) as whisper_class, \
				mock.patch.object(self.app, "load_glossary", return_value={"田中": "田中"}), \
				mock.patch.object(
					self.app, "rebuild_transcript_from_full_audio", return_value=Path("out.md"), side_effect=rebuild_error
				) as rebuild:
			result = self.app.rebuild_saved_recording(wav_path)
		return result, translator_class, whisper_class, rebuild

	def test_rebuild_uses_the_backend_main_would_pick_the_glossary_and_the_episode_title(self):
		cases = (
			({"TRANSLATION_BACKEND": "local"}, {"translation_backend": "gemini"}, "local"),
			({}, {"translation_backend": "local"}, "local"),
			({}, None, "gemini"),
			({"TRANSLATION_BACKEND": "ollama"}, {"translation_backend": "local"}, "gemini"),
		)
		for env, settings, expected in cases:
			with self.subTest(env=env, settings=settings), mock.patch.dict(os.environ, env):
				wav_path = self.saved_recording(settings=settings)
				result, translator_class, whisper_class, rebuild = self.rebuild_saved(wav_path)
				self.assertEqual(result, Path("out.md"))
				translator_class.assert_called_once_with(expected)
				whisper_class.assert_called_once_with(
					self.app.WHISPER_MODEL_SIZE, device=self.app.WHISPER_DEVICE, compute_type=self.app.WHISPER_COMPUTE_TYPE
				)
				rebuild.assert_called_once_with(
					wav_path, whisper_class.return_value, translator_class.return_value, {"田中": "田中"}, "節目"
				)
		self.assertIn("改用 Gemini API", self.printed_text())

	def test_rebuild_without_the_raw_transcript_has_no_episode_title(self):
		_, _, _, rebuild = self.rebuild_saved(self.saved_recording(title=None))
		self.assertEqual(rebuild.call_args.args[4], "")

	def test_unreadable_raw_transcript_still_rebuilds_without_an_episode_title(self):
		wav_path = self.saved_recording(title=None)
		# Re-saved as Big5 (Notepad's ANSI on a Taiwanese Windows), so it is not valid UTF-8.
		wav_path.with_name("transcript_20260926_010203.txt").write_bytes("#TITLE: 節目\n\n".encode("big5"))
		result, translator_class, whisper_class, rebuild = self.rebuild_saved(wav_path)
		self.assertEqual(result, Path("out.md"))
		rebuild.assert_called_once_with(
			wav_path, whisper_class.return_value, translator_class.return_value, {"田中": "田中"}, ""
		)
		self.assertRegex(self.printed_text(), r"讀取 transcript_20260926_010203\.txt 的節目名稱失敗，改用預設標題：.*can't decode")
		self.assertNotIn("用完整錄音重新整理逐字稿時發生錯誤", self.printed_text())

	def test_success_prints_where_the_new_transcript_is(self):
		result, _, _, _ = self.rebuild_saved(self.saved_recording())
		self.assertEqual(result, Path("out.md"))
		self.assertIn("已產生用完整錄音重新辨識、準確度更好的逐字稿：out.md", self.printed_text())

	def test_rebuild_error_returns_none_and_prints_the_command_to_try_again(self):
		wav_path = self.saved_recording()
		error = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
		result, _, _, _ = self.rebuild_saved(wav_path, rebuild_error=error)
		self.assertIsNone(result)
		self.assertIn("用完整錄音重新整理逐字稿時發生錯誤：Library cublas64_12.dll is not found", self.printed_text())
		self.assertIn(f"之後可以用這行指令重新處理：{self.app._rebuild_command(wav_path)}", self.printed_text())
		self.assertTrue(wav_path.exists())

	def test_whisper_load_error_returns_none_and_prints_the_command_to_try_again(self):
		wav_path = self.saved_recording()
		error = RuntimeError("CUDA failed with error no CUDA-capable device is detected")
		result, translator_class, whisper_class, rebuild = self.rebuild_saved(wav_path, whisper_error=error)
		self.assertIsNone(result)
		# Built before the slow Whisper load, so a bad API key is reported without waiting for the model.
		translator_class.assert_called_once()
		whisper_class.assert_called_once()
		rebuild.assert_not_called()
		self.assertIn("用完整錄音重新整理逐字稿時發生錯誤：CUDA failed with error no CUDA-capable device", self.printed_text())
		self.assertIn(f"之後可以用這行指令重新處理：{self.app._rebuild_command(wav_path)}", self.printed_text())
		self.assertTrue(wav_path.exists())

	def test_rebuild_error_after_the_recording_was_deleted_does_not_print_the_command(self):
		wav_path = self.saved_recording()

		def delete_then_fail(wav, *_):
			wav.unlink()
			raise RuntimeError("整理到一半發生錯誤")

		result, _, _, _ = self.rebuild_saved(wav_path, rebuild_error=delete_then_fail)
		self.assertIsNone(result)
		self.assertFalse(wav_path.exists())
		self.assertIn("用完整錄音重新整理逐字稿時發生錯誤：整理到一半發生錯誤", self.printed_text())
		# The recording is gone, so the command could only answer 找不到錄音檔.
		self.assertNotIn("之後可以用這行指令重新處理", self.printed_text())

	def test_missing_recording_returns_none_before_loading_any_model(self):
		temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(temp_dir.cleanup)
		missing = Path(temp_dir.name) / "transcript_20260926_010203_audio.wav"
		result, translator_class, whisper_class, rebuild = self.rebuild_saved(missing)
		self.assertIsNone(result)
		translator_class.assert_not_called()
		whisper_class.assert_not_called()
		rebuild.assert_not_called()
		self.assertIn(str(missing), self.printed_text())

	def test_audio_the_app_did_not_keep_is_refused_before_loading_any_model(self):
		# A successful rebuild deletes its input, so it must only ever touch the app's own recordings.
		for name in ("podcast.wav", "podcast_audio.wav", "transcript_20260926_010203_notebooklm_style.md"):
			with self.subTest(name=name):
				other = self.saved_recording(name=name)
				result, translator_class, whisper_class, rebuild = self.rebuild_saved(other)
				self.assertIsNone(result)
				translator_class.assert_not_called()
				whisper_class.assert_not_called()
				rebuild.assert_not_called()
				self.assertEqual(other.read_bytes(), b"RIFF")
				self.assertIn(str(other), self.printed_text())

	def test_translator_failure_returns_none_before_loading_whisper(self):
		error = RuntimeError("找不到 Gemini API 金鑰")
		result, _, whisper_class, rebuild = self.rebuild_saved(self.saved_recording(), translator_error=error)
		self.assertIsNone(result)
		whisper_class.assert_not_called()
		rebuild.assert_not_called()
		self.assertIn("翻譯引擎初始化失敗：找不到 Gemini API 金鑰", self.printed_text())



class RebuildCommandLineTests(TranslationBackendTestCase):
	def test_no_arguments_open_the_app_as_before(self):
		with mock.patch.object(self.app, "main") as main, \
				mock.patch.object(self.app, "rebuild_saved_recording") as rebuild:
			self.assertEqual(self.app._run_cli([]), 0)
		main.assert_called_once_with()
		rebuild.assert_not_called()

	def test_rebuild_exit_code_says_whether_a_transcript_was_produced(self):
		wav = "transcript_20260926_010203_audio.wav"
		for produced, expected in ((Path("transcript_20260926_010203_notebooklm_style.md"), 0), (None, 1)):
			with self.subTest(produced=produced):
				with mock.patch.object(self.app, "main") as main, \
						mock.patch.object(self.app, "rebuild_saved_recording", return_value=produced) as rebuild:
					self.assertEqual(self.app._run_cli(["--rebuild", wav]), expected)
				rebuild.assert_called_once_with(Path(wav))
				main.assert_not_called()

	def test_other_arguments_print_usage_and_exit_2(self):
		for argv in (["--help"], ["--rebuild"], ["transcript_20260926_010203_audio.wav"], ["--rebuild", "a.wav", "b.wav"]):
			with self.subTest(argv=argv):
				with mock.patch.object(self.app, "main") as main, \
						mock.patch.object(self.app, "rebuild_saved_recording") as rebuild:
					self.assertEqual(self.app._run_cli(argv), 2)
				main.assert_not_called()
				rebuild.assert_not_called()
		self.assertIn("--rebuild", self.printed_text())



if __name__ == "__main__":
	unittest.main()
