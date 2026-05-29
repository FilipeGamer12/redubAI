"""
Redublagem automática de vídeos MKV com IA.
(Com suporte opcional a CPU e AMD GPU via ROCm)

Fluxo:
1) Extrai o áudio do vídeo (pode ser pulado com --skip-transcription)
2) Transcreve com Whisper e salva um SRT original (pode ser pulado com --original-srt)
3) Traduz o SRT completo com IA via Ollama (pode ser pulado com --skip-translation)
4) Gera áudio dublado com edge-tts
5) Ajusta cada trecho para caber no tempo do segmento
6) Converte a faixa dublada para AAC
7) Adiciona a faixa dublada como áudio extra no MKV original

Requisitos:
- ffmpeg e ffprobe instalados no sistema
- openai-whisper
- edge-tts
- pydub
- ollama rodando localmente com um modelo IA baixado
- PyTorch (opcionalmente com CUDA ou ROCm para acelerar Whisper)

Antes de usar:
    ollama pull [IA]
    ollama serve

Exemplo:
    python redublagem_ia.py --input input/video.mkv --output output/video_dublado.mkv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


# ----------------------------
# Estruturas de dados
# ----------------------------

@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str
    translated_text: str = ""
    tts_file: str = ""
    tts_duration: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))


# ----------------------------
# Utilidades gerais
# ----------------------------

def die(message: str, code: int = 1) -> None:
    print(f"[ERRO] {message}", file=sys.stderr)
    raise SystemExit(code)


def info(message: str) -> None:
    print(f"[INFO] {message}")


def run(cmd: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def ensure_command_exists(name: str) -> None:
    if shutil.which(name) is None:
        die(f"O comando '{name}' não foi encontrado no PATH. Instale o FFmpeg e tente novamente.")


def safe_import(module: str, pip_name: str | None = None) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:
        pkg = pip_name or module
        die(f"Dependência ausente: '{pkg}'. Instale com: pip install {pkg}\nDetalhe: {exc}")


def ask_device() -> str:
    """Pergunta ao usuário qual dispositivo usar: CPU ou GPU (CUDA/ROCm)."""
    print("\nEscolha o dispositivo para execução do Whisper:")
    print("1 - CPU (mais lento, mas compatível universalmente)")
    print("2 - GPU (CUDA ou ROCm - mais rápido, requer PyTorch com suporte)")
    choice = input("Digite 1 ou 2: ").strip()
    if choice == "1":
        return "cpu"
    elif choice == "2":
        return "cuda"
    else:
        print("Opção inválida. Usando CPU por padrão.")
        return "cpu"


def setup_device(device_choice: str) -> str:
    """
    Valida e configura o dispositivo escolhido.
    Retorna o string do dispositivo ('cuda' ou 'cpu') a ser usado.
    """
    torch = safe_import("torch")
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            die(
                "Você escolheu GPU, mas o PyTorch não detectou nenhuma GPU compatível.\n"
                "Certifique-se de ter instalado o PyTorch com CUDA (NVIDIA) ou ROCm (AMD).\n"
                "Ou escolha a opção CPU na próxima execução."
            )
        info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        info("Usando dispositivo: CPU (nenhuma GPU será utilizada).")
        return "cpu"


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def clean_text(text: str) -> str:
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("\"'“”‘’")
    return text


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_srt(path: Path, segments: Iterable[Segment], *, translated: bool = False) -> None:
    lines: list[str] = []
    for seg in segments:
        text = seg.translated_text if translated else seg.text
        text = clean_text(text)
        if not text:
            continue
        lines.append(str(seg.index))
        lines.append(f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_srt(path: Path) -> list[Segment]:
    """
    Lê um arquivo SRT e retorna uma lista de Segmentos.
    Índices são reatribuídos sequencialmente.
    """
    if not path.exists():
        die(f"Arquivo SRT não encontrado: {path}")
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", content.strip())
    segments: list[Segment] = []
    idx = 1
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        try:
            ts_match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", lines[1])
            if not ts_match:
                continue
            start_str = ts_match.group(1)
            end_str = ts_match.group(2)

            def ts_to_seconds(ts: str) -> float:
                h, m, s = ts.split(":")
                sec, ms = s.split(",")
                return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0

            start = ts_to_seconds(start_str)
            end = ts_to_seconds(end_str)
            text = " ".join(lines[2:])
            text = clean_text(text)
            if not text:
                continue
            segments.append(Segment(index=idx, start=start, end=end, text=text))
            idx += 1
        except Exception:
            continue
    if not segments:
        die("Nenhum segmento válido encontrado no SRT.")
    return segments


# ----------------------------
# Extração de áudio
# ----------------------------

def extract_audio(video_path: Path, wav_path: Path) -> None:
    ensure_command_exists("ffmpeg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    info("Extraindo áudio do vídeo...")
    run(cmd)


# ----------------------------
# Whisper / ASR com escolha de dispositivo
# ----------------------------

def transcribe_with_whisper(
    wav_path: Path,
    *,
    asr_model: str = "large-v3",
    device: str = "cuda",  # "cuda" ou "cpu"
) -> list[Segment]:
    whisper = safe_import("whisper", "openai-whisper")
    torch = safe_import("torch")

    info(f"Carregando Whisper ({asr_model}) em device={device}...")
    model = whisper.load_model(asr_model, device=device)

    try:
        # Para CPU, fp16 pode causar problemas, então desabilitar
        use_fp16 = (device == "cuda")
        result = model.transcribe(
            str(wav_path),
            task="transcribe",
            verbose=False,
            fp16=use_fp16,
        )
    finally:
        info("Descarregando Whisper da memória...")
        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    segments_raw = result.get("segments", [])
    if not segments_raw:
        die("Whisper não retornou segmentos de transcrição.")

    language = result.get("language", "desconhecido")
    info(f"Idioma detectado: {language}")

    segments: list[Segment] = []
    for s in segments_raw:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        text = clean_text(str(s.get("text", "")))
        if not text:
            continue
        segments.append(Segment(index=len(segments) + 1, start=start, end=end, text=text))

    if not segments:
        die("Nenhum segmento válido foi produzido pela transcrição.")

    return segments


# ----------------------------
# Tradução do SRT completo com Ollama (nova versão)
# ----------------------------

class OllamaTranslator:
    def __init__(
        self,
        model_name: str = "gemma2:9b",
        *,
        base_url: str = "http://localhost:11434",
        max_new_tokens: int = 4096,    # suficiente para muitas legendas
        temperature: float = 0.2,
        top_p: float = 0.9,
        timeout: int = 600,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def translate_full_srt(self, segments: list[Segment]) -> list[str]:
        """
        Envia o SRT completo para tradução e retorna uma lista de traduções
        na mesma ordem dos segmentos.
        """
        if not segments:
            return []

        # Monta o texto com os índices e falas originais
        srt_text_lines = []
        for seg in segments:
            srt_text_lines.append(f"{seg.index}: {seg.text}")
        full_text = "\n".join(srt_text_lines)

        # Prompt de sistema (em inglês para melhor aderência)
        system_prompt = (
            "You are a professional subtitle translator. "
            "You will be given a list of subtitle lines, each prefixed with its index number (e.g., '1: Hello'). "
            "Translate each line from its original language to Brazilian Portuguese. "
            "Preserve meaning, tone, names, and cultural references. "
            "Output ONLY the translations, one per line, in the same order, using the format 'index: translation'. "
            "Do not add any extra text, explanations, greetings, or commentary. "
            "Do not ask for more input. Just output the numbered translations."
        )

        user_content = (
            "Traduza as seguintes legendas para português do Brasil. "
            "Responda apenas com o número e a tradução, uma por linha, no formato 'índice: tradução'.\n\n"
            f"{full_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": 0,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_new_tokens,
                "stop": []   # sem stop extra
            },
        }

        url = f"{self.base_url}/api/chat"

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    full_response = []
                    info("Traduzindo SRT completo (streaming):")
                    for line in resp:
                        line = line.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = data.get("message", {}).get("content", "")
                        if content:
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            full_response.append(content)
                        if data.get("done", False):
                            break
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                    decoded = "".join(full_response)
                    # Parseia o resultado: espera linhas no formato "índice: tradução"
                    translations: dict[int, str] = {}
                    for line in decoded.splitlines():
                        line = line.strip()
                        match = re.match(r"^(\d+)\s*:\s*(.*)$", line)
                        if match:
                            idx = int(match.group(1))
                            trans = clean_text(match.group(2))
                            translations[idx] = trans
                        else:
                            # Tenta capturar linhas soltas como fallback (ex: sem índice)
                            pass

                    # Se não conseguiu parsear pelo padrão, assume que a resposta é uma sequência de linhas
                    # na ordem correta, sem os números.
                    if not translations and len(decoded.splitlines()) == len(segments):
                        lines = decoded.splitlines()
                        for i, seg in enumerate(segments):
                            translations[seg.index] = clean_text(lines[i])

                    # Preenche a lista de traduções na ordem dos segmentos
                    result: list[str] = []
                    for seg in segments:
                        trans = translations.get(seg.index, "")
                        if not trans:
                            info(f"Aviso: tradução não encontrada para o segmento {seg.index}. Mantendo original.")
                            trans = seg.text  # fallback
                        result.append(trans)
                    return result

            except (urllib.error.URLError, Exception) as exc:
                if attempt < self.max_retries:
                    info(f"Tentativa {attempt}/{self.max_retries} falhou. Retentando em {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    if isinstance(exc, urllib.error.URLError):
                        die(
                            f"Falha ao acessar o Ollama em {self.base_url}. "
                            f"Verifique se o Ollama está rodando e se o modelo '{self.model_name}' foi baixado. "
                            f"Detalhe: {exc}"
                        )
                    else:
                        die(f"Erro inesperado ao traduzir com Ollama: {exc}")

        return [seg.text for seg in segments]  # fallback


# ----------------------------
# TTS com edge-tts
# ----------------------------

async def synthesize_tts_async(text: str, out_mp3: Path, voice: str) -> None:
    edge_tts = safe_import("edge_tts")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(out_mp3))


def synthesize_tts(text: str, out_mp3: Path, voice: str) -> None:
    text = clean_text(text)
    if not text:
        raise ValueError("Texto vazio para TTS.")
    asyncio.run(synthesize_tts_async(text, out_mp3, voice))


# ----------------------------
# Ajuste e montagem da faixa de áudio
# ----------------------------

def probe_duration_seconds(media_path: Path) -> float:
    ensure_command_exists("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    result = run(cmd, capture_output=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def count_audio_streams(media_path: Path) -> int:
    ensure_command_exists("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(media_path),
    ]
    result = run(cmd, capture_output=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return len(lines)


def atempo_chain(rate: float) -> str:
    if rate <= 0:
        raise ValueError("rate deve ser positivo")
    factors: list[float] = []
    remaining = rate
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={f:.8f}" for f in factors)


def fit_audio_to_duration(
    input_audio: Path,
    output_audio: Path,
    target_duration_sec: float,
    *,
    max_speed_ratio: float = 1.35,
) -> None:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(input_audio))
    current_sec = len(audio) / 1000.0
    target_sec = max(0.01, float(target_duration_sec))

    if current_sec <= 0.01:
        silence_ms = max(1, int(target_sec * 1000))
        AudioSegment.silent(duration=silence_ms, frame_rate=48000).export(str(output_audio), format="wav")
        return

    ratio = current_sec / target_sec
    if ratio > 1.02:
        speed = min(max_speed_ratio, ratio)
        with tempfile.TemporaryDirectory() as td:
            sped = Path(td) / "sped.wav"
            ensure_command_exists("ffmpeg")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_audio),
                "-filter:a", atempo_chain(speed),
                str(sped),
            ]
            run(cmd)
            audio = AudioSegment.from_file(str(sped))
            current_sec = len(audio) / 1000.0

    if current_sec < target_sec:
        pad_ms = int(round((target_sec - current_sec) * 1000))
        audio = audio + AudioSegment.silent(duration=pad_ms, frame_rate=audio.frame_rate)

    max_ms = int(round(target_sec * 1000))
    if len(audio) > max_ms:
        audio = audio[:max_ms]

    audio = audio.set_frame_rate(48000).set_channels(2)
    audio.export(str(output_audio), format="wav")


def build_dub_track(
    segments: list[Segment],
    *,
    tts_dir: Path,
    dub_wav: Path,
    voice: str,
    total_duration_sec: float,
) -> None:
    from pydub import AudioSegment

    tts_dir.mkdir(parents=True, exist_ok=True)

    timeline = AudioSegment.silent(
        duration=int(math.ceil(total_duration_sec * 1000)),
        frame_rate=48000
    ).set_channels(2)

    for seg in segments:
        translated = clean_text(seg.translated_text)
        if not translated:
            continue

        raw_tts_mp3 = tts_dir / f"seg_{seg.index:05d}.mp3"
        fitted_wav = tts_dir / f"seg_{seg.index:05d}_fit.wav"

        info(f"TTS segmento {seg.index}/{len(segments)}...")
        synthesize_tts(translated, raw_tts_mp3, voice)

        seg.tts_file = str(raw_tts_mp3)
        fit_audio_to_duration(raw_tts_mp3, fitted_wav, seg.duration)

        tts_audio = AudioSegment.from_file(str(fitted_wav))
        seg.tts_duration = len(tts_audio) / 1000.0

        start_ms = int(round(seg.start * 1000))
        timeline = timeline.overlay(tts_audio, position=start_ms)

    timeline.export(str(dub_wav), format="wav")


def convert_wav_to_m4a(input_wav: Path, output_m4a: Path) -> None:
    ensure_command_exists("ffmpeg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_wav),
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_m4a),
    ]
    info("Convertendo a faixa dublada para AAC/M4A...")
    run(cmd)


def mux_into_mkv(
    original_video: Path,
    dub_audio_m4a: Path,
    output_video: Path,
) -> None:
    ensure_command_exists("ffmpeg")
    original_audio_count = count_audio_streams(original_video)
    dub_audio_index = original_audio_count
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_video),
        "-i", str(dub_audio_m4a),
        "-map", "0",
        "-map", "1:a:0",
        "-c", "copy",
        "-metadata:s:a:%d" % dub_audio_index, "language=por",
        "-metadata:s:a:%d" % dub_audio_index, "title=Dublagem PT-BR",
        str(output_video),
    ]
    info("Muxando MKV final com faixa de áudio adicional...")
    run(cmd)


# ----------------------------
# Pipeline principal
# ----------------------------

def process_video(
    input_video: Path,
    output_video: Path,
    *,
    asr_model: str,
    qwen_model: str,
    ollama_url: str,
    voice: str,
    keep_temp: bool,
    temp_dir: Optional[Path],
    skip_transcription: bool = False,
    original_srt: Optional[Path] = None,
    skip_translation: bool = False,
    translated_srt: Optional[Path] = None,
    device: str = "cuda",  # novo parâmetro
) -> None:
    # Configura o dispositivo (valida e exibe info)
    device = setup_device(device)

    if not input_video.exists():
        die(f"Arquivo de entrada não encontrado: {input_video}")

    if input_video.suffix.lower() != ".mkv":
        die("O vídeo de entrada deve obrigatoriamente ser MKV.")

    ensure_command_exists("ffmpeg")
    ensure_command_exists("ffprobe")

    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    script_dir.mkdir(parents=True, exist_ok=True)

    if temp_dir is None:
        tmp_obj = tempfile.TemporaryDirectory(prefix="redublagem_ai_")
        workdir = Path(tmp_obj.name)
    else:
        workdir = temp_dir
        workdir.mkdir(parents=True, exist_ok=True)
        tmp_obj = None

    try:
        segments: list[Segment] = []

        if skip_transcription:
            if not original_srt or not original_srt.exists():
                die("--skip-transcription requer um SRT original válido via --original-srt")
            info(f"Usando SRT original fornecido: {original_srt}")
            segments = parse_srt(original_srt)
            total_duration_sec = probe_duration_seconds(input_video)
            if total_duration_sec <= 0:
                total_duration_sec = max((s.end for s in segments), default=0.0)
        else:
            audio_wav = workdir / "audio.wav"
            extract_audio(input_video, audio_wav)
            segments = transcribe_with_whisper(audio_wav, asr_model=asr_model, device=device)
            original_srt_path = script_dir / "original.srt"
            write_srt(original_srt_path, segments, translated=False)
            info(f"SRT original salvo em: {original_srt_path}")

        if not segments:
            die("Nenhum segmento disponível após a etapa de transcrição.")

        if not skip_translation:
            info("Traduzindo SRT completo com Ollama...")
            translator = OllamaTranslator(
                model_name=qwen_model,
                base_url=ollama_url,
            )
            translations = translator.translate_full_srt(segments)
            # Atribui as traduções aos segmentos
            for seg, trans in zip(segments, translations):
                seg.translated_text = trans

            translated_srt_path = script_dir / "translated.srt"
            write_srt(translated_srt_path, segments, translated=True)
            info(f"SRT traduzido salvo em: {translated_srt_path}")
        else:
            if not translated_srt or not translated_srt.exists():
                die("--skip-translation requer um SRT traduzido válido via --translated-srt")
            info(f"Usando SRT traduzido fornecido: {translated_srt}")
            translated_segments = parse_srt(translated_srt)
            if skip_transcription:
                segments = translated_segments
            else:
                if len(segments) != len(translated_segments):
                    die(
                        f"Número de segmentos original ({len(segments)}) diferente do traduzido "
                        f"({len(translated_segments)}). Impossível combinar."
                    )
                for orig, trans in zip(segments, translated_segments):
                    orig.translated_text = trans.text

        if not segments or all(not clean_text(s.translated_text) for s in segments):
            die("Nenhum segmento com tradução disponível para gerar áudio.")

        total_duration_sec = probe_duration_seconds(input_video)
        if total_duration_sec <= 0:
            total_duration_sec = max((s.end for s in segments), default=0.0)

        dub_wav = workdir / "dub_track.wav"
        dub_m4a = workdir / "dub_track.m4a"
        tts_dir = workdir / "tts"

        build_dub_track(
            segments,
            tts_dir=tts_dir,
            dub_wav=dub_wav,
            voice=voice,
            total_duration_sec=total_duration_sec,
        )

        convert_wav_to_m4a(dub_wav, dub_m4a)

        output_video.parent.mkdir(parents=True, exist_ok=True)
        mux_into_mkv(input_video, dub_m4a, output_video)

        info(f"Concluído: {output_video}")

        if not keep_temp and temp_dir is None and tmp_obj is not None:
            tmp_obj.cleanup()

    except Exception as exc:
        die(f"Falha no pipeline: {exc}")


# ----------------------------
# CLI
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sistema de redublagem automática com IA para vídeos MKV."
    )
    p.add_argument("--input", "-i", required=True, help="Caminho do vídeo MKV de entrada.")
    p.add_argument("--output", "-o", required=True, help="Caminho do MKV de saída.")
    p.add_argument(
        "--asr-model",
        default="large-v3",
        help="Modelo Whisper.",
    )
    p.add_argument(
        "--qwen-model",
        default=os.getenv("OLLAMA_TRANSLATOR_MODEL", "gemma2:9b"),
        help="Modelo do Ollama para tradução.",
    )
    p.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="URL do servidor Ollama.",
    )
    p.add_argument(
        "--voice",
        default=os.getenv("EDGE_TTS_VOICE", "pt-BR-AntonioNeural"),
        help="Voz do edge-tts.",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Mantém arquivos temporários para depuração.",
    )
    p.add_argument(
        "--temp-dir",
        default=None,
        help="Diretório temporário fixo. Se omitido, será criado automaticamente.",
    )
    p.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Pula a extração de áudio e transcrição. Requer --original-srt.",
    )
    p.add_argument(
        "--original-srt",
        default=None,
        help="Caminho para o SRT original a ser usado quando --skip-transcription for ativado.",
    )
    p.add_argument(
        "--skip-translation",
        action="store_true",
        help="Pula a tradução. Requer --translated-srt.",
    )
    p.add_argument(
        "--translated-srt",
        default=None,
        help="Caminho para o SRT já traduzido a ser usado quando --skip-translation for ativado.",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Força o uso de CPU ou GPU (CUDA/ROCm). Se omitido, pergunta interativamente.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    input_video = Path(args.input).expanduser().resolve()
    output_video = Path(args.output).expanduser().resolve()
    temp_dir = Path(args.temp_dir).expanduser().resolve() if args.temp_dir else None
    original_srt = Path(args.original_srt).expanduser().resolve() if args.original_srt else None
    translated_srt = Path(args.translated_srt).expanduser().resolve() if args.translated_srt else None

    if args.skip_transcription and not original_srt:
        die("--skip-transcription exige que --original-srt seja informado.")
    if args.skip_translation and not translated_srt:
        die("--skip-translation exige que --translated-srt seja informado.")

    # Escolha do dispositivo: se não fornecido via --device, pergunta no terminal
    if args.device:
        device_choice = args.device
    else:
        device_choice = ask_device()

    process_video(
        input_video,
        output_video,
        asr_model=args.asr_model,
        qwen_model=args.qwen_model,
        ollama_url=args.ollama_url,
        voice=args.voice,
        keep_temp=args.keep_temp,
        temp_dir=temp_dir,
        skip_transcription=args.skip_transcription,
        original_srt=original_srt,
        skip_translation=args.skip_translation,
        translated_srt=translated_srt,
        device=device_choice,
    )


if __name__ == "__main__":
    main()