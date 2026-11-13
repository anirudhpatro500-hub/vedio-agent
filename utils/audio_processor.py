import yt_dlp
import os
import sys
import static_ffmpeg
from typing import Any

static_ffmpeg.add_paths()

from pydub import AudioSegment
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts: Any = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "restrictfilenames": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".mp3"
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        mp3_path = download_youtube_audio(source)
        print("Converting to WAV (16kHz mono)...")
        wav_path = convert_to_wav(mp3_path)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]🎬 Video+Agent[/bold cyan]\n"
        "[dim]Paste a YouTube link or local file path[/dim]",
        border_style="cyan"
    ))

    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = Prompt.ask("[bold green]🔗 Paste link or file path[/bold green]")

    chunks = process_input(source)
    for c in chunks:
        console.print(f"  [green]✓[/green] {c}")   