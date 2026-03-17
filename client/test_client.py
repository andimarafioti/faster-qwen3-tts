#!/usr/bin/env python3
"""
Faster Qwen3 TTS Test Client

Unified client for testing both streaming (WebSocket) and non-streaming (HTTP POST) endpoints.
Supports sentence, paragraph, and whole-document splitting modes.
Optionally saves output as WAV files and plays audio.

Usage:
    # Streaming mode (default)
    python client/test_client.py <SERVER_IP> "Your text to synthesize." --voice my_voice
    python client/test_client.py <SERVER_IP> --file document.txt --mode paragraph --voice my_voice

    # Non-streaming mode
    python client/test_client.py <SERVER_IP> "Your text here." --voice my_voice --no-streaming
    python client/test_client.py <SERVER_IP> --file doc.txt --voice my_voice --no-streaming --save output.wav

    # Advanced options
    python client/test_client.py <SERVER_IP> --file doc.txt --voice my_voice --mode whole --no-play
    python client/test_client.py <SERVER_IP> --file doc.txt --voice my_voice --language French --output-dir output/

Requirements:
    pip install websockets numpy sounddevice requests
"""

import asyncio
import argparse
import csv
import json
import os
import re
import struct
import sys
import time
from datetime import datetime
from typing import List, Optional
import numpy as np
import requests

# Optional dependencies
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def split_by_sentence(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'([.!?]+)\s+', text)

    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i + 1])
        else:
            result.append(sentences[i])

    if len(sentences) % 2 == 1 and len(sentences) > 1:
        result.append(sentences[-1])
    elif len(sentences) == 1:
        result = [sentences[0]]

    return [s.strip() for s in result if s.strip()]


def split_by_paragraph(text: str) -> List[str]:
    """Split text on line breaks."""
    paragraphs = text.split('\n')
    result = []
    for p in paragraphs:
        cleaned = re.sub(r'\s+', ' ', p).strip()
        if cleaned:
            result.append(cleaned)
    return result


def split_whole(text: str) -> List[str]:
    """Return entire text as a single item."""
    cleaned = re.sub(r'\s+', ' ', text).strip()
    if cleaned:
        return [cleaned]
    return []


SPLIT_MODES = {
    "sentence": split_by_sentence,
    "paragraph": split_by_paragraph,
    "whole": split_whole,
}


def create_wav_file(audio_data: np.ndarray, sample_rate: int, filepath: str):
    """Write a numpy float32 audio array to a WAV file."""
    pcm = (audio_data * 32767).astype(np.int16)
    num_samples = len(pcm)
    data_size = num_samples * 2
    byte_rate = sample_rate * 2
    block_align = 2

    with open(filepath, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", data_size + 36))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", 1))   # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", 16))  # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm.tobytes())


def pcm_float32_to_wav(pcm_bytes: bytes, sample_rate: int, filepath: str):
    """Write raw PCM float32 bytes to a WAV file (as int16)."""
    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
    create_wav_file(audio, sample_rate, filepath)


class AudioPlayer:
    """Plays complete audio buffers with smooth playback."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.segments_played = 0

    def play_segment(self, audio_data: np.ndarray, segment_num: int, all_chunks_received_time: float):
        """Play a complete segment audio buffer (blocking)."""
        if not HAS_SOUNDDEVICE:
            print(f"  [Warning] Cannot play audio: pip install sounddevice")
            return

        play_start_time = time.time()
        play_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        buffer_delay_ms = (play_start_time - all_chunks_received_time) * 1000
        audio_duration = len(audio_data) / self.sample_rate

        print(f"  [{play_timestamp}] ▶ Playing segment {segment_num} "
              f"({audio_duration:.2f}s audio, buffer delay: {buffer_delay_ms:.0f}ms)")

        sd.play(audio_data, self.sample_rate)
        sd.wait()

        play_end_time = time.time()
        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        actual_duration = play_end_time - play_start_time

        print(f"  [{end_timestamp}] ■ Finished playing segment {segment_num} "
              f"(actual: {actual_duration:.2f}s)")

        self.segments_played += 1

    def stop(self):
        """Stop the audio player."""
        if self.segments_played > 0:
            print(f"\n  [Audio Stats] Total segments played: {self.segments_played}")


# ============================================================
# Streaming Mode (WebSocket)
# ============================================================

async def stream_segment_tts(
    websocket,
    segment: str,
    audio_player: AudioPlayer,
    language: str = "English",
    voice: Optional[str] = None,
    uid: Optional[str] = None,
):
    """
    Stream a single segment to the server, buffer all chunks, then play complete audio.

    Returns timing metrics and audio buffer.
    """
    send_time = time.time()
    send_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"  [{send_timestamp}] → Sending: '{segment[:60]}{'...' if len(segment) > 60 else ''}'")

    request = {
        "text": segment,
        "language": language,
    }
    if voice is not None:
        request["voice"] = voice
    if uid is not None:
        request["uid"] = uid

    await websocket.send(json.dumps(request))

    audio_chunks = []
    chunk_count = 0
    first_audio_time = None
    sample_rate = 24000

    while True:
        try:
            response = await websocket.recv()
            recv_time = time.time()
            recv_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            data = json.loads(response)

            if data["type"] == "start":
                sample_rate = data.get("sample_rate", 24000)
                audio_player.sample_rate = sample_rate
                voice_used = data.get("voice", "default")
                print(f"  [{recv_timestamp}] ← Server started (sample_rate={sample_rate}Hz, voice={voice_used})")

            elif data["type"] == "audio":
                chunk_count += 1

                if first_audio_time is None:
                    first_audio_time = recv_time
                    ttfa = (first_audio_time - send_time) * 1000
                    print(f"  [{recv_timestamp}] ← First audio chunk received (TTFA: {ttfa:.0f}ms)")

                audio_data = np.array(data["data"], dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32767.0
                audio_chunks.append(audio_float)

            elif data["type"] == "end":
                total_time = data.get("total_time_seconds", 0)
                total_chunks = data.get("total_chunks", 0)
                end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                all_chunks_received_time = recv_time
                total_elapsed = (recv_time - send_time) * 1000
                ttfa_ms = (first_audio_time - send_time) * 1000 if first_audio_time else 0

                print(f"  [{end_timestamp}] ✓ All {total_chunks} chunks received in {total_time:.2f}s")
                print(f"    Timing: TTFA={ttfa_ms:.0f}ms, Total={total_elapsed:.0f}ms")

                if audio_chunks:
                    complete_audio = np.concatenate(audio_chunks)
                else:
                    complete_audio = np.array([], dtype=np.float32)

                return {
                    "total_time": total_time,
                    "ttfa_ms": ttfa_ms,
                    "total_elapsed_ms": total_elapsed,
                    "chunks": total_chunks,
                    "audio": complete_audio,
                    "all_chunks_received_time": all_chunks_received_time
                }

            elif data["type"] == "error":
                error_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"  [{error_timestamp}] ✗ Error: {data.get('message', 'Unknown error')}")
                return {
                    "total_time": 0,
                    "ttfa_ms": 0,
                    "total_elapsed_ms": 0,
                    "chunks": 0,
                    "audio": np.array([], dtype=np.float32),
                    "all_chunks_received_time": time.time()
                }

        except websockets.exceptions.ConnectionClosed:
            error_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"  [{error_timestamp}] ✗ Connection closed unexpectedly")
            return {
                "total_time": 0,
                "ttfa_ms": 0,
                "total_elapsed_ms": 0,
                "chunks": 0,
                "audio": np.array([], dtype=np.float32),
                "all_chunks_received_time": time.time()
            }


async def run_streaming_mode(
    server_ip: str,
    text: str,
    port: int = 30800,
    language: str = "English",
    voice: Optional[str] = None,
    uid: Optional[str] = None,
    mode: str = "sentence",
    output_dir: Optional[str] = None,
    play_audio: bool = True,
    delay: float = 0,
    resume: bool = False,
):
    """Stream text to TTS server using WebSocket with concurrent playback."""
    if not HAS_WEBSOCKETS:
        print("Error: websockets not installed. Install with: pip install websockets")
        return

    segments = SPLIT_MODES[mode](text)

    if not segments:
        print("Error: No segments found in input text")
        return

    wav_dir = None
    if output_dir:
        wav_dir = os.path.join(output_dir, "wav")
        os.makedirs(wav_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Faster Qwen3 TTS Streaming Client")
    print(f"{'='*70}")
    print(f"Server:    {server_ip}:{port}")
    print(f"Mode:      {mode} (streaming)")
    print(f"Segments:  {len(segments)}")
    print(f"Language:  {language}")
    print(f"Voice:     {voice or '(default)'}")
    print(f"Play:      {play_audio}")
    if wav_dir:
        print(f"Output:    {wav_dir}/")
    print(f"{'='*70}\n")

    audio_player = AudioPlayer()

    try:
        ws_url = f"ws://{server_ip}:{port}/tts/ws"
        print(f"Connecting to {ws_url}...\n")

        async with websockets.connect(ws_url, ping_timeout=None, ping_interval=None) as websocket:
            print("✓ Connected!\n")

            total_time = 0
            total_ttfa = 0
            total_elapsed = 0
            total_chunks = 0
            segment_metrics = []
            skipped_resume = 0

            for i, segment in enumerate(segments, 1):
                if resume and wav_dir:
                    wav_path = os.path.join(wav_dir, f"{i:04d}.wav")
                    if os.path.exists(wav_path):
                        skipped_resume += 1
                        continue

                if i > 1 and delay > 0:
                    await asyncio.sleep(delay)

                print(f"[Segment {i}/{len(segments)}]")

                metrics = await stream_segment_tts(
                    websocket,
                    segment,
                    audio_player,
                    language=language,
                    voice=voice,
                    uid=uid,
                )

                segment_metrics.append(metrics)
                total_time += metrics["total_time"]
                total_ttfa += metrics["ttfa_ms"]
                total_elapsed += metrics["total_elapsed_ms"]
                total_chunks += metrics["chunks"]

                if wav_dir and len(metrics["audio"]) > 0:
                    wav_path = os.path.join(wav_dir, f"{i:04d}.wav")
                    create_wav_file(metrics["audio"], audio_player.sample_rate, wav_path)
                    print(f"  Saved: {wav_path}")

                if play_audio and len(metrics["audio"]) > 0:
                    audio_player.play_segment(
                        metrics["audio"],
                        i,
                        metrics["all_chunks_received_time"]
                    )

                print()

            print("All segments processed!")

            print(f"\n{'='*70}")
            print(f"Summary:")
            print(f"  Total segments: {len(segments)}")
            if skipped_resume:
                print(f"  Resumed (skipped): {skipped_resume}")
            print(f"  Total chunks: {total_chunks}")
            print(f"  Total generation time: {total_time:.2f}s")
            print(f"  Average per segment: {total_time/len(segments):.2f}s")
            print(f"  Average TTFA: {total_ttfa/len(segments):.0f}ms")
            print(f"  Average latency: {total_elapsed/len(segments):.0f}ms")
            if wav_dir:
                print(f"  WAV files: {wav_dir}/")
            print(f"{'='*70}\n")

            if output_dir:
                report_path = os.path.join(output_dir, "report.csv")
                with open(report_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["index", "text", "ttfa_ms", "total_elapsed_ms"])
                    for i, (seg, m) in enumerate(zip(segments, segment_metrics), 1):
                        writer.writerow([i, seg, f"{m['ttfa_ms']:.0f}", f"{m['total_elapsed_ms']:.0f}"])
                print(f"  Report: {report_path}")

    except websockets.exceptions.WebSocketException as e:
        print(f"\n✗ WebSocket error: {e}")
        print(f"  Make sure the server is running at {server_ip}:{port}")
    except ConnectionRefusedError:
        print(f"\n✗ Connection refused: {server_ip}:{port}")
        print(f"  Make sure the server is running and accessible")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
    finally:
        audio_player.stop()


# ============================================================
# Non-Streaming Mode (HTTP POST)
# ============================================================

def run_non_streaming_mode(
    server_ip: str,
    text: str,
    port: int = 30800,
    language: str = "English",
    voice: Optional[str] = None,
    uid: Optional[str] = None,
    mode: str = "whole",
    output_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    play_audio: bool = True,
    delay: float = 0,
    resume: bool = False,
):
    """Send text to TTS server using HTTP POST endpoint."""
    segments = SPLIT_MODES[mode](text)

    if not segments:
        print("Error: No segments found in input text")
        return

    wav_dir = None
    if output_dir:
        wav_dir = os.path.join(output_dir, "wav")
        os.makedirs(wav_dir, exist_ok=True)

    url = f"http://{server_ip}:{port}/tts"

    print(f"\n{'='*70}")
    print(f"Faster Qwen3 TTS Non-Streaming Client")
    print(f"{'='*70}")
    print(f"Server:    {server_ip}:{port}")
    print(f"Mode:      {mode} (non-streaming)")
    print(f"Segments:  {len(segments)}")
    print(f"Language:  {language}")
    print(f"Voice:     {voice or '(required)'}")
    if uid:
        print(f"UID:       {uid}")
    if wav_dir:
        print(f"Output:    {wav_dir}/")
    print(f"{'='*70}\n")

    total_gen_time = 0
    total_audio_duration = 0
    total_elapsed = 0
    segment_metrics = []

    session = requests.Session()
    skipped_resume = 0

    for i, segment in enumerate(segments, 1):
        if resume and wav_dir:
            wav_path = os.path.join(wav_dir, f"{i:04d}.wav")
            if os.path.exists(wav_path):
                skipped_resume += 1
                continue

        if i > 1 and delay > 0:
            time.sleep(delay)

        print(f"[Segment {i}/{len(segments)}]")
        print(f"  text: {segment[:80]}{'...' if len(segment) > 80 else ''}")

        payload = {
            "text": segment,
            "language": language,
            "voice": voice,
        }
        if uid:
            payload["uid"] = uid

        start = time.time()
        try:
            resp = session.post(url, json=payload, timeout=300)
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"  Error: {e}")
            print()
            continue

        elapsed = time.time() - start

        if resp.status_code != 200:
            print(f"  Error {resp.status_code}: {resp.text}")
            print()
            continue

        sample_rate = int(resp.headers.get("X-Sample-Rate", 24000))
        audio_duration = float(resp.headers.get("X-Audio-Duration", 0))
        gen_time = float(resp.headers.get("X-Generation-Time", 0))

        pcm_bytes = resp.content
        audio = np.frombuffer(pcm_bytes, dtype=np.float32)

        total_gen_time += gen_time
        total_audio_duration += audio_duration
        total_elapsed += elapsed
        segment_metrics.append({
            "text": segment,
            "gen_time": gen_time,
            "audio_duration": audio_duration,
            "round_trip": elapsed,
        })

        print(f"  audio: {audio_duration}s | gen: {gen_time}s | round-trip: {elapsed:.2f}s")

        if wav_dir:
            wav_path = os.path.join(wav_dir, f"{i:04d}.wav")
            pcm_float32_to_wav(pcm_bytes, sample_rate, wav_path)
            print(f"  saved: {wav_path}")
        elif save_path and len(segments) == 1:
            pcm_float32_to_wav(pcm_bytes, sample_rate, save_path)
            print(f"  saved: {save_path}")

        if play_audio and len(audio) > 0:
            if HAS_SOUNDDEVICE:
                print(f"  playing {audio_duration}s...")
                sd.play(audio, sample_rate)
                sd.wait()
            else:
                print("  Cannot play: pip install sounddevice")

        print()

    print(f"{'='*70}")
    print(f"Summary:")
    print(f"  Total segments:       {len(segments)}")
    if skipped_resume:
        print(f"  Resumed (skipped):    {skipped_resume}")
    print(f"  Total audio duration: {total_audio_duration:.2f}s")
    print(f"  Total generation:     {total_gen_time:.2f}s")
    print(f"  Total round-trip:     {total_elapsed:.2f}s")
    if wav_dir:
        print(f"  WAV files:            {wav_dir}/")
    print(f"{'='*70}\n")

    if output_dir and segment_metrics:
        report_path = os.path.join(output_dir, "report.csv")
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "text", "gen_time_s", "audio_duration_s", "round_trip_s"])
            for i, m in enumerate(segment_metrics, 1):
                writer.writerow([
                    i,
                    m["text"],
                    f"{m['gen_time']:.2f}",
                    f"{m['audio_duration']:.2f}",
                    f"{m['round_trip']:.2f}",
                ])
        print(f"  Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Faster Qwen3 TTS Test Client - Streaming and non-streaming modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Streaming mode (default)
  python client/test_client.py <SERVER_IP> "Hello world." --voice my_voice
  python client/test_client.py <SERVER_IP> --file doc.txt --mode paragraph --voice my_voice

  # Non-streaming mode
  python client/test_client.py <SERVER_IP> "Hello world." --voice my_voice --no-streaming
  python client/test_client.py <SERVER_IP> --file doc.txt --voice my_voice --no-streaming --save output.wav

  # Advanced
  python client/test_client.py <SERVER_IP> --file doc.txt --voice my_voice --mode whole --no-play
  python client/test_client.py <SERVER_IP> --file doc.txt --voice my_voice --language French --output-dir output/
        """
    )

    parser.add_argument(
        "server_ip",
        help="Public IP address or hostname of the TTS server"
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize. Use --file to read from file instead."
    )

    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to text file to synthesize"
    )

    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        required=True,
        help="Voice name to use (required)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=30800,
        help="Server port (default: 30800 for Kubernetes NodePort)"
    )

    parser.add_argument(
        "--language",
        "-l",
        default="English",
        help="Language for TTS (default: English)"
    )

    parser.add_argument(
        "--uid",
        default=None,
        help="User ID for user-specific voices (optional)"
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["sentence", "paragraph", "whole"],
        default="sentence",
        help="Splitting mode (default: sentence for streaming, whole for non-streaming)"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to save WAV files (saved to <output-dir>/wav/)"
    )

    parser.add_argument(
        "--save",
        "-s",
        default=None,
        help="Save output to a single WAV file (for single text input, non-streaming mode)"
    )

    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip audio playback"
    )

    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0,
        help="Delay in seconds between requests (default: 0)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip segments whose WAV files already exist (requires --output-dir)"
    )

    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Use non-streaming mode (HTTP POST) instead of streaming (WebSocket)"
    )

    args = parser.parse_args()

    if not args.text and not args.file:
        print("Error: Provide text or --file")
        parser.print_help()
        sys.exit(1)

    if args.text and args.file:
        print("Error: Cannot specify both text and --file. Choose one.")
        sys.exit(1)

    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                lines = [l for l in f if not l.lstrip().startswith("#")]
                text = "".join(lines).strip()
            if not text:
                print(f"Error: File '{args.file}' is empty")
                sys.exit(1)
            print(f"Reading text from: {args.file}")
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
    else:
        text = args.text

    # Adjust default mode based on streaming setting
    if args.mode == "sentence" and args.no_streaming:
        # Override default for non-streaming
        mode = "whole"
    else:
        mode = args.mode

    play_audio = not args.no_play

    try:
        if args.no_streaming:
            # Non-streaming mode
            run_non_streaming_mode(
                server_ip=args.server_ip,
                text=text,
                port=args.port,
                language=args.language,
                voice=args.voice,
                uid=args.uid,
                mode=mode,
                output_dir=args.output_dir,
                save_path=args.save,
                play_audio=play_audio,
                delay=args.delay,
                resume=args.resume,
            )
        else:
            # Streaming mode
            asyncio.run(run_streaming_mode(
                server_ip=args.server_ip,
                text=text,
                port=args.port,
                language=args.language,
                voice=args.voice,
                uid=args.uid,
                mode=mode,
                output_dir=args.output_dir,
                play_audio=play_audio,
                delay=args.delay,
                resume=args.resume,
            ))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except ConnectionRefusedError:
        print(f"\nError: Connection refused at {args.server_ip}:{args.port}")
        sys.exit(1)


if __name__ == "__main__":
    main()
