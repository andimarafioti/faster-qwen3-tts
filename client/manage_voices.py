#!/usr/bin/env python3
"""
Voice Management Utility for Faster Qwen3 TTS Server

Helper script to list, upload, and test voices on the TTS server.

Usage:
    # List all voices
    python manage_voices.py <SERVER_IP> list

    # Upload a new voice from WAV + transcript files
    python manage_voices.py <SERVER_IP> upload my_voice voice.wav --transcript transcript.txt

    # Test synthesis with a specific voice
    python manage_voices.py <SERVER_IP> test my_custom_voice "Hello world"

Examples:
    python manage_voices.py <PUBLIC IP> list
    python manage_voices.py <PUBLIC IP> upload john_voice sample.wav --transcript sample.txt
    python manage_voices.py <PUBLIC IP> test en_female "Testing female voice"
"""

import argparse
import sys
import requests
import json
import os
from typing import Optional


def list_voices(server_ip: str, port: int, uid: Optional[str] = None):
    """List all available voices on the server."""
    url = f"http://{server_ip}:{port}/voices"

    params = {}
    if uid:
        params["uid"] = uid

    try:
        print(f"Fetching voices from {url}...\n")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        print("=" * 70)
        print("VOICE STATUS")
        print("=" * 70)

        system_voices = data.get("system_voices", [])
        user_voices = data.get("user_voices", [])

        print(f"\n✓ System Voices ({len(system_voices)}):")
        if system_voices:
            for voice in system_voices:
                print(f"  • {voice}")
        else:
            print("  (none)")

        if uid:
            print(f"\n✓ User Voices for UID '{uid}' ({len(user_voices)}):")
            if user_voices:
                for voice_info in user_voices:
                    voice_id = voice_info.get("id", "unknown")
                    duration = voice_info.get("duration", 0)
                    print(f"  • {voice_id} (duration: {duration}s)")
            else:
                print("  (none)")

        total = len(system_voices) + len(user_voices)
        print(f"\nTotal voices: {total}")
        print("=" * 70)
        print("\nNote: All voices are automatically loaded and ready to use.")
        print("=" * 70)

        return True

    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {server_ip}:{port}")
        print(f"  Make sure the server is running and accessible")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Error: Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        return False


def upload_voice(
    server_ip: str,
    port: int,
    voice_name: str,
    wav_file: str,
    txt_file: str,
    uid: Optional[str] = None
):
    """Upload a new voice to the server."""
    url = f"http://{server_ip}:{port}/voices/upload"

    # Validate files exist
    if not wav_file.endswith('.wav'):
        print(f"✗ Error: Audio file must be a .wav file")
        return False

    if not os.path.exists(wav_file):
        print(f"✗ Error: WAV file not found: {wav_file}")
        return False

    if not os.path.exists(txt_file):
        print(f"✗ Error: Transcript file not found: {txt_file}")
        return False

    try:
        print(f"Uploading voice '{voice_name}'...")
        print(f"  WAV file: {wav_file}")
        print(f"  Transcript: {txt_file}")
        if uid:
            print(f"  UID: {uid}")
        print()

        # Prepare multipart form data
        data = {
            'voice_name': voice_name
        }
        if uid:
            data['uid'] = uid

        files = {
            'wav_file': (os.path.basename(wav_file), open(wav_file, 'rb'), 'audio/wav'),
            'txt_file': (os.path.basename(txt_file), open(txt_file, 'rb'), 'text/plain'),
        }

        response = requests.post(url, data=data, files=files, timeout=60)
        response.raise_for_status()

        # Close files
        for file_tuple in files.values():
            if hasattr(file_tuple[1], 'close'):
                file_tuple[1].close()

        result = response.json()

        status = result.get("status")
        if status == "success":
            load_time = result.get("load_time_seconds", 0)
            duration = result.get("duration", 0)
            message = result.get("message", "")

            print("=" * 70)
            print("✓ UPLOAD SUCCESSFUL")
            print("=" * 70)
            print(f"Voice name: {voice_name}")
            print(f"Duration: {duration}s")
            print(f"Processing time: {load_time:.2f}s")
            print(f"Message: {message}")
            print("=" * 70)
            print(f"\n✓ Voice '{voice_name}' is now loaded and ready to use!")
            print(f"Test it with: python manage_voices.py {server_ip} test {voice_name} \"Hello world\"")

            return True
        else:
            print(f"✗ Unexpected response: {result}")
            return False

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            detail = e.response.json().get("detail", "Unknown error")
            print(f"✗ Bad request: {detail}")
        elif e.response.status_code == 500:
            detail = e.response.json().get("detail", "Unknown error")
            print(f"✗ Server error: {detail}")
        else:
            print(f"✗ HTTP error {e.response.status_code}: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {server_ip}:{port}")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Error: Request timed out (voice upload can take 10-60 seconds)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_voice(
    server_ip: str,
    port: int,
    voice_name: str,
    text: str,
    language: str = "English",
    uid: Optional[str] = None
):
    """Test synthesis with a specific voice."""
    url = f"http://{server_ip}:{port}/tts/stream"

    payload = {
        "text": text,
        "language": language,
        "voice": voice_name
    }
    if uid:
        payload["uid"] = uid

    try:
        print(f"Testing voice '{voice_name}' with text: '{text}'\n")
        response = requests.post(url, json=payload, timeout=60, stream=True)
        response.raise_for_status()

        # Save to file
        output_file = f"test_{voice_name}.wav"
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("=" * 70)
        print("✓ SUCCESS")
        print("=" * 70)
        print(f"Voice: {voice_name}")
        print(f"Text: {text}")
        print(f"Language: {language}")
        print(f"Output: {output_file}")
        print("=" * 70)
        print(f"\nAudio saved to {output_file}")
        print(f"Play with: play {output_file}")

        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            try:
                detail = e.response.json().get("detail", "Unknown error")
            except:
                detail = e.response.text
            print(f"✗ Voice not found: {detail}")
            print(f"\nVoice '{voice_name}' may not be loaded.")
            print(f"Run: python manage_voices.py {server_ip} list")
        elif e.response.status_code == 400:
            try:
                detail = e.response.json().get("detail", "Unknown error")
            except:
                detail = e.response.text
            print(f"✗ Bad request: {detail}")
        else:
            print(f"✗ HTTP error: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {server_ip}:{port}")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Error: Request timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Voice Management Utility for Faster Qwen3 TTS Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all voices (system and user-specific)
  python manage_voices.py <PUBLIC IP> list

  # List voices for specific user
  python manage_voices.py <PUBLIC IP> list --uid user123

  # Upload a new voice with transcript (required)
  python manage_voices.py <PUBLIC IP> upload john_voice sample.wav --transcript transcript.txt

  # Upload user-specific voice
  python manage_voices.py <PUBLIC IP> upload jane_voice voice.wav --transcript text.txt --uid user123

  # Test a voice with sample text
  python manage_voices.py <PUBLIC IP> test en_female "Hello, this is a test"

  # Test with different language
  python manage_voices.py <PUBLIC IP> test fr_male "Bonjour" --language French

  # Test user-specific voice
  python manage_voices.py <PUBLIC IP> test my_voice "Testing" --uid user123
        """
    )

    parser.add_argument(
        "server_ip",
        help="Public IP address or hostname of the TTS server"
    )

    parser.add_argument(
        "command",
        choices=["list", "upload", "test"],
        help="Command to execute: list (show voices), upload (upload new voice), test (synthesize with voice)"
    )

    parser.add_argument(
        "voice_name",
        nargs="?",
        help="Voice name (required for 'upload' and 'test' commands)"
    )

    parser.add_argument(
        "wav_file",
        nargs="?",
        help="Path to WAV file (required for 'upload' command)"
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize (required for 'test' command, used as 4th positional arg)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=30800,
        help="Server port (default: 30800)"
    )

    parser.add_argument(
        "--language",
        "-l",
        default="English",
        help="Language for synthesis (default: English)"
    )

    parser.add_argument(
        "--transcript",
        "-t",
        required=False,
        help="Path to transcript text file (required for 'upload' command)"
    )

    parser.add_argument(
        "--uid",
        default=None,
        help="User ID for user-specific voices (optional)"
    )

    args = parser.parse_args()

    # Validate command-specific arguments
    if args.command == "upload":
        if not args.voice_name or not args.wav_file:
            print("Error: 'upload' command requires voice_name and wav_file arguments")
            print("\nUsage:")
            print("  python manage_voices.py <SERVER_IP> upload <VOICE_NAME> <WAV_FILE> --transcript <TXT_FILE>")
            parser.print_help()
            sys.exit(1)
        if not args.transcript:
            print("Error: 'upload' command requires --transcript argument")
            print("\nUsage:")
            print("  python manage_voices.py <SERVER_IP> upload <VOICE_NAME> <WAV_FILE> --transcript <TXT_FILE>")
            parser.print_help()
            sys.exit(1)

    if args.command == "test":
        if not args.voice_name:
            print("Error: 'test' command requires voice_name argument")
            parser.print_help()
            sys.exit(1)
        # For test command, text might be in wav_file position or text position
        if not args.text and not args.wav_file:
            print("Error: 'test' command requires text argument")
            parser.print_help()
            sys.exit(1)
        # Move text from wav_file to text if needed
        if not args.text and args.wav_file:
            args.text = args.wav_file

    # Execute command
    print()

    if args.command == "list":
        success = list_voices(args.server_ip, args.port, args.uid)
    elif args.command == "upload":
        success = upload_voice(
            args.server_ip,
            args.port,
            args.voice_name,
            args.wav_file,
            args.transcript,
            args.uid
        )
    elif args.command == "test":
        success = test_voice(
            args.server_ip,
            args.port,
            args.voice_name,
            args.text,
            args.language,
            args.uid
        )

    print()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
