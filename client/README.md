# Faster Qwen3 TTS Client Tools

Python utilities for testing and managing the Faster Qwen3 TTS deployment with real-time audio playback and voice management.

## Tools Included

### 1. `test_client.py` - Unified TTS Client
- **Streaming Mode (WebSocket)**: Real-time audio playback with low latency
- **Non-Streaming Mode (HTTP POST)**: Complete audio generation
- **Sentence/Paragraph/Whole Splitting**: Automatic text segmentation
- **WAV File Saving**: Save outputs to disk
- **Multi-User Support**: User-specific voices via UID parameter

### 2. `manage_voices.py` - Voice Management Utility
- **List Voices**: See all available system and user-specific voices
- **Upload Voices**: Upload new voice samples with transcripts
- **Test Voices**: Quick synthesis test with any voice

## Installation

### Option 1: Using pyenv (Recommended)

Create an isolated Python environment for the clients:

```bash
# Navigate to client directory
cd client

# Create a new virtual environment
python -m venv venv

# Activate the environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Reactivating the environment later:**
```bash
cd client
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

**Deactivating when done:**
```bash
deactivate
```

### Option 2: System-Wide Installation

```bash
cd client
pip install -r requirements.txt
```

### Audio System Dependencies

**macOS:**
```bash
brew install portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**Windows:**
Usually works out of the box with the pip packages.

## Usage

### Voice Management (`manage_voices.py`)

**List all voices:**
```bash
python manage_voices.py <PUBLIC_IP> list
```

Output:
```
======================================================================
VOICE STATUS
======================================================================

✓ System Voices (4):
  • english-male
  • english-female
  • french-male
  • french-female

Total voices: 4
======================================================================
Note: All voices are automatically loaded and ready to use.
======================================================================
```

**List user-specific voices:**
```bash
python manage_voices.py <PUBLIC_IP> list --uid user123
```

**Upload a new voice:**
```bash
python manage_voices.py <PUBLIC_IP> upload my_voice sample.wav --transcript transcript.txt
```

**Upload user-specific voice:**
```bash
python manage_voices.py <PUBLIC_IP> upload my_voice sample.wav --transcript transcript.txt --uid user123
```

**Test a voice:**
```bash
python manage_voices.py <PUBLIC_IP> test english-male "Hello, testing male voice"
```

**Test with different language:**
```bash
python manage_voices.py <PUBLIC_IP> test french-female "Bonjour le monde" --language French
```

### TTS Client (`test_client.py`)

The unified client supports both streaming and non-streaming modes.

#### Streaming Mode (Default - WebSocket)

**Basic usage:**
```bash
python test_client.py <PUBLIC_IP> "Hello world. This is a test." --voice my_voice
```

**Read from file:**
```bash
python test_client.py <PUBLIC_IP> --file document.txt --voice my_voice
```

**Paragraph mode with output saving:**
```bash
python test_client.py <PUBLIC_IP> --file document.txt --mode paragraph --voice my_voice --output-dir output/
```

**No playback (just save WAVs):**
```bash
python test_client.py <PUBLIC_IP> --file document.txt --voice my_voice --no-play --output-dir output/
```

**User-specific voice:**
```bash
python test_client.py <PUBLIC_IP> "Hello world" --voice my_voice --uid user123
```

#### Non-Streaming Mode (HTTP POST)

**Basic usage:**
```bash
python test_client.py <PUBLIC_IP> "Hello world" --voice my_voice --no-streaming --save output.wav
```

**Process entire file:**
```bash
python test_client.py <PUBLIC_IP> --file document.txt --voice my_voice --no-streaming --mode whole --save output.wav
```

**Multiple segments:**
```bash
python test_client.py <PUBLIC_IP> --file document.txt --voice my_voice --no-streaming --mode paragraph --output-dir output/
```

## Command-Line Options

### `manage_voices.py`

```
positional arguments:
  server_ip             Public IP address of the TTS server
  command               Command: list, upload, or test
  voice_name            Voice name (for upload/test commands)
  wav_file              WAV file path (for upload command)
  text                  Text to synthesize (for test command)

options:
  -h, --help            Show help message
  --port PORT           Server port (default: 30800)
  --language LANGUAGE   Language for synthesis (default: English)
  --transcript FILE, -t FILE
                        Transcript text file (required for upload)
  --uid UID             User ID for user-specific voices
```

### `test_client.py`

```
positional arguments:
  server_ip             Public IP address of the TTS server
  text                  Text to synthesize

options:
  -h, --help            Show help message
  --file FILE, -f FILE  Read text from file instead
  --voice VOICE, -v VOICE
                        Voice name to use (required)
  --port PORT           Server port (default: 30800)
  --language LANGUAGE, -l LANGUAGE
                        Language for TTS (default: English)
  --uid UID             User ID for user-specific voices
  --mode MODE, -m MODE  Splitting mode: sentence, paragraph, or whole
  --output-dir DIR, -o DIR
                        Directory to save WAV files
  --save FILE, -s FILE  Save to single WAV file (non-streaming mode)
  --no-play             Skip audio playback
  --delay SECONDS, -d SECONDS
                        Delay between requests (default: 0)
  --resume              Skip segments with existing WAV files
  --no-streaming        Use non-streaming mode (HTTP POST)
```

## Voice Upload Feature

The `upload` command allows you to add new custom voices to the server without SSH access or restarts.

### How Voice Upload Works

1. **Upload Files**: Client uploads `.wav` and `.txt` files to the server
2. **Server Processing**: Server saves files to `/app/voices/` (or `/app/voices/{uid}/` for user-specific)
3. **Automatic Caching**: FasterQwen3TTS model caches speaker embeddings internally on first use
4. **Ready to Use**: Voice is instantly available for synthesis

### Voice File Requirements

- **Audio File**: WAV format, ≥3 seconds, mono or stereo, any sample rate
- **Transcript**: Plain text file with exact words spoken in the audio

### Example Upload Session

```bash
$ python manage_voices.py 35.123.45.67 upload john_voice sample.wav --transcript transcript.txt

Uploading voice 'john_voice'...
  WAV file: sample.wav
  Transcript: transcript.txt

======================================================================
✓ UPLOAD SUCCESSFUL
======================================================================
Voice name: john_voice
Duration: 8.5s
Processing time: 0.45s
Message: Voice 'john_voice' uploaded successfully

======================================================================

✓ Voice 'john_voice' is now loaded and ready to use!
Test it with: python manage_voices.py 35.123.45.67 test john_voice "Hello world"
```

## How Streaming Works

### Streaming Mode (WebSocket)

1. **Text Splitting**: Client splits text into segments (sentence/paragraph/whole)
2. **Segment-by-Segment Streaming**:
   - Segment 1 sent to server
   - Audio chunks arrive via WebSocket
   - All chunks buffered, then played as complete segment
   - While playing, Segment 2 is sent
3. **Buffered Playback**: Each segment plays completely after all chunks received (prevents audio artifacts)

### Progress Display

```
[Segment 1/3]
  [12:34:56.789] → Sending: 'Hello world. This is a test.'
  [12:34:56.825] ← Server started (sample_rate=24000Hz, voice=my_voice)
  [12:34:57.012] ← First audio chunk received (TTFA: 223ms)
  [12:34:57.198] ✓ All 12 chunks received in 0.37s
    Timing: TTFA=223ms, Total=409ms
  [12:34:57.199] ▶ Playing segment 1 (2.1s audio, buffer delay: 1ms)
  [12:34:59.310] ■ Finished playing segment 1 (actual: 2.11s)

[Segment 2/3]
  ...
```

## Output Example

```
======================================================================
Faster Qwen3 TTS Streaming Client
======================================================================
Server:    35.123.45.67:30800
Mode:      sentence (streaming)
Segments:  3
Language:  English
Voice:     my_voice
Play:      True
======================================================================

Connecting to ws://35.123.45.67:30800/tts/ws...

✓ Connected!

[Segment 1/3]
  [12:34:56.789] → Sending: 'Hello world.'
  [12:34:56.825] ← Server started (sample_rate=24000Hz, voice=my_voice)
  [12:34:57.012] ← First audio chunk received (TTFA: 223ms)
  [12:34:57.198] ✓ All 8 chunks received in 0.37s
    Timing: TTFA=223ms, Total=409ms
  Saved: output/wav/0001.wav
  [12:34:57.199] ▶ Playing segment 1 (1.2s audio, buffer delay: 1ms)
  [12:34:58.410] ■ Finished playing segment 1 (actual: 1.21s)

All segments processed!

======================================================================
Summary:
  Total segments: 3
  Total chunks: 24
  Total generation time: 1.12s
  Average per segment: 0.37s
  Average TTFA: 218ms
  Average latency: 402ms
  WAV files: output/wav/
======================================================================

  Report: output/report.csv
```

## Saved Outputs

When using `--output-dir`, the client creates:

```
output/
├── wav/
│   ├── 0001.wav    # Segment 1 audio
│   ├── 0002.wav    # Segment 2 audio
│   └── 0003.wav    # Segment 3 audio
└── report.csv      # Performance metrics
```

**report.csv:**
```csv
index,text,ttfa_ms,total_elapsed_ms
1,"Hello world.",223,409
2,"This is a test.",215,395
3,"Audio streams in real time.",230,420
```

## Troubleshooting

### Connection Refused

```
✗ Connection refused: <PUBLIC_IP>:30800
```

**Solutions:**
- Verify server IP and port are correct
- Check TTS service is running: `curl http://<IP>:30800/health`
- Verify firewall rules allow port 30800
- Check pod status: `ssh ubuntu@<IP> ./tts-info.sh`

### Audio Not Playing

**Linux:**
```bash
sudo apt-get install alsa-utils pulseaudio
```

**macOS:**
- Check system audio is not muted
- Verify output device is selected in System Preferences

**Test audio system:**
```python
import sounddevice as sd
sd.query_devices()  # Should list available audio devices
```

### WebSocket Errors

If WebSocket connection fails:
- Check server is running: `curl http://<IP>:30800/health`
- Verify WebSocket endpoint is accessible
- Check server logs for errors: `ssh ubuntu@<IP> ./tts-logs.sh`

### Voice Not Found

```
✗ Voice 'my_voice' not found
```

**Solutions:**
1. List available voices: `python manage_voices.py <IP> list`
2. Upload the voice: `python manage_voices.py <IP> upload my_voice voice.wav --transcript text.txt`
3. Check voices are in Docker image or uploaded

### Import Errors

```
ModuleNotFoundError: No module named 'websockets'
```

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Advanced Usage

### Resume Long Documents

Process large documents and resume if interrupted:

```bash
# Start processing
python test_client.py <IP> --file long_document.txt --voice my_voice --output-dir output/

# If interrupted, resume (skips already-generated segments)
python test_client.py <IP> --file long_document.txt --voice my_voice --output-dir output/ --resume
```

### Benchmark Performance

```bash
# Time the entire process
time python test_client.py <IP> --file benchmark.txt --voice my_voice --no-play --output-dir results/

# Review metrics in results/report.csv
```

### Different Languages

```bash
# French
python test_client.py <IP> "Bonjour le monde" --language French --voice my_voice

# Chinese
python test_client.py <IP> "你好世界" --language Chinese --voice my_voice

# Using language codes (auto-mapped)
python test_client.py <IP> "Hello" --language en --voice my_voice
```

**Supported languages:**
- English (en), French (fr), Chinese (zh), Japanese (ja), Korean (ko)
- German (de), Spanish (es), Italian (it), Portuguese (pt)
- Russian (ru), Arabic (ar)

### Batch Processing

```bash
# Process multiple texts
for text in "Hello world." "This is sentence two." "And sentence three."; do
  python test_client.py <IP> "$text" --voice my_voice --save "output_${RANDOM}.wav" --no-streaming
  sleep 1
done
```

### Delay Between Requests

Add delay to avoid overwhelming the server:

```bash
python test_client.py <IP> --file document.txt --voice my_voice --delay 2.0
```

## Performance Notes

### Expected Performance (RTX 4090, 1.7B model)

- **TTFA (Time to First Audio)**: ~150-180ms
- **RTF (Real-Time Factor)**: ~4.2x (faster than real-time)
- **Throughput**: ~15ms per codec step
- **Chunk arrival**: Every ~667ms (with default chunk_size=8)

### First Request vs Subsequent

- **First request**: Slower due to CUDA graph capture (adds ~1-2s)
- **Subsequent requests**: Full speed with captured graphs

### Streaming vs Non-Streaming

**Streaming (WebSocket):**
- Lower latency (audio starts playing sooner)
- Better for real-time applications
- Uses WebSocket connection

**Non-Streaming (HTTP POST):**
- Simpler protocol (HTTP)
- Returns complete audio at once
- Better for batch processing

## Client Requirements

Create a `requirements.txt` in the client directory:

```txt
# Core dependencies
requests>=2.31.0
numpy>=1.24.0

# Streaming support (optional, required for WebSocket mode)
websockets>=12.0

# Audio playback (optional)
sounddevice>=0.4.6
```

Install all:
```bash
pip install -r requirements.txt
```

Or install selectively:
```bash
# Minimum (non-streaming mode only)
pip install requests numpy

# Add streaming support
pip install websockets

# Add audio playback
pip install sounddevice
```

## Examples

### Example 1: Quick Test

```bash
python test_client.py 35.123.45.67 "Hello world" --voice english-male
```

### Example 2: Process Document with Saving

```bash
python test_client.py 35.123.45.67 \
  --file my_document.txt \
  --voice english-female \
  --mode paragraph \
  --output-dir output/my_document \
  --language English
```

### Example 3: Non-Streaming Batch

```bash
python test_client.py 35.123.45.67 \
  --file script.txt \
  --voice french-male \
  --language French \
  --no-streaming \
  --mode whole \
  --save script_french.wav
```

### Example 4: Multi-User Workflow

```bash
# Upload user-specific voice
python manage_voices.py 35.123.45.67 upload user_voice voice.wav --transcript text.txt --uid alice

# Use user-specific voice
python test_client.py 35.123.45.67 "Hello from Alice" --voice user_voice --uid alice
```

### Example 5: Resume Long Document

```bash
# Start processing (may be interrupted)
python test_client.py 35.123.45.67 --file long_book.txt --voice my_voice --output-dir book_output/

# Resume later (skips already-generated segments)
python test_client.py 35.123.45.67 --file long_book.txt --voice my_voice --output-dir book_output/ --resume
```

## Voice Upload Workflow

### Preparing Voice Samples

1. **Record or find clean audio**:
   - Duration: 3-30 seconds (5-10 seconds recommended)
   - Quality: Clear speech, minimal background noise
   - Format: WAV file (any sample rate, mono or stereo)

2. **Create exact transcript**:
   - Write exactly what is spoken in the audio
   - Include all words, contractions, punctuation
   - Save as plain text file

3. **Upload to server**:
   ```bash
   python manage_voices.py <IP> upload my_voice sample.wav --transcript transcript.txt
   ```

4. **Test the voice**:
   ```bash
   python manage_voices.py <IP> test my_voice "Hello, this is a test"
   ```

5. **Use in synthesis**:
   ```bash
   python test_client.py <IP> "Your text here" --voice my_voice
   ```

## Related Documentation

- Deployment guide: [../DEPLOYMENT.md](../DEPLOYMENT.md)
- API documentation: Check `/health` endpoint for server status
- faster-qwen3-tts: https://github.com/andimarafioti/faster-qwen3-tts

## Tips and Best Practices

### For Best Performance

1. **Use streaming mode** for real-time applications
2. **Use non-streaming mode** for batch processing
3. **Adjust chunk_size** based on latency requirements (configured on server)
4. **Use sentence mode** for natural pauses between sentences
5. **Use paragraph mode** for document processing
6. **Use whole mode** for short texts or single outputs

### For Best Quality

1. **Provide good voice samples**: Clear, 5-10 seconds, minimal noise
2. **Use accurate transcripts**: Exact match to audio improves quality
3. **Match language**: Use voice samples in the target language when possible

### Debugging

Enable verbose output:
```bash
# Add timing information
python test_client.py <IP> "Test" --voice my_voice 2>&1 | tee debug.log

# Check server logs
ssh ubuntu@<IP> ./tts-logs.sh
```

## Support

For issues:
- Check server health: `curl http://<IP>:30800/health`
- Review client documentation: This file
- Check server logs: `ssh ubuntu@<IP> ./tts-logs.sh`
- faster-qwen3-tts issues: https://github.com/andimarafioti/faster-qwen3-tts/issues
