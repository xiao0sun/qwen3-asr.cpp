# Qwen3-ASR.cpp

A high-performance C++ implementation of Qwen3-ASR and Qwen3-ForcedAligner using the GGML tensor library. Optimized for Apple Silicon with Metal GPU acceleration, providing fast speech recognition and word-level timestamp alignment.

## Features

- **Automatic Speech Recognition (ASR)**: Transcribe audio files to text in 30+ languages
- **Forced Alignment**: Align reference text to audio with word-level timestamps
- **Combined Pipeline** (`--transcribe-align`): Automatically runs ASR then alignment with auto language detection
- **Flash Attention**: Uses `ggml_flash_attn_ext()` for fast decoding (3.7x speedup)
- **Metal GPU Acceleration**: Optimized for Apple Silicon with dual CPU+Metal backend
- **Accelerate/vDSP**: Highly optimized mel spectrogram computation (45x speedup)
- **mmap Weight Loading**: Zero-copy GPU transfer for fast model initialization
- **F16 KV Cache**: Reduced memory bandwidth with half-precision key-value cache
- **Language-Aware Alignment Units**: Korean dictionary splitting, Chinese/Japanese UTF-8 character splitting, whitespace splitting for other languages
- **Quantization Support**: Q8_0 quantization for reduced memory usage (~40% smaller)
- **Persistent Worker Mode**: Keep the ASR model loaded and serve JSONL transcription requests over stdin/stdout
- **Pure C++20**: No Python runtime required for inference

## Supported Models

| Model | Size | Description |
|-------|------|-------------|
| `qwen3-asr-0.6b-f16.gguf` | ~1.8 GB | ASR model, F16 precision |
| `qwen3-asr-0.6b-q8_0.gguf` | ~1.3 GB | ASR model, Q8_0 quantized |
| `qwen3-asr-1.7b-f16.gguf` | ~3.4 GB | Larger ASR model, F16 precision |
| `qwen3-asr-1.7b-q4_1.gguf` | ~2.3 GB | Larger ASR model, Q4_1 quantized |
| `qwen3-forced-aligner-0.6b-f16.gguf` | ~1.8 GB | Forced alignment model |

## Requirements

- CMake 3.14+
- C++20 compatible compiler
- Apple Silicon recommended (Metal GPU support)
- GGML library (included as submodule)

## Building

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/xiao0sun/qwen3-asr.cpp.git
cd qwen3-asr.cpp

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)
```

On Linux, replace `$(sysctl -n hw.ncpu)` with `$(nproc)`.

## Quick Start

### 1. Transcription (ASR)

Transcribe audio files to text:

```bash
# Basic transcription
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav

# Larger 1.7B ASR model (recommended when you can afford the memory)
./build/qwen3-asr-cli -m models/qwen3-asr-1.7b-q4_1.gguf -f audio.wav

# With quantized model (faster, less memory)
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-q8_0.gguf -f audio.wav

# Save output to file
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav -o transcript.txt

# Multi-threaded processing
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav -t 8
```

### 2. Forced Alignment

Align reference text to audio with word-level timestamps:

```bash
# Basic alignment
./build/qwen3-asr-cli \
  -m models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f audio.wav \
  --align \
  --text "transcript text" \
  --lang korean

# Japanese and Chinese are split into UTF-8 characters for stable timestamps
./build/qwen3-asr-cli \
  -m models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f ja.wav \
  --align \
  --text "うちの中学は弁当制で持っていけない場合は50円の学校販売のパンを買う。" \
  --lang japanese

# Save alignment to JSON file
./build/qwen3-asr-cli \
  -m models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f audio.wav \
  --align \
  --text "Hello world" \
  -o alignment.json
```

### 3. Combined Pipeline (Transcribe + Align)

Automatically transcribe and then align the result (recommended):

```bash
./build/qwen3-asr-cli \
  -m models/qwen3-asr-1.7b-q4_1.gguf \
  --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f audio.wav \
  --transcribe-align
```

This mode automatically:
- Runs ASR to get the transcript
- Detects the language from the ASR output
- Runs forced alignment with the detected language
- Outputs word-level timestamps as JSON

### 4. Persistent Worker Mode

Worker mode keeps the ASR model loaded and processes one JSON request per line from stdin:

```bash
./build/qwen3-asr-cli \
  -m models/qwen3-asr-1.7b-q4_1.gguf \
  --worker \
  --lang japanese
```

The worker first prints a ready message:

```json
{"type":"ready","ok":true}
```

Then send JSONL requests:

```json
{"id":"1","audio":"sample.wav","language":"japanese","threads":4,"max_tokens":512}
{"id":"2","shutdown":true}
```

Successful transcription responses look like:

```json
{"id":"1","ok":true,"text":"transcript text","language":"japanese"}
```

### Output Formats

**Transcription** outputs plain text:
```
안녕하세요 여러분 오늘은...
```

**Forced Alignment** outputs JSON with word-level timestamps:
```json
{
  "words": [
    {"word": "안녕하세요", "start": 0.000, "end": 0.480},
    {"word": "여러분", "start": 0.480, "end": 0.880},
    {"word": "오늘은", "start": 0.880, "end": 1.200}
  ]
}
```

Use `-osrt` or `--output-srt` with forced alignment or transcribe+align mode to emit SRT:

```bash
./build/qwen3-asr-cli \
  -m models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f audio.wav \
  --align \
  --text "Hello world" \
  --output-srt \
  -o subtitles.srt
```

## Performance

Benchmark on 92-second Korean audio, Apple M2 Pro (10-core CPU, 16-core GPU):

| Stage | Time |
|-------|------|
| Mel spectrogram | 98 ms |
| Audio encoding | 715 ms |
| Text decoding (323 tokens) | 4,194 ms |
| **ASR Total** | **5,007 ms** |
| Forced alignment (183 words) | 12,998 ms |
| **Combined Total** | **18,005 ms** |

**Memory Usage:** ~247 MB RSS, ~294 MB Metal

### Key Optimizations

- **Flash Attention** (`ggml_flash_attn_ext`): 3.7x decode speedup vs. standard attention
- **Metal GPU Dual Backend**: Automatic scheduling between CPU and GPU for optimal performance
- **mmap + Zero-Copy GPU Transfer**: Fast model loading via `ggml_backend_dev_buffer_from_host_ptr`
- **F16 KV Cache**: Half-precision key-value cache reduces memory bandwidth
- **Selective Logits**: Only compute last token logits for lm_head (saves computation)
- **Weight Tying**: token_embd = output weight (saves memory)
- **vDSP/Accelerate Mel**: 45x speedup for mel spectrogram computation on Apple platforms
- **Language-Aware Alignment Units**: Korean uses the bundled Soynlp-style dictionary, Chinese/Japanese use UTF-8 character units, and other languages use whitespace-delimited words

## Model Conversion

Convert HuggingFace models to GGUF format:

```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Convert ASR model (F16)
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ASR-0.6B \
    --output models/qwen3-asr-0.6b-f16.gguf \
    --type f16

# Convert ASR model (Q8_0 quantized)
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ASR-0.6B \
    --output models/qwen3-asr-0.6b-q8_0.gguf \
    --type q8_0

# Convert 1.7B ASR model (Q4_1 quantized)
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ASR-1.7B \
    --output models/qwen3-asr-1.7b-q4_1.gguf \
    --type q4_1

# Convert ForcedAligner model
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ForcedAligner-0.6B \
    --output models/qwen3-forced-aligner-0.6b-f16.gguf \
    --type f16
```

## Supported Languages

The model supports 30+ languages:

| Language | Code | Language | Code |
|----------|------|----------|------|
| Chinese (Mandarin) | zh | English | en |
| Cantonese | yue | Japanese | ja |
| Korean | ko | German | de |
| French | fr | Spanish | es |
| Italian | it | Portuguese | pt |
| Russian | ru | Arabic | ar |
| Hindi | hi | Thai | th |
| Vietnamese | vi | Indonesian | id |
| Malay | ms | Turkish | tr |
| Polish | pl | Dutch | nl |
| Swedish | sv | Norwegian | no |
| Danish | da | Finnish | fi |
| Greek | el | Czech | cs |
| Hungarian | hu | Romanian | ro |
| Ukrainian | uk | Hebrew | he |

For forced alignment, language affects how reference text is split before timestamp classification:

| Language | Accepted values | Alignment unit strategy |
|----------|-----------------|-------------------------|
| Korean | `korean` | Soynlp-style LTokenizer with `assets/korean_dict_jieba.dict`; falls back to whitespace if the dictionary is unavailable |
| Chinese | `chinese`, `zh` | UTF-8 character units |
| Japanese | `japanese`, `ja`, `jp` | UTF-8 character units |
| Other languages | language name or code | Whitespace-delimited words |

## Audio Requirements

- **Format**: WAV (PCM)
- **Sample rate**: 16 kHz
- **Channels**: Mono
- **Bit depth**: 16-bit

Convert audio with ffmpeg:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## Performance Profiling

Build with timing instrumentation to see detailed breakdowns:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DQWEN3_ASR_TIMING=ON
cmake --build . -j$(sysctl -n hw.ncpu)

# Run with --profile flag
./qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f sample.wav --profile
```

For production builds, omit `-DQWEN3_ASR_TIMING=ON` to remove timing overhead.

## Project Structure

```
qwen3-asr.cpp/
├── src/
│   ├── qwen3_asr.cpp/h       # High-level ASR API
│   ├── forced_aligner.cpp/h  # Forced alignment implementation
│   ├── audio_encoder.cpp/h   # Audio feature encoder
│   ├── text_decoder.cpp/h    # Text decoder (Qwen2 architecture)
│   ├── mel_spectrogram.cpp/h # Mel spectrogram computation
│   ├── audio_injection.cpp/h # Audio-text embedding injection
│   ├── gguf_loader.cpp/h     # GGUF model loading
│   └── timing.h              # Timing instrumentation macros
├── cli/
│   └── main.cpp              # CLI entry point
├── tests/
│   ├── test_mel.cpp          # Mel spectrogram tests
│   ├── test_encoder.cpp      # Audio encoder tests
│   ├── test_decoder.cpp      # Text decoder tests
│   ├── test_forced_aligner_tokenize.cpp # Alignment unit splitting regression test
│   └── reference/            # Reference data for validation
├── scripts/
│   └── convert_hf_to_gguf.py # Model conversion script
├── assets/
│   └── korean_dict_jieba.dict # Korean word dictionary (17,968 words)
├── models/                   # GGUF model files (not tracked in git)
├── ggml/                     # GGML library (git submodule)
└── CMakeLists.txt
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) - Tensor library for machine learning
- [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) - Original ASR models by Alibaba
- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) - Original aligner model by Alibaba

---
