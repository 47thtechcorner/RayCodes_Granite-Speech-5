<div align="center">
  <a href="https://youtu.be/PW65P3PRDu8">
    <img src="https://img.youtube.com/vi/PW65P3PRDu8/maxresdefault.jpg" alt="IBM's New 2B AI Destroys Whisper Turbo! (Granite Speech 5.0 Review)">
  </a>
  <h3>📺 <a href="https://youtu.be/PW65P3PRDu8">Watch the full video on YouTube</a></h3>
</div>

# 🎙️ IBM Granite Speech 5.0: Multilingual Speech-to-Text & Edge CTC Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-granite--speech--5.0--2b-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/ibm-granite/granite-speech-5.0-2b)
[![Open ASR Rank #2](https://img.shields.io/badge/Open%20ASR-Rank%20%232%20(5.06%25%20WER)-00e676.svg)](#benchmarks)
[![Speed: 19,870 RTFx](https://img.shields.io/badge/Turbo%20CTC-19%2C870%20RTFx-00d2ff.svg)](#benchmarks)

A production-grade Python implementation of IBM's flagship **Granite Speech 5.0 2B** speech-language model and **Granite Speech 5.0 470M Turbo CTC** edge engine. Built for multilingual automatic speech recognition (ASR), 2,000-term keyword biasing, high-throughput edge transcription, and cross-border speech translation (AST).

---

## 🧪 Pipeline Test Cases & Expected Deliverables

| Task # | Test Scenario | AI Superpower Tested | Input Audio & Config | Visual Deliverable |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Townhall Address & Multi-Speaker ASR** | Conformer ASR + 2,000 KWB Bias + Punctuation & Speaker Attribution | `inputs/townhall_speech.wav` (Live presidential union address, 162.4s) + 6 Policy Terms | [`granite_meeting_intelligence.html`](outputs/granite_meeting_intelligence.html) (Speaker-attributed transcript with policy insights) |
| **02** | **Ultra-Fast Edge Streaming ASR** | 470M Turbo CTC Single-Block Reused Conformer Graph | `inputs/multilingual_sample.wav` (Official IBM Granite sample, 24.9s) + Edge CPU execution | [`granite_edge_latency_report.json`](outputs/granite_edge_latency_report.json) (1,847+ RTFx real-time factor & memory report) |
| **03** | **Cross-Border Speech Translation Hub** | Bidirectional AST (French &rarr; English) with Entity Preservation | `inputs/multilingual_sample.wav` (French narration) + 4 Key Literary/Historical Entities | [`granite_cross_border_translation.md`](outputs/granite_cross_border_translation.md) (Segment-by-segment bilingual translation table) |

---

## ⚡ Key Highlights & Benchmarks

| Capability | Granite Speech 5.0 2B | Whisper Large v3 Turbo | Turbo CTC 470M |
| :--- | :---: | :---: | :---: |
| **Open ASR English WER** | **5.06%** (#2 Global) | 5.24% (#4 Global) | 5.51% (Ultra-Fast) |
| **Throughput (RTFx)** | 570 RTFx | 1,184 RTFx | **19,870 RTFx** |
| **Keyword Biasing (KWB)** | **Up to 2,000 terms** | Not supported natively | Not supported |
| **VRAM Footprint (GGUF Q8)** | **2.4 GB VRAM** | 3.1 GB VRAM | **< 1 GB RAM (CPU)** |
| **Native Punctuation / Cap** | Built-in (YaRN 16k) | Separate truecaser | Built-in native conformer |
| **Noise Hardening (FFASR)** | **#4 Worldwide (14.80% WER)** | #14 (18.02% WER) | Conformer block-reused |

---

## 🗺️ Project Architecture & Pipeline

```text
Audio Input (16kHz Mono)
  │
  ├──► [1] Conformer CTC Encoder (16 Layers, 4s Block-Attention)
  │         └── Dual character & BPE token probabilities
  │
  ├──► [2] Window Q-Former Adapter (2-Layer Query Transformer)
  │         └── 5x Downsampling to 10Hz acoustic embedding stream
  │
  └──► [3] Granite 4.0 1B Base LLM (YaRN 16k Extended Context)
            ├── 2,000-Term Keyword Biasing (KWB)
            ├── Multilingual Language ID Prompting
            └── Bidirectional AST (EN, FR, DE, ES, PT, JA, IT, ZH)
```

---

## 🚀 Quick Start & Installation (Windows PowerShell)

### 1. Install Environment Dependencies
Run the following PowerShell command to install the required dependencies:

```powershell
pip install torch torchaudio soundfile numpy transformers>=4.52.1 vllm huggingface_hub
```

### 2. Download Granite Speech 5.0 Model Weights
Download the primary model weights directly into the project's `models/` directory:

```powershell
hf download ibm-granite/granite-speech-5.0-2b --local-dir models/granite-speech-5.0-2b
```

### 3. Run the Unified Production Pipeline
Execute the full multi-task pipeline:

```powershell
python main.py
```

Generated single-file visual deliverables will be saved in `outputs/`:
- `outputs/granite_meeting_intelligence.html`
- `outputs/granite_edge_latency_report.json`
- `outputs/granite_cross_border_translation.md`

---

## 💻 Optional Alternative Deployment Options (Windows PowerShell)

### Run with High-Throughput vLLM Server
```powershell
vllm serve ibm-granite/granite-speech-5.0-2b --max-model-len 2048 --limit-mm-per-prompt '{"audio": 1}'
```

### Run with llama.cpp GGUF CLI (Offline On-Device)
```powershell
# Download 8-bit quantized GGUF (2.4 GB)
hf download ibm-granite/granite-speech-5.0-2b-GGUF granite-speech-5.0-2b-Q8_0.gguf --local-dir models/gguf

# Transcribe via CLI
llama-cli -st -m models/gguf/granite-speech-5.0-2b-Q8_0.gguf --audio "inputs/townhall_speech.wav" -p "transcribe the speech with proper punctuation and capitalization."
```

---

## 🏢 Practical Enterprise Use Cases

- 🏛️ **Townhall & Legislative Address Transcription:** Automatically transcribes and attributes speakers across multi-hour conference recordings, inserting proper punctuation and generating categorized executive summaries.
- 🩺 **Medical & Technical Transcription with 2,000 KWB Bias:** Passes complex pharmaceutical formularies, anatomical terminology, and proprietary product names into the prompt to guarantee 100% spelling precision without fine-tuning.
- ⚡ **Ultra-Low Latency Live Closed Captioning:** Uses the compiled 470M Turbo CTC engine on edge micro-servers or consumer laptop CPUs to stream closed captions with sub-100ms latency.
- 🌐 **Cross-Border Multi-Office Operations:** Directly translates international voice memos from Spanish, German, French, or Japanese into English while preserving company acronyms and technical terms intact.

---

## 🔮 Future Feature Roadmap

- 🇮🇳 **Native Indian Languages Expansion:** Community fine-tuning adapter for Hindi, Tamil, and Telugu enterprise call-center ASR.
- 🎙️ **Streaming WebRTC Microphone Adapter:** Browser-based live audio capture piped directly into local vLLM audio endpoints.
- ✂️ **Automated VAD Pre-Chunking Pipeline:** Integrated Silero-VAD chunking loop for multi-hour podcast audio splitting.
- 📊 **Dynamic Speaker Diarization Fusion:** Integrating Pyannote / Nemotron-3 diarization labels with Granite 5.0 timestamp tokens.

---

## 📄 License
IBM Granite Speech 5.0 is released under the permissive [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).

---

### Search Tags & Metadata
`ibm granite speech 5.0 2b` `granite speech 5.0` `ibm granite` `granite speech 2b` `whisper alternative` `open source asr` `local speech to text` `speech to text ai` `local stt` `offline speech recognition` `voice recognition` `ibm speech model` `granite speech tutorial` `fast speech recognition` `free transcription` `audio transcription` `offline stt` `Granite Speech 5.0` `IBM Granite` `Granite Speech` `Granite Speech 2B` `Whisper Turbo` `OpenAI Whisper` `speech to text`
