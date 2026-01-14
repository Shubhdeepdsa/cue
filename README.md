# CUE v0.1 • Intent Capture System

<div align="center">

```
╔═══════════════════════════════════════════════════════════╗
║              CUE v0.1 • Intent Capture System             ║
║                  Local Simulation Edition                 ║
╚═══════════════════════════════════════════════════════════╝
```

**A production-ready headless intent-capture system that simulates a hardware device.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## 🎯 Overview

CUE (Capture, Understand, Execute) is a voice-based intent processing system designed for local development and testing. It simulates a hardware device by:

- **Capturing** voice input via spacebar hotkey
- **Understanding** speech using Deepgram STT + Ollama LLM
- **Executing** by generating structured intent tickets

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CUE System                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────────┐   │
│   │ Keyboard │────▶│ State Machine│────▶│ LED Simulator    │   │
│   │ (Space)  │     │              │     │ (Rich Console)   │   │
│   └──────────┘     └──────┬───────┘     └──────────────────┘   │
│                           │                                     │
│   ┌──────────┐            │                                     │
│   │   Mic    │────────────┤                                     │
│   │(sounddev)│            ▼                                     │
│   └──────────┘   ┌────────────────┐                             │
│                  │  Intelligence  │                             │
│                  │     Layer      │                             │
│                  ├────────┬───────┤                             │
│                  │Deepgram│Ollama │                             │
│                  │  STT   │ LLM   │                             │
│                  └────────┴───┬───┘                             │
│                               │                                 │
│                               ▼                                 │
│                  ┌────────────────────┐                         │
│                  │  Ticket Printer    │──────▶ tickets/*.txt   │
│                  └────────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
- **[Ollama](https://ollama.ai/)** - Local LLM runtime
- **Deepgram API Key** - [Get one free](https://console.deepgram.com/)

### Installation

```bash
# Clone or navigate to the project
cd cue

# Install dependencies with uv
uv sync

# Install dev dependencies (optional)
uv sync --extra dev

# Copy environment template
cp .env.example .env
```

### Configuration

1. **Edit `.env`** - Add your Deepgram API key:
   ```env
   DEEPGRAM_API_KEY=your_actual_api_key_here
   ```

2. **Start Ollama** with Llama 3.1:
   ```bash
   ollama pull llama3.1
   ollama serve
   ```

3. **Review `config.yaml`** - Adjust settings as needed:
   - Audio sample rate and channels
   - LED colors and labels
   - LLM prompts (scratchpad strategy)
   - Ticket template

### Run

```bash
# Start CUE
uv run cue

# Or run directly
uv run python -m cue.main
```

---

## 🎮 Usage

| Key | Action |
|-----|--------|
| `SPACE` | Toggle recording (start/stop) |
| `Ctrl+C` | Exit application |

### Workflow

1. **IDLE** (Green) - System ready, press SPACE to start
2. **RECORDING** (Red) - Speak your intent, press SPACE to stop
3. **PROCESSING** (Orange) - AI analyzing your speech
4. **SUCCESS** (Green flash) - Ticket generated!

### Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║                        CUE TICKET #00001                         ║
╠══════════════════════════════════════════════════════════════════╣
║  TIMESTAMP   │ 2026-01-13 17:30:00 UTC                           ║
╠══════════════════════════════════════════════════════════════════╣
║  INTENT      │ Schedule a meeting with the team                  ║
╠══════════════════════════════════════════════════════════════════╣
║  ACTION      │ schedule                                          ║
╠══════════════════════════════════════════════════════════════════╣
║  ENTITIES    │ attendees: team, date: tomorrow                   ║
╠══════════════════════════════════════════════════════════════════╣
║  CONFIDENCE  │ 92.5%                                             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
cue/
├── __init__.py         # Package init, exports settings
├── config.py           # Pydantic config (YAML + .env merger)
├── hardware.py         # Audio, hotkey, LED simulation
├── intelligence.py     # Deepgram STT + Ollama LLM
├── printer.py          # Ticket generation
├── main.py             # State machine entry point
└── utils.py            # Rich logger, helpers

config.yaml             # Behavioral configuration
.env                    # Secrets (API keys)
tickets/                # Generated intent tickets
tests/                  # Pytest test suite
```

---

## ⚙️ Configuration

### Separation of Concerns

| File | Purpose |
|------|---------|
| `.env` | **Secrets only** (API keys) |
| `config.yaml` | **Behavior** (prompts, colors, settings) |

### Key Configuration Sections

```yaml
hardware:
  audio:
    sample_rate: 16000    # Deepgram optimal
    channels: 1           # Mono
  hotkey: "space"         # Toggle key

led_states:
  idle: { color: "#00FF88", label: "● READY" }
  recording: { color: "#FF3366", label: "◉ RECORDING" }
  processing: { color: "#FFAA00", label: "◐ PROCESSING" }

llm:
  model: "llama3.1"
  scratchpad_prompt: |    # Multi-step reasoning prompt
    ...
```

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=cue --cov-report=html

# Run specific test file
uv run pytest tests/test_config.py -v
```

---

## 🔧 Development

```bash
# Type checking
uv run mypy cue/

# Linting
uv run ruff check cue/

# Formatting
uv run ruff format cue/
```

---

## 🐛 Troubleshooting

### "Ollama is not running"

```bash
# Start Ollama service
ollama serve

# Verify model is available
ollama list
```

### "No audio devices found"

Ensure microphone permissions are granted for Terminal/iTerm on macOS:
**System Preferences → Security & Privacy → Privacy → Microphone**

### "Deepgram API error"

1. Verify your API key in `.env`
2. Check your Deepgram console for quota

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ using Python, Rich, Deepgram & Ollama**

</div>
