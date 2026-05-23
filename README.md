# Ollama Anthropic Proxy

Run local LLMs with Claude Code via a lightweight Anthropic Messages API → OpenAI Chat Completions API proxy.

## Overview

This project bridges **Ollama** (local LLM inference) with **Claude Code** (Anthropic's coding assistant) by translating between the Anthropic Messages API format that Claude Code speaks and the OpenAI-compatible API that Ollama provides.

```
Claude Code  ──▶  Anthropic API Proxy  ──▶  Ollama  ──▶  Local GPU
  (client)        (format translation)     (server)     (inference)
```

## Features

- **Full tool-use support** — Bash, Read, Edit, Write, NotebookEdit and any other tools Claude Code provides
- **Streaming responses** — real-time SSE streaming with token-by-token delivery
- **Real token tracking** — actual input/output token counts from Ollama
- **Multi-turn conversations** — handles assistant messages with `tool_use` blocks and user messages with `tool_result` blocks
- **Enhanced system prompt** — injects coding-agent instructions so the local model knows how to use tools effectively
- **Robust error handling** — returns Anthropic-standard error responses

## Requirements

- NVIDIA GPU with ≥24GB VRAM (tested on RTX 5090 32GB)
- Ubuntu 22.04+ (should work on other Linux distros)
- Python 3.10+

## Quick Start

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a model

For 24-32GB VRAM GPUs, Qwen3 30B-A3B is recommended:

```bash
ollama pull qwen3:30b-a3b
```

For smaller GPUs (8-16GB):

```bash
ollama pull qwen3:8b
```

Then update `DEFAULT_MODEL` in `anthropic_proxy.py` accordingly.

### 3. Install Python dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda install fastapi uvicorn httpx
```

### 4. Start the proxy

```bash
bash start_proxy.sh
```

The proxy will be available at `http://localhost:8056`.

### 5. Configure Claude Code

Update your `~/.claude/settings.json`:

```json
{
    "env": {
        "ANTHROPIC_AUTH_TOKEN": "local-token",
        "ANTHROPIC_BASE_URL": "http://localhost:8056",
        "ANTHROPIC_MODEL": "qwen3:30b-a3b",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3:30b-a3b",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3:30b-a3b",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3:30b-a3b"
    }
}
```

Then restart Claude Code and you're ready to go!

## Project Structure

```
ollama-anthropic-proxy/
├── anthropic_proxy.py   # The API proxy server (main code)
├── start_proxy.sh       # Startup script (handles conda + proxy env)
├── gpu_test.py          # GPU benchmark script (PyTorch test)
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License
└── .gitignore           # Git ignore rules
```

## Configuration

Edit the top of `anthropic_proxy.py`:

```python
OLLAMA_BASE_URL = "http://localhost:11434/v1"  # Ollama endpoint
DEFAULT_MODEL   = "qwen3:30b-a3b"              # Default model name
HOST, PORT      = "0.0.0.0", 8056              # Proxy listen address
```

The enhanced system prompt (`ENHANCED_SYSTEM`) can also be customised to change how the local model behaves as a coding assistant.

## How It Works

### API Translation

| Anthropic Messages API | OpenAI Chat Completions API |
|---|---|
| `system` (string or list) | `messages[0]` with `role: system` |
| `messages[].content` (text blocks) | `messages[].content` (string) |
| `messages[].content` (tool_use blocks) | `messages[].tool_calls` |
| `messages[].content` (tool_result blocks) | `messages[]` with `role: tool` |
| `tools[].input_schema` | `tools[].function.parameters` |
| SSE `content_block_delta` | SSE `delta.content` / `delta.tool_calls` |
| `usage.input_tokens` / `output_tokens` | `usage.prompt_tokens` / `completion_tokens` |

### Qwen3 Thinking Mode

Qwen3 models use "thinking" tokens for reasoning. The proxy handles this by:

1. Appending `/no_think` to the system prompt to suppress extended reasoning
2. Setting a minimum `max_tokens` of 8192 to ensure enough headroom for thinking + response
3. Falling back to `reasoning` field if `content` is empty

### Enhanced System Prompt

The proxy injects a detailed system prompt that tells the local model:

- It's running inside Claude Code with tool access
- When and how to use each tool (Bash, Read, Edit, Write, etc.)
- Coding style guidelines (be direct, use tools proactively, etc.)
- Safety rules (ask before destructive operations, etc.)

## GPU Performance

Tested on RTX 5090 32GB with Qwen3 30B-A3B (Q4 quantisation):

| Metric | Value |
|---|---|
| Generation speed | ~81 tokens/s |
| Model size | 18 GB |
| VRAM usage | ~20 GB / 32 GB |
| Prompt processing | ~6 tokens/s |

## GPU Test Script

The included `gpu_test.py` verifies your GPU is properly configured for inference:

```bash
# Requires PyTorch: pip install torch torchvision
python gpu_test.py
```

This runs a small neural network training loop to confirm CUDA is working.

## Troubleshooting

### "ModuleNotFoundError: No module named 'fastapi'"

Install dependencies: `pip install -r requirements.txt`

### Proxy returns 500 Internal Server Error

Check the proxy logs. Common causes:
- Ollama not running (`systemctl start ollama`)
- Model not pulled (`ollama pull qwen3:30b-a3b`)
- SOCKS proxy env vars leaking into httpx (the `start_proxy.sh` script handles this)

### Claude Code shows "API Error"

Ensure `ANTHROPIC_BASE_URL` in `~/.claude/settings.json` points to the proxy (e.g. `http://localhost:8056`), not directly to Ollama.

### Slow responses on first request

Ollama loads the model into VRAM on first use (~44s for Qwen3 30B). Subsequent requests are fast.

## License

MIT — see [LICENSE](LICENSE).
