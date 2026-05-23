#!/bin/bash
# Start the Anthropic-to-Ollama proxy v3
# Usage: bash ~/start_proxy.sh

echo "Starting Ollama + Anthropic API Proxy v3..."
echo "  Features: tool_use, token tracking, enhanced system prompt"
echo ""

# Activate conda
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Unset proxy env vars (Ollama is local, doesn't need proxy)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# Start proxy
echo "Proxy will be available at http://localhost:8056"
echo "Press Ctrl+C to stop"
echo ""
python3 ~/anthropic_proxy.py
