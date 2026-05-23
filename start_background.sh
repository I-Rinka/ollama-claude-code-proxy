#!/bin/bash
# Start the proxy in the background
# Usage: bash ~/ollama-anthropic-proxy/start_background.sh
#        or:  bash ~/ollama-anthropic-proxy/start_background.sh stop

PROXY_SCRIPT="$HOME/ollama-anthropic-proxy/anthropic_proxy.py"
LOG_FILE="$HOME/ollama-anthropic-proxy/proxy.log"
PID_FILE="$HOME/ollama-anthropic-proxy/.proxy.pid"

case "${1:-start}" in
  stop)
    if [ -f "$PID_FILE" ]; then
      kill "$(cat "$PID_FILE")" 2>/dev/null && echo "Proxy stopped (PID $(cat "$PID_FILE"))" || echo "Proxy not running"
      rm -f "$PID_FILE"
    else
      echo "No PID file found. Proxy may not be running."
    fi
    exit 0
    ;;
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Proxy is running (PID $(cat "$PID_FILE"))"
    else
      echo "Proxy is NOT running"
    fi
    exit 0
    ;;
  start)
    ;;
  *)
    echo "Usage: $0 {start|stop|status}"
    exit 1
    ;;
esac

# Check if already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Proxy is already running (PID $(cat "$PID_FILE"))"
  exit 0
fi

# Activate conda, unset proxy env vars, launch in background
eval "$(~/miniconda3/bin/conda shell.bash hook)"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

nohup python3 "$PROXY_SCRIPT" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 2
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Proxy started (PID $(cat "$PID_FILE"))"
  echo "  Endpoint: http://localhost:8056"
  echo "  Log file: $LOG_FILE"
else
  echo "Failed to start proxy. Check log:"
  tail -20 "$LOG_FILE"
fi
