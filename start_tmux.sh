#!/usr/bin/env bash
set -e

SESSION=codex_oauth_proxy
DIR=/home/jonas/codex_oauth_proxy

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -c "$DIR" \
  "bash -lc 'source .venv/bin/activate && exec python codex_oauth_proxy.py'"
