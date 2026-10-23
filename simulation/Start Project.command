#!/bin/zsh
set -e
PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
if [[ -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
  "$PROJECT_DIR/.venv/bin/python3" "$PROJECT_DIR/start.py"
else
  python3 "$PROJECT_DIR/start.py"
fi
