#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-sleep"
REQUIREMENTS="$PROJECT_DIR/requirements-sleep-inference.txt"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"

if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
  echo "没有找到 python3。请先安装 Python 3.10–3.12。" >&2
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "首次运行：正在创建独立环境……"
  "$PYTHON_COMMAND" -m venv "$VENV_DIR"
fi

REQ_HASH="$($VENV_DIR/bin/python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$REQUIREMENTS")"
STAMP_FILE="$VENV_DIR/.sleep-requirements-sha256"
INSTALLED_HASH=""
if [ -f "$STAMP_FILE" ]; then
  INSTALLED_HASH="$(tr -d '\r\n' < "$STAMP_FILE")"
fi

if [ "$REQ_HASH" != "$INSTALLED_HASH" ]; then
  echo "首次运行或依赖有更新：正在安装推理环境……"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"
  "$VENV_DIR/bin/python" -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]+"\n")' "$STAMP_FILE" "$REQ_HASH"
fi

exec "$VENV_DIR/bin/python" "$PROJECT_DIR/sleep_chat.py" "$@"
