#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

venv_dir="${1:-.venv-oww-py310}"

echo "==> Creating venv at: $venv_dir"
uv venv --python 3.10 "$venv_dir"

echo "==> Installing openWakeWord (editable, no deps) + training deps"
source "$venv_dir/bin/activate"
uv pip install -e openWakeWord --no-deps
uv pip install -r requirements-oww-train.txt

echo "==> Done"
echo "Activate with: source \"$venv_dir/bin/activate\""

