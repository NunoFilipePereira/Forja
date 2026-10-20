#!/bin/bash
# Lança a Forja 3D. Na primeira vez (ou se a pasta mudar de sítio) cria o ambiente Python.
cd "$(dirname "$0")"
PY=$(command -v python3.14 || command -v python3.13 || command -v python3.12 || command -v python3)
if ! .venv/bin/python -c "import PySide6, trimesh, manifold3d" 2>/dev/null; then
  echo "A preparar o ambiente (só da primeira vez, pode demorar uns minutos)..."
  rm -rf .venv
  "$PY" -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt || { echo "Falhou a instalação."; read -n1; exit 1; }
fi
exec .venv/bin/python -m forja "$@"
