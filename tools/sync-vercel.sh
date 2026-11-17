#!/usr/bin/env bash
# Sincroniza o código-fonte único (acao-clara/) com o pacote publicado na Vercel
# (acao-clara-vercel/). O app.py de cada pasta é diferente de propósito: a versão
# local usa http.server e a versão publicada é uma aplicação Flask.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/acao-clara"
DEST="$ROOT/acao-clara-vercel"

echo "Sincronizando $SRC -> $DEST"

rm -rf "$DEST/engine" "$DEST/data"
cp -R "$SRC/engine" "$DEST/engine"
cp -R "$SRC/data" "$DEST/data"

cp "$SRC/server_core.py" "$DEST/server_core.py"
cp "$SRC/requirements.txt" "$DEST/requirements.txt"

mkdir -p "$DEST/static" "$DEST/public"
cp "$SRC/static/." "$DEST/static/" -R
cp "$SRC/static/." "$DEST/public/" -R

rm -rf "$DEST/tests"
cp -R "$SRC/tests" "$DEST/tests"

find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} +
rm -rf "$DEST/runtime"

echo "Pronto. Confira o diff com: git status"
