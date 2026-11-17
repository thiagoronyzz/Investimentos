"""Ponto de entrada do repositório.

A aplicação completa (análise clássica + motor quantitativo de oito núcleos) mora
em `acao-clara/`. Este arquivo apenas a executa, para que `python app.py` na raiz
do repositório abra a mesma interface.
"""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "acao-clara"

if not (APP_DIR / "app.py").exists():  # pragma: no cover - instalação incompleta
    raise SystemExit(f"aplicação não encontrada em {APP_DIR}")

# O motor e os módulos clássicos são importados a partir de acao-clara/.
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

if __name__ == "__main__":
    runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
