"""Aplicação Flask do Ação Clara para publicação na Vercel.

Os mesmos endpoints do servidor local: análise clássica e motor quantitativo.
`public/` serve o frontend na Vercel; `static/` atende o servidor local.
"""
import os
import traceback

from flask import Flask, jsonify, request, send_from_directory

from engine import api as engine_api
from server_core import (
    ENGINE_DATA,
    UserError,
    analyze,
    make_ledger,
    quote_live,
    ranking_batch,
    search,
    universe,
)

PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
LEDGER = make_ledger()
app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path="")


@app.get("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/api/search")
def api_search():
    try:
        return jsonify({"results": search(request.args.get("q", ""))})
    except UserError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Não foi possível concluir a busca."}), 500


@app.get("/api/analyze")
def api_analyze():
    try:
        return jsonify(analyze(request.args.get("symbol", ""), int(request.args.get("horizon", "20"))))
    except UserError as error:
        return jsonify({"error": str(error)}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "Parâmetros inválidos."}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Não foi possível concluir a análise. Tente novamente."}), 500


@app.get("/api/quote")
def api_quote():
    try:
        return jsonify(quote_live(request.args.get("symbol", "").upper()))
    except UserError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Não foi possível obter a cotação."}), 500


@app.get("/api/universe")
def api_universe():
    try:
        return jsonify(universe())
    except UserError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Não foi possível obter a lista de ações."}), 500


@app.get("/api/ranking")
def api_ranking():
    try:
        symbols = request.args.get("symbols", "").split(",")
        return jsonify(ranking_batch(symbols, int(request.args.get("horizon", "20"))))
    except UserError as error:
        return jsonify({"error": str(error)}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "Parâmetros inválidos."}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Não foi possível concluir o lote do ranking."}), 500


@app.get("/api/quant")
@app.get("/api/backtest")
@app.get("/api/macro")
@app.get("/api/ledger")
@app.get("/api/health")
def api_engine():
    params = {key: request.args.getlist(key) for key in request.args}
    try:
        return jsonify(engine_api.handle(request.path, params, ENGINE_DATA, LEDGER))
    except engine_api.ApiError as error:
        return jsonify({"error": error.message}), error.status


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Não encontrado."}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
