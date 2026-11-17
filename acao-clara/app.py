"""Servidor local do Ação Clara (http.server).

Expõe os endpoints clássicos (/api/search, /api/analyze, /api/quote,
/api/universe, /api/ranking) e o motor quantitativo (/api/quant, /api/backtest,
/api/macro, /api/ledger, /api/health), servindo os arquivos de static/.
"""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import json, os, traceback

from engine import api as engine_api
# `server_core` concentra a lógica clássica; os nomes são reexportados aqui para
# manter a superfície pública que os testes e integrações existentes usam.
from server_core import (
    ENGINE_DATA,
    UserError,
    analyze,
    calculate,
    features,
    history,
    make_ledger,
    quote_live,
    ranking_batch,
    ranking_item,
    search,
    split_indices,
    universe,
)

ROOT = Path(__file__).resolve().parent
LEDGER = make_ledger()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(ROOT/'static'),**kwargs)
    def do_GET(self):
        parsed=urlparse(self.path)
        if not parsed.path.startswith('/api/'): return super().do_GET()
        try:
            q=parse_qs(parsed.query)
            if parsed.path in engine_api.routes():
                try: data=engine_api.handle(parsed.path,q,ENGINE_DATA,LEDGER)
                except engine_api.ApiError as e: return self.json_response({'error':e.message},e.status)
            elif parsed.path=='/api/search': data={'results':search(q.get('q',[''])[0])}
            elif parsed.path=='/api/analyze': data=analyze(q.get('symbol',[''])[0],int(q.get('horizon',['20'])[0]))
            elif parsed.path=='/api/quote': data=quote_live(q.get('symbol',[''])[0].upper())
            elif parsed.path=='/api/universe': data=universe()
            elif parsed.path=='/api/ranking': data=ranking_batch(q.get('symbols',[''])[0].split(','),int(q.get('horizon',['20'])[0]))
            else: return self.json_response({'error':'Não encontrado.'},404)
            self.json_response(data)
        except UserError as e: self.json_response({'error':str(e)},400)
        except (ValueError,TypeError): self.json_response({'error':'Parâmetros inválidos.'},400)
        except Exception:
            traceback.print_exc(); self.json_response({'error':'Não foi possível concluir a análise. Tente novamente.'},500)
    def json_response(self,data,status=200):
        payload=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)


if __name__=='__main__':
    port=int(os.environ.get('PORT','8000'))
    print(f'Ação Clara disponível em http://0.0.0.0:{port}',flush=True)
    ThreadingHTTPServer(('0.0.0.0',port),Handler).serve_forever()
