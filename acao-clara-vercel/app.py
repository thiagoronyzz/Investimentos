from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json, os, re, traceback, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parent
SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; AcaoClara/1.0)'})
HEADERS = {'Accept': 'application/json'}

class UserError(Exception): pass

def yahoo(url, params):
    try:
        response = SESSION.get(url, params={**params, '_': str(int(time.time()))}, headers={**HEADERS, 'Cache-Control': 'no-cache'}, timeout=18)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as e:
        raise UserError('Não foi possível consultar a fonte de dados agora. Tente novamente mais tarde.') from e

def search(q):
    if len(q.strip()) < 2: return []
    data = yahoo('https://query1.finance.yahoo.com/v1/finance/search', {'q': q.strip(), 'quotesCount': 20, 'newsCount': 0, 'enableFuzzyQuery': 'false'})
    results=[]
    for item in data.get('quotes', []):
        symbol=item.get('symbol','')
        if item.get('quoteType') == 'EQUITY' and symbol.upper().endswith('.SA'):
            results.append({'symbol':symbol,'name':item.get('longname') or item.get('shortname') or symbol,'exchange': item.get('exchDisp','B3')})
    return results[:10]

def history(symbol):
    data=yahoo('https://query1.finance.yahoo.com/v8/finance/chart/'+symbol, {'range':'10y','interval':'1d','events':'div,splits'})
    chart=data.get('chart',{})
    if chart.get('error') or not chart.get('result'): raise UserError('Não encontramos histórico para esse ativo.')
    result=chart['result'][0]
    timestamps=result.get('timestamp',[])
    quote=result.get('indicators',{}).get('quote',[{}])[0]
    adj=result.get('indicators',{}).get('adjclose',[{}])[0].get('adjclose')
    if not adj: raise UserError('A fonte não forneceu fechamento ajustado para esse ativo; não é possível calcular a análise com segurança.')
    df=pd.DataFrame({'date':pd.to_datetime(timestamps,unit='s',utc=True).tz_convert('America/Sao_Paulo').date,'close':adj,'raw_close':quote.get('close',[None]*len(timestamps)),'volume':quote.get('volume',[None]*len(timestamps))})
    # Accept today's daily bar only after the market close plus a 30-minute buffer.
    now=datetime.now(ZoneInfo('America/Sao_Paulo')); today=now.date()
    market_end=result.get('meta',{}).get('currentTradingPeriod',{}).get('regular',{}).get('end')
    closed=bool(market_end and now.timestamp() >= market_end + 1800 and datetime.fromtimestamp(market_end,ZoneInfo('America/Sao_Paulo')).date()==today)
    df=df[df.date <= today if closed else df.date < today].copy()
    df['close']=pd.to_numeric(df.close,errors='coerce'); df['volume']=pd.to_numeric(df.volume,errors='coerce')
    df=df.dropna(subset=['date','close']).loc[lambda x: x.close>0].drop_duplicates('date').sort_values('date').reset_index(drop=True)
    return df, result.get('meta',{})

FEATURES=['ret5','ret20','ret60','vol20','ma20','ma60']
def features(df):
    p=df.close.astype(float)
    out=pd.DataFrame(index=df.index)
    for n in (5,20,60): out[f'ret{n}']=p.pct_change(n,fill_method=None)
    out['vol20']=p.pct_change(fill_method=None).rolling(20).std()
    for n in (20,60): out[f'ma{n}']=p/p.rolling(n).mean()-1
    return out.replace([np.inf,-np.inf],np.nan)

def split_indices(valid, start, end, horizon, max_labeled):
    # Training outcomes must be known strictly before any test feature date.
    indices = np.arange(len(valid))
    train = np.flatnonzero(valid & (indices + horizon < start))
    test = np.flatnonzero(valid & (indices >= start) & (indices < end) & (indices <= max_labeled))
    return train, test

def calculate(df, horizon):
    if horizon not in (5,10,20,60): raise UserError('Prazo inválido.')
    x=features(df); p=df.close
    valid=x.notna().all(axis=1).to_numpy()
    # Labels are known only when index + horizon is available.
    max_labeled=len(df)-1-horizon
    usable=np.flatnonzero(valid & (np.arange(len(df))<=max_labeled))
    if len(usable)<550: raise UserError('Não há histórico suficiente para treinar e testar o modelo com segurança. São necessários aproximadamente três anos de pregões.')
    # Four sequential non-overlapping test blocks; purge labels crossing the boundary.
    test_size=min(126, max(50,(len(usable)-350)//4))
    first_test=max_labeled-4*test_size+1
    if first_test<0: raise UserError('Histórico insuficiente para validar o modelo.')
    y=(p.shift(-horizon)>p).astype(int).to_numpy()
    probs=[]; base=[]; actual=[]; dates=[]; price_errors=[]; price_baseline_errors=[]
    future_returns=(p.shift(-horizon)/p-1).to_numpy()
    for block in range(4):
        start=first_test+block*test_size; end=start+test_size
        train,test=split_indices(valid,start,end,horizon,max_labeled)
        if len(train)<250 or len(test)==0 or len(np.unique(y[train]))<2: raise UserError('Os dados não têm variedade suficiente para treinar e testar o modelo.')
        model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        model.fit(x.iloc[train],y[train])
        probs.extend(model.predict_proba(x.iloc[test])[:,1]); base.extend([float(y[train].mean())]*len(test)); actual.extend(y[test]); dates.extend(df.date.iloc[test].tolist())
        price_model=make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        price_model.fit(x.iloc[train],future_returns[train])
        estimated_returns=np.clip(price_model.predict(x.iloc[test]),-.95,3.0)
        price_errors.extend(np.abs(estimated_returns-future_returns[test]))
        price_baseline_errors.extend(np.abs(future_returns[test]))
    probs=np.array(probs); base=np.array(base); actual=np.array(actual)
    if not valid[-1] or len(np.unique(y[usable]))<2: raise UserError('Não há dados válidos suficientes para gerar a estimativa atual.')
    final=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0))
    final.fit(x.iloc[usable], y[usable])
    probability=float(final.predict_proba(x.iloc[[-1]])[0,1])
    final_price=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
    final_price.fit(x.iloc[usable],future_returns[usable])
    projected_return=float(np.clip(final_price.predict(x.iloc[[-1]])[0],-.95,3.0))
    bins=[]
    for lo in np.arange(0,1,0.2):
        mask=(probs>=lo)&(probs<(lo+0.2) if lo<0.8 else probs<=1)
        if mask.sum(): bins.append({'min':int(round(lo*100)),'max':int(round((lo+.2)*100)),'count':int(mask.sum()),'predicted':float(probs[mask].mean()),'observed':float(actual[mask].mean())})
    brier=float(brier_score_loss(actual,probs)); baseline_brier=float(brier_score_loss(actual,base))
    return {'probability':probability,'projected_return':projected_return,'price_mae':float(np.mean(price_errors)),'price_baseline_mae':float(np.mean(price_baseline_errors)),'accuracy':float(((probs>=.5)==actual).mean()),'brier':brier,'baseline_brier':baseline_brier,'baseline_accuracy':float(((base>=.5)==actual).mean()),'count':len(actual),'start':str(dates[0]),'end':str(dates[-1]),'bins':bins,'test_size':test_size}

def analyze(symbol,horizon):
    symbol=symbol.upper().strip()
    if not re.fullmatch(r'[A-Z0-9]{4,12}\.SA',symbol): raise UserError('Escolha uma ação brasileira válida na busca.')
    df,meta=history(symbol)
    if df.empty: raise UserError('Não há cotações completas disponíveis para esta ação.')
    metrics=calculate(df,horizon)
    # About two years plotted; calculations always use full history.
    chart=df.tail(504)
    return {'symbol':symbol,'name':meta.get('longName') or meta.get('shortName') or symbol,'date':str(df.date.iloc[-1]),'price':float(df.close.iloc[-1]),'projected_price':float(df.close.iloc[-1])*(1+metrics['projected_return']),'quote':quote_from_meta(meta),'currency':meta.get('currency','BRL'),'horizon':horizon,'metrics':metrics,'chart':[{'date':str(r.date),'close':round(float(r.close),4)} for r in chart.itertuples()],'total_rows':len(df)}


def quote_from_meta(meta):
    price=meta.get('regularMarketPrice'); stamp=meta.get('regularMarketTime')
    if price is None or not stamp: return None
    return {'price':float(price),'time':datetime.fromtimestamp(stamp,ZoneInfo('America/Sao_Paulo')).isoformat()}

def quote_live(symbol):
    if not re.fullmatch(r'[A-Z0-9]{4,12}\.SA',symbol): raise UserError('Ativo inválido.')
    data=yahoo('https://query1.finance.yahoo.com/v8/finance/chart/'+symbol,{'range':'5d','interval':'1d'})
    result=data.get('chart',{}).get('result') or []
    if not result: raise UserError('Cotação mais recente indisponível.')
    quote=quote_from_meta(result[0].get('meta',{}))
    if not quote: raise UserError('A fonte não forneceu cotação recente.')
    return quote

def universe():
    try:
        r=requests.get('https://brapi.dev/api/quote/list',params={'type':'stock','limit':1000},headers={'User-Agent':'Mozilla/5.0','Cache-Control':'no-cache'},timeout=20)
        r.raise_for_status(); data=r.json()
    except (requests.RequestException,ValueError) as e:
        raise UserError('Não foi possível obter a lista de ações da B3. Tente novamente.') from e
    if data.get('totalPages',1)>1: raise UserError('A lista da fonte veio incompleta; tente mais tarde.')
    result=[];seen=set()
    for row in data.get('stocks',[]):
        ticker=str(row.get('stock','')).upper()
        if not re.fullmatch(r'[A-Z0-9]{4,10}(?:3|4|5|6|11)',ticker) or ticker in seen: continue
        seen.add(ticker); result.append({'symbol':ticker+'.SA','name':row.get('name') or ticker})
    if not result: raise UserError('A fonte não retornou ações disponíveis para o ranking.')
    return {'items':result,'source':'brapi.dev: lista de ativos classificados como ações','total_in_source':data.get('totalCount',len(result))}

def ranking_item(item,horizon):
    try:
        symbol=item['symbol']; df,_=history(symbol)
        if df.empty or (datetime.now(ZoneInfo('America/Sao_Paulo')).date()-df.date.iloc[-1]).days>7: return None
        x=features(df); valid=x.notna().all(axis=1).to_numpy(); last=len(df)-1-horizon
        usable=np.flatnonzero(valid & (np.arange(len(df))<=last))
        if len(usable)<550 or not valid[-1]: return None
        y=(df.close.shift(-horizon)>df.close).astype(int).to_numpy()
        if len(np.unique(y[usable]))<2:return None
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0))
        model.fit(x.iloc[usable],y[usable]); prob=float(model.predict_proba(x.iloc[[-1]])[0,1])
        return {'symbol':symbol,'name':item['name'],'probability':prob,'date':str(df.date.iloc[-1])}
    except Exception:
        return None

def ranking_batch(symbols,horizon):
    if horizon not in (5,10,20,60): raise UserError('Prazo inválido.')
    if len(symbols)>4: raise UserError('Lote de ranking muito grande.')
    items=[]
    for symbol in symbols:
        symbol=symbol.strip().upper()
        if re.fullmatch(r'[A-Z0-9]{4,12}\.SA',symbol): items.append({'symbol':symbol,'name':symbol.removesuffix('.SA')})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda x:ranking_item(x,horizon),items))
    return {'results':[r for r in results if r is not None], 'attempted':len(items),'failed':len(items)-sum(r is not None for r in results)}

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(ROOT/'static'),**kwargs)
    def do_GET(self):
        parsed=urlparse(self.path)
        if not parsed.path.startswith('/api/'): return super().do_GET()
        try:
            q=parse_qs(parsed.query)
            if parsed.path=='/api/search': data={'results':search(q.get('q',[''])[0])}
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

# Vercel loads this WSGI application; the local development server below remains available.
from flask import Flask, request, jsonify
app = Flask(__name__, static_folder=None)

@app.get('/api/search')
def api_search():
    try:
        return jsonify({'results': search(request.args.get('q', ''))})
    except UserError as e:
        return jsonify({'error': str(e)}), 400
    except Exception:
        app.logger.exception('Falha na pesquisa')
        return jsonify({'error': 'Não foi possível concluir a pesquisa. Tente novamente.'}), 500

@app.get('/api/analyze')
def api_analyze():
    try:
        return jsonify(analyze(request.args.get('symbol', ''), int(request.args.get('horizon', '20'))))
    except UserError as e:
        return jsonify({'error': str(e)}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Parâmetros inválidos.'}), 400
    except Exception:
        app.logger.exception('Falha na análise')
        return jsonify({'error': 'Não foi possível concluir a análise. Tente novamente.'}), 500

@app.after_request
def no_cache_api(response):
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response

@app.get('/api/quote')
def api_quote():
    try: return jsonify(quote_live(request.args.get('symbol','').upper()))
    except UserError as e: return jsonify({'error':str(e)}), 400
    except Exception:
        app.logger.exception('Falha na cotação')
        return jsonify({'error':'Não foi possível atualizar a cotação.'}), 500

@app.get('/api/universe')
def api_universe():
    try: return jsonify(universe())
    except UserError as e: return jsonify({'error':str(e)}), 400
    except Exception:
        app.logger.exception('Falha na lista')
        return jsonify({'error':'Não foi possível obter a lista de ações.'}), 500

@app.get('/api/ranking')
def api_ranking():
    try: return jsonify(ranking_batch(request.args.get('symbols','').split(','),int(request.args.get('horizon','20'))))
    except UserError as e: return jsonify({'error':str(e)}), 400
    except (ValueError,TypeError): return jsonify({'error':'Parâmetros inválidos.'}), 400
    except Exception:
        app.logger.exception('Falha no ranking')
        return jsonify({'error':'Não foi possível calcular este lote do ranking.'}), 500

if __name__=='__main__':
    port=int(os.environ.get('PORT','8000'))
    print(f'Ação Clara disponível em http://0.0.0.0:{port}',flush=True)
    ThreadingHTTPServer(('0.0.0.0',port),Handler).serve_forever()
