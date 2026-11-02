from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, os, re, traceback
import numpy as np
import pandas as pd
import requests
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parent
SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; AcaoClara/1.0)'})
HEADERS = {'Accept': 'application/json'}

class UserError(Exception): pass

def yahoo(url, params):
    try:
        response = SESSION.get(url, params=params, headers=HEADERS, timeout=18)
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
    df=pd.DataFrame({'date':pd.to_datetime(timestamps,unit='s',utc=True).tz_convert('America/Sao_Paulo').date,'close':adj,'volume':quote.get('volume',[None]*len(timestamps))})
    # Today's bar may still be forming; only previous completed calendar days are used.
    today=datetime.now(ZoneInfo('America/Sao_Paulo')).date()
    df=df[df.date < today].copy()
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
    probs=[]; base=[]; actual=[]; dates=[]
    for block in range(4):
        start=first_test+block*test_size; end=start+test_size
        train,test=split_indices(valid,start,end,horizon,max_labeled)
        if len(train)<250 or len(test)==0 or len(np.unique(y[train]))<2: raise UserError('Os dados não têm variedade suficiente para treinar e testar o modelo.')
        model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        model.fit(x.iloc[train],y[train])
        probs.extend(model.predict_proba(x.iloc[test])[:,1]); base.extend([float(y[train].mean())]*len(test)); actual.extend(y[test]); dates.extend(df.date.iloc[test].tolist())
    probs=np.array(probs); base=np.array(base); actual=np.array(actual)
    if not valid[-1] or len(np.unique(y[usable]))<2: raise UserError('Não há dados válidos suficientes para gerar a estimativa atual.')
    final=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0))
    final.fit(x.iloc[usable], y[usable])
    probability=float(final.predict_proba(x.iloc[[-1]])[0,1])
    bins=[]
    for lo in np.arange(0,1,0.2):
        mask=(probs>=lo)&(probs<(lo+0.2) if lo<0.8 else probs<=1)
        if mask.sum(): bins.append({'min':int(round(lo*100)),'max':int(round((lo+.2)*100)),'count':int(mask.sum()),'predicted':float(probs[mask].mean()),'observed':float(actual[mask].mean())})
    brier=float(brier_score_loss(actual,probs)); baseline_brier=float(brier_score_loss(actual,base))
    return {'probability':probability,'accuracy':float(((probs>=.5)==actual).mean()),'brier':brier,'baseline_brier':baseline_brier,'baseline_accuracy':float(((base>=.5)==actual).mean()),'count':len(actual),'start':str(dates[0]),'end':str(dates[-1]),'bins':bins,'test_size':test_size}

def analyze(symbol,horizon):
    symbol=symbol.upper().strip()
    if not re.fullmatch(r'[A-Z0-9]{4,12}\.SA',symbol): raise UserError('Escolha uma ação brasileira válida na busca.')
    df,meta=history(symbol)
    if df.empty: raise UserError('Não há cotações completas disponíveis para esta ação.')
    metrics=calculate(df,horizon)
    # About two years plotted; calculations always use full history.
    chart=df.tail(504)
    return {'symbol':symbol,'name':meta.get('longName') or meta.get('shortName') or symbol,'date':str(df.date.iloc[-1]),'price':float(df.close.iloc[-1]),'currency':meta.get('currency','BRL'),'horizon':horizon,'metrics':metrics,'chart':[{'date':str(r.date),'close':round(float(r.close),4)} for r in chart.itertuples()],'total_rows':len(df)}

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(ROOT/'static'),**kwargs)
    def do_GET(self):
        parsed=urlparse(self.path)
        if not parsed.path.startswith('/api/'): return super().do_GET()
        try:
            q=parse_qs(parsed.query)
            if parsed.path=='/api/search': data={'results':search(q.get('q',[''])[0])}
            elif parsed.path=='/api/analyze': data=analyze(q.get('symbol',[''])[0],int(q.get('horizon',['20'])[0]))
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
