const $ = id => document.getElementById(id);
const input = $('search'), suggestions = $('suggestions'), errorBox = $('error');
let selected = null, timer = null, requestId = 0, currentResult = null;
const pct = n => (n * 100).toLocaleString('pt-BR', {maximumFractionDigits: 1}) + '%';
const dateBR = s => { const [y,m,d] = s.split('-'); return `${d}/${m}/${y}`; };
function showError(message) { errorBox.textContent = message; errorBox.hidden = !message; }
function clearResults() { $('results').hidden = true; $('empty').hidden = false; }
async function getJSON(url) { const r = await fetch(url); let data; try { data = await r.json(); } catch { throw Error('Não foi possível ler a resposta do servidor.'); } if (!r.ok) throw Error(data.error || 'Não foi possível concluir a solicitação.'); return data; }
function message(text) { suggestions.replaceChildren(); const node=document.createElement('div'); node.className='suggestion-message'; node.textContent=text; suggestions.append(node); suggestions.hidden=false; }
async function doSearch(q) {
  const id=++requestId;
  if(q.trim().length<2){suggestions.hidden=true;return;}
  message('Buscando ações...');
  try {
    const data=await getJSON('/api/search?q='+encodeURIComponent(q));
    if(id!==requestId || selected) return;
    suggestions.replaceChildren();
    if(!data.results.length){message('Nenhuma ação brasileira encontrada. Tente outro nome ou código.');return;}
    for(const item of data.results){
      const button=document.createElement('button'); button.type='button';button.className='suggestion';button.setAttribute('role','option');
      const left=document.createElement('div');const strong=document.createElement('strong');strong.textContent=item.symbol.replace(/\.SA$/,'');const small=document.createElement('small');small.textContent=item.name;left.append(strong,small);
      const exchange=document.createElement('span');exchange.textContent='B3  ↗';button.append(left,exchange);
      button.addEventListener('click',()=>{selected=item; input.value=`${item.symbol.replace(/\.SA$/,'')} · ${item.name}`; suggestions.hidden=true; $('clear').hidden=false; showError('');});
      suggestions.append(button);
    }
    suggestions.hidden=false;
  }catch(e){if(id===requestId) message(e.message);}
}
input.addEventListener('input',()=>{selected=null;requestId++;clearTimeout(timer);$('clear').hidden=!input.value;showError('');if(input.value.trim().length>=2) timer=setTimeout(()=>doSearch(input.value),300);else suggestions.hidden=true;});
input.addEventListener('focus',()=>{if(!selected && input.value.trim().length>=2) doSearch(input.value);});
input.addEventListener('keydown',e=>{if(e.key==='Escape')suggestions.hidden=true;if(e.key==='Enter'){e.preventDefault();if(!suggestions.hidden){const first=suggestions.querySelector('button');if(first)first.click();}else if(selected)$('analyze').click();}});
$('clear').addEventListener('click',()=>{input.value='';selected=null;requestId++;suggestions.hidden=true;$('clear').hidden=true;input.focus();showError('');});
document.addEventListener('click',e=>{if(!e.target.closest('.search-field')) suggestions.hidden=true;});
document.querySelectorAll('[data-query]').forEach(btn=>btn.addEventListener('click',()=>{selected=null;input.value=btn.dataset.query;$('clear').hidden=false;input.focus();doSearch(input.value);}));
$('analyze').addEventListener('click',async()=>{
  showError('');
  if(!selected){showError('Pesquise e selecione uma ação da lista para continuar.');input.focus();if(input.value.trim().length>=2)doSearch(input.value);return;}
  const symbol=selected.symbol,horizon=$('horizon').value;
  suggestions.hidden=true;$('loading').hidden=false;$('analyze').disabled=true;$('empty').hidden=true;$('results').hidden=true;
  try{const data=await getJSON('/api/analyze?symbol='+encodeURIComponent(symbol)+'&horizon='+horizon);currentResult=data;render(data);$('results').hidden=false;}
  catch(e){showError(e.message);clearResults();}
  finally{$('loading').hidden=true;$('analyze').disabled=false;}
});
function render(data){
 const m=data.metrics;
 $('asset-name').textContent=data.name; $('asset-symbol').textContent=data.symbol.replace(/\.SA$/,'');$('asset-date').textContent='Dados até '+dateBR(data.date);
 $('probability').textContent=(m.probability*100).toLocaleString('pt-BR',{maximumFractionDigits:1});$('prob-bar').style.width=(m.probability*100)+'%';
 const hue=Math.round(Math.max(0,Math.min(1,m.probability))*120);document.querySelector('.probability-card').style.setProperty('--prob-bg',`hsl(${hue} 55% 27%)`);
 $('prob-copy').textContent=`Chance estimada de o fechamento ajustado ser maior em ${data.horizon} pregões. Não é uma garantia de alta.`;
 $('price').textContent=money(data.price,data.currency);$('projected-price').textContent=money(data.projected_price,data.currency);
 $('projected-copy').textContent=`Estimativa para ${data.horizon} pregões. Valor no referencial ajustado; não é preço garantido.`;
 showQuote(data.quote,data.currency);
 $('accuracy').textContent=pct(m.accuracy);$('brier').textContent=m.brier.toLocaleString('pt-BR',{minimumFractionDigits:3,maximumFractionDigits:3});$('baseline').textContent=m.baseline_brier.toLocaleString('pt-BR',{minimumFractionDigits:3,maximumFractionDigits:3});$('sample').textContent=m.count.toLocaleString('pt-BR');
 $('test-date').textContent=`Testes: ${dateBR(m.start)} a ${dateBR(m.end)} (datas das previsões)`;
 const comparison=m.brier < m.baseline_brier ? 'Nos testes, o Brier do modelo foi menor que o da referência simples.' : 'Nos testes, o modelo não superou a referência simples pelo Brier.';
 $('assessment').textContent=comparison+' Isso não garante que o desempenho se repita. Confira também a calibração abaixo.';
 const better=m.price_mae<m.price_baseline_mae?'superou':'não superou';
 $('price-assessment').textContent=`Projeção de preço: nos testes, o erro médio de retorno foi ${pct(m.price_mae)}; o modelo ${better} a hipótese de preço inalterado (erro de ${pct(m.price_baseline_mae)}). Estes erros são em pontos percentuais do preço de referência, não garantia para o futuro.`;
 const tbody=$('calibration');tbody.replaceChildren();for(const b of m.bins){const tr=document.createElement('tr');for(const val of [`${b.min}% a ${b.max}%`,b.count,pct(b.predicted),pct(b.observed)]){const td=document.createElement('td');td.textContent=val;tr.append(td);}tbody.append(tr);}
 document.querySelectorAll('[data-days]').forEach(b=>b.classList.toggle('active',b.dataset.days==='252'));drawChart(252);
}
function drawChart(days){
 if(!currentResult)return;
 const values=currentResult.chart.slice(-days);if(values.length<2)return;
 const projected=currentResult.projected_price, last=values[values.length-1].close;
 const w=720,h=200,left=54,right=18,top=12,bottom=29,pw=w-left-right,ph=h-top-bottom;
 const vals=values.map(v=>v.close).concat([projected]);let min=Math.min(...vals),max=Math.max(...vals);const pad=(max-min)*.12||1;min-=pad;max+=pad;
 const historyWidth=.75, x=i=>left+(i/(values.length-1))*pw*historyWidth, futureX=left+pw, y=v=>top+(max-v)/(max-min)*ph;
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('preserveAspectRatio','none');svg.setAttribute('role','img');svg.setAttribute('aria-label','Histórico ajustado e linha ilustrativa da projeção até o fim do prazo');
 function node(tag,attrs,text){const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;svg.append(el);return el;}
 const defs=document.createElementNS(ns,'defs');defs.innerHTML='<linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#9fc99a" stop-opacity="0.33"/><stop offset="100%" stop-color="#9fc99a" stop-opacity="0.02"/></linearGradient>';svg.append(defs);
 for(let i=0;i<4;i++){let yy=top+i*ph/3;node('line',{x1:left,y1:yy,x2:w-right,y2:yy,stroke:'#edf2eb','stroke-width':'1'});node('text',{x:left-9,y:yy+4,'text-anchor':'end',fill:'#9caea0','font-size':'10'},(max-i*(max-min)/3).toLocaleString('pt-BR',{maximumFractionDigits:2}));}
 const coords=values.map((v,i)=>`${x(i).toFixed(2)},${y(v.close).toFixed(2)}`);node('path',{d:`M ${coords.join(' L ')} L ${x(values.length-1)},${top+ph} L ${left},${top+ph} Z`,fill:'url(#chartFill)'});node('polyline',{points:coords.join(' '),fill:'none',stroke:'#4b8c64','stroke-width':'2.5','stroke-linecap':'round','stroke-linejoin':'round','vector-effect':'non-scaling-stroke'});
 node('line',{x1:x(values.length-1),y1:top,x2:x(values.length-1),y2:top+ph,stroke:'#d9e2d5','stroke-dasharray':'3 4'});
 node('line',{x1:x(values.length-1),y1:y(last),x2:futureX,y2:y(projected),stroke:'#c18e3d','stroke-width':'2.5','stroke-dasharray':'5 5','vector-effect':'non-scaling-stroke'});
 node('circle',{cx:futureX,cy:y(projected),r:3.5,fill:'#c18e3d'});
 [0,Math.floor((values.length-1)/2),values.length-1].forEach((i,j)=>{const [yr,mo]=values[i].date.split('-');node('text',{x:x(i),y:h-5,'text-anchor':j===0?'start':j===2?'end':'middle',fill:'#9caea0','font-size':'10'},`${mo}/${yr}`);});
 node('text',{x:futureX,y:h-5,'text-anchor':'end',fill:'#b2884e','font-size':'10'},`+${currentResult.horizon} pregões`);
 $('real-chart').replaceChildren(svg);
}
document.querySelectorAll('[data-days]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-days]').forEach(x=>x.classList.remove('active'));b.classList.add('active');drawChart(Number(b.dataset.days));}));

function money(value,currency='BRL'){return new Intl.NumberFormat('pt-BR',{style:'currency',currency:currency||'BRL'}).format(value);}
function showQuote(q,currency){
 if(!q){$('live-price').textContent='Indisponível';$('live-time').textContent='A fonte não informou uma cotação atual.';return;}
 $('live-price').textContent=money(q.price,currency);
 const d=new Date(q.time);$('live-time').textContent='Cotação em '+new Intl.DateTimeFormat('pt-BR',{timeZone:'America/Sao_Paulo',dateStyle:'short',timeStyle:'short'}).format(d)+' · pode ter atraso';
}
async function refreshLive(){if(!currentResult||$('results').hidden)return;const symbol=currentResult.symbol;try{const q=await getJSON('/api/quote?symbol='+encodeURIComponent(symbol));if(currentResult&&currentResult.symbol===symbol)showQuote(q,currentResult.currency);}catch(e){$('live-time').textContent='Falha ao atualizar: '+e.message;}}
setInterval(refreshLive,300000);
$('refresh-quote').addEventListener('click',()=>{if(!currentResult)return;selected={symbol:currentResult.symbol,name:currentResult.name};$('horizon').value=String(currentResult.horizon);$('analyze').click();});
let rankingRunning=false, rankingStop=false;
$('open-ranking').addEventListener('click',()=>{ $('ranking-horizon').textContent='Prazo: '+$('horizon').value+' pregões';$('ranking-dialog').showModal(); });
$('close-ranking').addEventListener('click',()=>{$('ranking-dialog').close();rankingStop=true;});
$('ranking-dialog').addEventListener('close',()=>{rankingStop=true;});
$('stop-ranking').addEventListener('click',()=>{rankingStop=true;$('rank-progress').textContent+=' Interrompendo após o lote atual...';});
function showRankList(id,items){
 const node=$(id);node.replaceChildren();if(!items.length){const p=document.createElement('p');p.textContent='Nenhuma análise válida até agora.';node.append(p);return;}
 items.forEach((item,i)=>{const row=document.createElement('div');row.className='rank-item';const info=document.createElement('div');const name=document.createElement('strong');name.textContent=`${i+1}. ${item.symbol.replace(/\.SA$/,'')}`;const date=document.createElement('small');date.textContent='Base: '+dateBR(item.date);info.append(name,date);const p=document.createElement('span');p.className='rank-percent';p.style.color=`hsl(${Math.round(item.probability*120)} 60% 35%)`;p.textContent=pct(item.probability);row.append(info,p);row.title='Clique para analisar esta ação';row.setAttribute('role','button');row.tabIndex=0;const open=()=>{selected={symbol:item.symbol,name:item.name};input.value=item.symbol.replace(/\.SA$/,'');$('clear').hidden=false;$('ranking-dialog').close();$('analyze').click();window.scrollTo({top:0,behavior:'smooth'});};row.addEventListener('click',open);row.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}});node.append(row);});
}
$('start-ranking').addEventListener('click',async()=>{
 if(rankingRunning)return;rankingRunning=true;rankingStop=false;const horizon=Number($('horizon').value);$('ranking-horizon').textContent='Prazo: '+horizon+' pregões';$('start-ranking').disabled=true;$('stop-ranking').hidden=false;$('rank-progress-bar').style.width='0%';$('rank-progress').textContent='Obtendo lista de ações da B3...';showRankList('rank-high',[]);showRankList('rank-low',[]);
 try{
  const universe=await getJSON('/api/universe');const items=universe.items;let results=[],attempted=0,failed=0,emptyBatches=0;
  for(let i=0;i<items.length&&!rankingStop;i+=4){
   const batch=items.slice(i,i+4);
   const data=await getJSON('/api/ranking?symbols='+encodeURIComponent(batch.map(x=>x.symbol).join(','))+'&horizon='+horizon);
   attempted+=data.attempted;failed+=data.failed;results.push(...data.results);emptyBatches=data.results.length?0:emptyBatches+1;
   results.sort((a,b)=>b.probability-a.probability);showRankList('rank-high',results.slice(0,15));showRankList('rank-low',[...results].reverse().slice(0,15));
   $('rank-progress-bar').style.width=(100*attempted/items.length)+'%';$('rank-progress').textContent=`PARCIAL · ${attempted} de ${items.length} ações verificadas · ${results.length} analisadas · ${failed} sem resultado. Fonte da lista: brapi.`;
   if(emptyBatches>=4){rankingStop=true;$('rank-progress').textContent+=' A verificação foi interrompida após várias falhas seguidas; a fonte pode estar indisponível ou limitando acessos.';break;}
   await new Promise(resolve=>setTimeout(resolve,300));
  }
  if(!rankingStop) $('rank-progress').textContent=`CONCLUÍDO · ${attempted} ações verificadas da lista brapi; ${results.length} analisadas e ${failed} sem resultado. Cobertura sujeita à fonte.`;
  else if(emptyBatches<4) $('rank-progress').textContent+=' Interrompido: resultados parciais.';
 }catch(e){$('rank-progress').textContent='Não foi possível concluir o ranking: '+e.message+' Os resultados exibidos, se houver, são parciais.';}
 finally{rankingRunning=false;$('start-ranking').disabled=false;$('stop-ranking').hidden=true;}
});
