/* Motor quantitativo — painel dos oito núcleos.
   Consome /api/quant, /api/backtest, /api/macro e /api/ledger.
   Nenhum valor é inventado aqui: o que a API não envia é exibido como indisponível. */
(function () {
  'use strict';
  const state = { symbol: null, name: null, horizon: 90, data: null, running: false };

  const brl = (value, currency) => (value === null || value === undefined || Number.isNaN(value))
    ? '—'
    : new Intl.NumberFormat('pt-BR', { style: 'currency', currency: currency || 'BRL' }).format(value);
  const pct = (value, digits) => (value === null || value === undefined || Number.isNaN(value))
    ? '—'
    : (value).toLocaleString('pt-BR', { minimumFractionDigits: digits ?? 1, maximumFractionDigits: digits ?? 1 }) + '%';
  const num = (value, digits) => (value === null || value === undefined || Number.isNaN(value))
    ? '—'
    : Number(value).toLocaleString('pt-BR', { minimumFractionDigits: digits ?? 2, maximumFractionDigits: digits ?? 2 });
  const dateBR = (value) => { if (!value) return '—'; const [y, m, d] = String(value).slice(0, 10).split('-'); return `${d}/${m}/${y}`; };
  const sign = (value) => (value > 0 ? '+' : '');

  function el(tag, props, children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (key === 'class') node.className = value;
        else if (key === 'text') node.textContent = value;
        else if (key === 'html') node.innerHTML = value;
        else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
        else if (value !== null && value !== undefined) node.setAttribute(key, value);
      }
    }
    (Array.isArray(children) ? children : children ? [children] : []).forEach((child) => {
      if (child === null || child === undefined) return;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    });
    return node;
  }

  function block(span, title, hint, body) {
    const node = el('div', { class: `quant-block ${span}` });
    if (title) node.append(el('h3', { text: title }));
    if (hint) node.append(el('p', { class: 'hint', text: hint }));
    if (body) node.append(body);
    return node;
  }

  function barRow(name, sub, value, max, suffix) {
    const width = Math.max(2, Math.min(50, (Math.abs(value) / (max || 1)) * 50));
    const negative = value < 0;
    const bar = el('i', { class: negative ? 'neg' : '' });
    bar.style.width = width + '%';
    if (negative) bar.style.left = (50 - width) + '%';
    return el('div', { class: 'bar-row' }, [
      el('div', { class: 'name' }, [name, sub ? el('small', { text: sub }) : null]),
      el('div', { class: 'bar' }, bar),
      el('div', { class: 'value', text: `${sign(value)}${num(value, 2)}${suffix || ' p.p.'}` })
    ]);
  }

  function pill(text, kind) { return el('span', { class: `pill-mini pill-${kind || 'flat'}`, text }); }

  function table(headers, rows, numericFrom) {
    const head = el('tr', {}, headers.map((header, index) => el('th', { class: index >= (numericFrom ?? headers.length) ? 'num' : '', text: header })));
    const body = rows.map((row) => el('tr', {}, row.map((cell, index) => {
      if (cell && cell.nodeType) return el('td', { class: index >= (numericFrom ?? headers.length) ? 'num' : '' }, cell);
      return el('td', { class: index >= (numericFrom ?? headers.length) ? 'num' : '', html: String(cell ?? '—') });
    })));
    const tableNode = el('table', { class: 'quant-table' }, [el('thead', {}, head), el('tbody', {}, body)]);
    const scroll = el('div', { class: 'table-scroll' }, tableNode);
    return scroll;
  }

  /* ------------------------------------------------------------------ KPIs */
  function kpiRow(data) {
    const projection = data.projection;
    const up = (projection.expected_return || 0) >= 0;
    const wrap = el('div', { class: 'quant-grid' });
    wrap.append(
      block('span-3', null, null, kpi('PREÇO ATUAL (AJUSTADO)', brl(data.price, data.currency), `Base ${dateBR(data.as_of)}`)),
      block('span-3', null, null, kpi(`PREÇO ESPERADO EM ${data.horizon_days} DIAS`, brl(projection.expected_price, data.currency), `${data.horizon_label} · ${data.horizon_trading_days} pregões`, up ? 'positive' : 'negative')),
      block('span-3', null, null, kpi('RETORNO ESPERADO', `${sign(projection.expected_return_pp)}${num(projection.expected_return_pp, 2)}%`, `Probabilidade de alta ${pct((projection.probability_up || 0) * 100, 1)}`, up ? 'positive' : 'negative')),
      block('span-3', null, null, kpi('FAIXA PROJETADA (P10–P90)', `${brl(projection.range?.low, data.currency)} – ${brl(projection.range?.high, data.currency)}`, `Confiabilidade ${num(data.risk?.confidence, 0)}/100 (${data.risk?.confidence_label || '—'})`))
    );
    return wrap;
  }

  function kpi(label, value, note, tone) {
    return el('div', { class: `kpi ${tone || ''}` }, [el('small', { text: label }), el('strong', { text: value }), note ? el('span', { text: note }) : null]);
  }

  /* -------------------------------------------------------------- Cenários */
  function scenarioBlock(data) {
    const scenarios = data.projection.scenarios || [];
    const body = el('div');
    if (!scenarios.length) { body.append(el('p', { class: 'hint', text: 'Simulação indisponível para este ativo.' })); return body; }
    const max = Math.max(...scenarios.map((item) => Math.abs(item.return_pp)));
    scenarios.forEach((item) => {
      const width = Math.max(3, (Math.abs(item.return_pp) / (max || 1)) * 100);
      const track = el('i', { class: item.return_pp < 0 ? 'neg' : '' });
      track.style.width = width + '%';
      body.append(el('div', { class: 'scenario-row' }, [
        el('div', { class: 'label', text: item.name }),
        el('div', { class: 'track' }, [track, el('span', { text: pct(item.probability * 100, 0) })]),
        el('div', { class: 'numbers' }, [el('b', { text: brl(item.price, data.currency) }), `${sign(item.return_pp)}${num(item.return_pp, 1)}% · ${brl(item.from, data.currency)}–${brl(item.to, data.currency)}`])
      ]));
    });
    const projection = data.projection;
    body.append(el('div', { class: 'two-col', style: 'margin-top:14px' }, [
      el('div', {}, [el('small', { class: 'section-kicker', text: 'ASSIMETRIA RISCO/RETORNO' }), el('p', { class: 'hint', style: 'margin:6px 0 0', text: `Ganho potencial ${brl(projection.potential_gain, data.currency)} · perda potencial ${brl(projection.potential_loss, data.currency)} · relação ${num(projection.risk_reward, 2)}` })]),
      el('div', {}, [el('small', { class: 'section-kicker', text: 'CAUDA DA DISTRIBUIÇÃO' }), el('p', { class: 'hint', style: 'margin:6px 0 0', text: `VaR 95% ${num(projection.var95_pp, 1)}% · CVaR 95% ${num(projection.cvar95_pp, 1)}%` })])
    ]));
    return body;
  }

  /* --------------------------------------------------- Contribuição/pesos */
  function contributionBlock(data) {
    const body = el('div');
    const rows = (data.combined?.contributions || []).slice();
    const max = Math.max(...rows.map((item) => Math.abs(item.contribution_pp)), 0.01);
    rows.forEach((item) => {
      body.append(barRow(item.label, `peso ${pct(item.weight * 100, 1)} · núcleo ${sign(item.return_pp)}${num(item.return_pp, 2)}%`, item.contribution_pp, max));
    });
    if (!rows.length) body.append(el('p', { class: 'hint', text: 'Nenhum núcleo produziu estimativa.' }));
    const detail = el('details', { class: 'quant-details' }, el('summary', { text: 'Como cada peso foi definido' }));
    const weightRows = Object.entries(data.weights || {}).map(([core, item]) => [
      item.label || core,
      pct((item.weight_base || 0) * 100, 1),
      num(item.confidence, 2),
      item.historical_skill === null || item.historical_skill === undefined ? 'sem registro' : num(item.historical_skill * 100, 0) + '%',
      num(item.multiplier, 2),
      pct((item.weight || 0) * 100, 1)
    ]);
    detail.append(table(['Núcleo', 'Peso-base', 'Confiança', 'Precisão histórica', 'Multiplicador', 'Peso final'], weightRows, 1));
    body.append(detail);
    return body;
  }

  /* -------------------------------------------------------- Pressões up/down */
  function pressureBlock(data) {
    const signals = [];
    (data.cores || []).forEach((core) => (core.signals || []).forEach((signal) => signals.push({ ...signal, core: core.label })));
    signals.sort((a, b) => Math.abs(b.contribution_pp) - Math.abs(a.contribution_pp));
    const up = signals.filter((item) => item.contribution_pp > 0).slice(0, 8);
    const down = signals.filter((item) => item.contribution_pp < 0).slice(0, 8);
    const list = (items) => {
      const wrap = el('div', { class: 'factor-list' });
      if (!items.length) wrap.append(el('p', { class: 'hint', text: 'Nenhum fator relevante nesta direção.' }));
      items.forEach((item) => {
        wrap.append(el('div', { class: `factor ${item.contribution_pp > 0 ? 'up' : 'down'}` }, [
          el('div', { class: 'title', text: item.name }),
          el('div', { class: 'amount', text: `${sign(item.contribution_pp)}${num(item.contribution_pp, 2)} p.p.` }),
          el('div', { class: 'detail', text: `${item.core}${item.detail ? ' · ' + item.detail : ''}` })
        ]));
      });
      return wrap;
    };
    const wrap = el('div', { class: 'two-col' }, [list(up), list(down)]);
    return wrap;
  }

  /* --------------------------------------------------- Matriz de dependência */
  function driverBlock(data) {
    const matrix = data.drivers?.matrix || {};
    const keys = Object.keys(matrix);
    if (!keys.length) return el('p', { class: 'hint', text: 'Nenhuma variável externa pôde ser alinhada a este ativo.' });
    const rows = keys.map((key) => {
      const item = matrix[key];
      const move = item.expected_move || {};
      return [
        `${item.label}<br><small style="color:#94a498">defasagem ${item.best_lag}d · n=${item.n}</small>`,
        num(item.pearson, 3),
        num(item.spearman, 3),
        num(item.stability === null ? null : item.stability * 100, 0) + '%',
        item.cointegration?.cointegrated ? pill('cointegrado', 'up') : (item.cointegration?.pvalue === null ? '—' : pill('não', 'flat')),
        num(item.beta, 3),
        move.ok ? `${sign(move.expected_pct * 100)}${num(move.expected_pct * 100, 2)}%` : '—',
        num(item.contribution_pp, 2)
      ];
    });
    return table(['Variável externa', 'Pearson', 'Spearman', 'Estabilidade', 'Longo prazo', 'Beta', 'Movimento esperado', 'Contribuição'], rows, 1);
  }

  /* ------------------------------------------------------------- Descoberta */
  function discoveryBlock(data) {
    const found = data.drivers?.discovery || {};
    const body = el('div');
    if (!found.ok) { body.append(el('p', { class: 'hint', text: `Descoberta de variáveis indisponível${found.reason ? ': ' + found.reason : '.'}` })); return body; }
    body.append(el('p', { class: 'hint', text: `${found.tests} testes em ${found.candidates} variáveis candidatas · ${found.accepted_before_filters} aprovados no controle de falsa descoberta (BH, q=${num((found.fdr_q || 0) * 100, 0)}%).` }));
    const discoveries = found.discoveries || [];
    if (!discoveries.length) {
      body.append(el('p', { class: 'hint', text: 'Nenhuma relação passou simultaneamente pelos filtros de significância, estabilidade entre subperíodos e desempenho fora da amostra.' }));
      return body;
    }
    const rows = discoveries.map((item) => [
      item.variable.replace('driver:', '').replace('macro:', 'macro: '),
      item.lag,
      num(item.corr, 3),
      num((item.pvalue || 0) * 100, 3) + '%',
      num(item.stability * 100, 0) + '%',
      num(item.oos_r2 * 100, 1) + '%'
    ]);
    body.append(table(['Variável descoberta', 'Defasagem', 'Correlação', 'p-valor', 'Estabilidade', 'R² fora da amostra'], rows, 1));
    return body;
  }

  /* ---------------------------------------------------------------- Eventos */
  function eventsBlock(data) {
    const events = data.events || [];
    const body = el('div');
    if (!events.length) { body.append(el('p', { class: 'hint', text: 'Nenhum acontecimento classificável por regras foi encontrado para esta empresa.' })); return body; }
    const list = el('div', { class: 'factor-list' });
    events.slice(0, 8).forEach((item) => {
      const impact = (item.impact || 0) * 100;
      list.append(el('div', { class: `factor ${impact > 0 ? 'up' : impact < 0 ? 'down' : ''}` }, [
        el('div', { class: 'title', text: item.title }),
        el('div', { class: 'amount', text: `${sign(impact)}${num(impact, 2)} p.p.` }),
        el('div', { class: 'detail', text: `${item.label} · ${dateBR(item.date)} (${item.age_days} dias) · intensidade ${num(item.intensity, 0)}/10 · certeza ${pct((item.certainty || 0) * 100, 0)} · parcela já precificada ${pct((1 - (item.priced_in_adjustment ?? 1)) * 100, 0)}${item.matched_keywords?.length ? ' · termos: ' + item.matched_keywords.join(', ') : ''}` })
      ]));
    });
    body.append(list);
    return body;
  }

  /* ------------------------------------------------------------------ Macro */
  function macroBlock(data) {
    const indicators = data.macro || {};
    const keys = Object.keys(indicators);
    if (!keys.length) return el('p', { class: 'hint', text: 'Nenhum indicador macroeconômico disponível.' });
    const rows = keys.map((key) => {
      const item = indicators[key];
      const trend = item.trend === 'alta' ? pill('alta', 'up') : item.trend === 'queda' ? pill('queda', 'down') : pill('estável', 'flat');
      return [
        `${item.label}<br><small style="color:#94a498">${item.source}${item.verified === false ? ' · nome não confirmado na fonte' : ''}</small>`,
        num(item.last, 2) + (item.unit ? ' ' + item.unit : ''),
        item.zscore === null ? '—' : num(item.zscore, 2) + 'σ',
        trend,
        num(item.direction_per_period, 3),
        num(item.acceleration, 4),
        item.change_3m === null ? '—' : sign(item.change_3m) + num(item.change_3m, 2)
      ];
    });
    return table(['Indicador', 'Valor atual', 'Desvio da média', 'Direção', 'Velocidade', 'Aceleração', 'Δ 3 meses'], rows, 1);
  }

  /* ------------------------------------------------------------------ Risco */
  function riskBlock(data) {
    const risk = data.risk || {};
    const body = el('div');
    const grid = el('div', { class: 'two-col' });
    const left = el('div');
    left.append(el('div', { class: 'kpi' }, [el('small', { text: 'CONFIABILIDADE DA ESTIMATIVA' }), el('strong', { text: `${num(risk.confidence, 0)} / 100` }), el('span', { text: `Classificação: ${risk.confidence_label || '—'}` })]));
    const gauge = el('i'); gauge.style.width = Math.max(2, Math.min(100, risk.confidence || 0)) + '%';
    left.append(el('div', { class: 'gauge' }, gauge));
    const components = Object.entries(risk.confidence_components || {}).map(([key, value]) => [key.replace(/_/g, ' '), pct(value * 100, 0)]);
    left.append(table(['Componente da nota', 'Peso relativo'], components, 1));
    const right = el('div');
    const vol = risk.volatility || {};
    right.append(table(['Métrica de risco', 'Valor'], [
      ['Volatilidade anualizada (252 pregões)', pct((vol['252'] || 0) * 100, 1)],
      ['Volatilidade projetada (GARCH)', vol.projected === null || vol.projected === undefined ? 'indisponível' : pct(vol.projected * 100, 1)],
      ['Beta em relação ao Ibovespa', risk.beta === null ? '—' : num(risk.beta, 2)],
      ['Drawdown médio esperado na simulação', risk.expected_max_drawdown === null || risk.expected_max_drawdown === undefined ? '—' : pct(risk.expected_max_drawdown * 100, 1)],
      ['Divergência entre os núcleos', risk.model_divergence === null ? '—' : pct(risk.model_divergence * 100, 2)],
      ['Regime de volatilidade', `${risk.regime?.regime || '—'} (z=${num(risk.regime?.vol_zscore, 2)})`]
    ], 1));
    const shocks = (risk.macro_shocks || []).slice(0, 6).map((item) => [item.variable, num(item.beta, 3), `${sign(item.impact_pp)}${num(item.impact_pp, 2)} p.p.`]);
    if (shocks.length) {
      right.append(el('p', { class: 'hint', style: 'margin:12px 0 4px', text: 'Choque de 1 desvio-padrão em cada variável externa:' }));
      right.append(table(['Variável', 'Beta', 'Impacto no preço'], shocks, 1));
    }
    grid.append(left, right);
    body.append(grid);
    const quality = risk.data_quality || {};
    body.append(el('p', { class: 'hint', style: 'margin-top:12px', text: `Qualidade dos dados: ${quality.cores_ok}/${quality.cores_total} núcleos ativos · ${quality.drivers} variáveis externas · ${quality.macro_series} séries macro · ${quality.events} eventos · histórico de ${num(quality.history_years, 1)} anos · ${quality.macro_verified_ratio === null ? 'verificação de fontes não aplicável' : 'fontes verificadas ' + pct(quality.macro_verified_ratio * 100, 0)}` }));
    const warnings = risk.warnings || [];
    if (warnings.length) body.append(el('ul', { class: 'warn-list' }, warnings.map((text) => el('li', { text }))));
    return body;
  }

  /* --------------------------------------------------- Precisão e aprendizado */
  function learningBlock(data) {
    const learning = data.learning || {};
    const body = el('div');
    if (!learning.enabled) { body.append(el('p', { class: 'hint', text: 'O livro de previsões não está disponível nesta instância; o motor usa apenas o backtesting.' })); return body; }
    const accuracy = learning.accuracy || {};
    const stats = learning.stats || {};
    body.append(el('p', { class: 'hint', text: `Livro de previsões: ${stats.total || 0} projeções registradas (${stats.scored || 0} já conferidas contra o preço realizado) para ${stats.symbols || 0} ativos.` }));
    if (accuracy.ok) {
      body.append(table(['Métrica registrada', 'Valor'], [
        ['Previsões conferidas (ativo e horizonte)', accuracy.n],
        ['Erro médio absoluto da projeção', num(accuracy.mae_pp, 2) + ' p.p.'],
        ['Erro da referência (preço inalterado)', num(accuracy.baseline_mae_pp, 2) + ' p.p.'],
        ['Superou a referência?', accuracy.better_than_baseline ? pill('sim', 'up') : pill('não', 'down')],
        ['Acerto direcional', pct(accuracy.directional_hit * 100, 1)],
        ['Brier das probabilidades', accuracy.brier === null ? '—' : num(accuracy.brier, 3)]
      ], 1));
      const modelRows = Object.entries(accuracy.models || {}).map(([name, item]) => [name, num(item.mae_pp, 2) + ' p.p.', item.n, pct((item.skill || 0) * 100, 0)]);
      if (modelRows.length) {
        body.append(el('p', { class: 'hint', style: 'margin:12px 0 4px', text: 'Erro registrado por núcleo — é o que realimenta os pesos dinâmicos:' }));
        body.append(table(['Núcleo', 'Erro médio', 'Amostra', 'Habilidade'], modelRows, 1));
      }
    } else {
      body.append(el('p', { class: 'hint', text: 'Ainda não há previsões maduras para conferir. Cada projeção feita aqui é gravada e comparada ao preço realizado quando o prazo vencer.' }));
    }
    const change = learning.change_since_last;
    if (change) {
      body.append(el('p', { class: 'hint', style: 'margin-top:12px', text: `Mudança desde a última projeção (${change.previous_created_at}): preço esperado anterior ${brl(change.previous_expected_price, data.currency)} (${sign(change.previous_expected_return_pp)}${num(change.previous_expected_return_pp, 2)}%) → atual ${brl(change.current_expected_price, data.currency)}; variação de ${sign(change.delta_pp)}${num(change.delta_pp, 2)} p.p.` }));
    }
    return body;
  }

  function backtestBlock(result) {
    const body = el('div');
    if (!result || !result.ok) { body.append(el('p', { class: 'hint', text: (result && result.reason) || 'Backtesting indisponível.' })); return body; }
    const summary = result.ensemble || {};
    body.append(table(['Métrica do conjunto', 'Valor'], [
      ['Origens testadas', summary.n],
      ['Erro médio absoluto', num(summary.mae_pp, 2) + ' p.p.'],
      ['Erro da referência (preço inalterado)', num(summary.baseline_mae_pp, 2) + ' p.p.'],
      ['Superou a referência?', summary.better_than_baseline ? pill('sim', 'up') : pill('não', 'down')],
      ['Acerto direcional', pct((summary.hit_rate || 0) * 100, 1)],
      ['Coeficiente de informação', summary.information_coefficient === null ? '—' : num(summary.information_coefficient, 3)]
    ], 1));
    const rows = (result.origins || []).map((row) => [
      dateBR(row.origin),
      `${sign(row.expected_return_pp)}${num(row.expected_return_pp, 2)}%`,
      `${sign(row.realized_return_pp)}${num(row.realized_return_pp, 2)}%`,
      num(row.error_pp, 2) + ' p.p.',
      row.hit ? pill('acertou', 'up') : pill('errou', 'down')
    ]);
    body.append(el('p', { class: 'hint', style: 'margin:12px 0 4px', text: 'O que o motor teria projetado em cada data, usando apenas o que existia até então:' }));
    body.append(table(['Data-base', 'Projetado', 'Realizado', 'Erro', 'Direção'], rows, 1));
    const coreRows = Object.entries(result.cores || {}).map(([core, item]) => [item.label || core, num(item.mae_pp, 2) + ' p.p.', pct((item.hit_rate || 0) * 100, 1), item.n]);
    if (coreRows.length) {
      body.append(el('p', { class: 'hint', style: 'margin:12px 0 4px', text: 'Desempenho por núcleo nas mesmas origens:' }));
      body.append(table(['Núcleo', 'Erro médio', 'Acerto direcional', 'Origens'], coreRows, 1));
    }
    body.append(el('p', { class: 'hint', style: 'margin-top:10px', text: result.note || '' }));
    return body;
  }

  /* ---------------------------------------------------------------- Núcleos */
  function coresBlock(data) {
    const rows = (data.cores || []).map((core) => [
      `${core.label}<br><small style="color:#94a498">${core.status === 'ok' ? 'estimativa válida' : core.status === 'skip' ? 'ignorado' : 'falhou'}</small>`,
      core.expected_return_pp === null ? '—' : `${sign(core.expected_return_pp)}${num(core.expected_return_pp, 2)}%`,
      core.sigma === null ? '—' : pct(core.sigma * 100, 2),
      pct(core.confidence * 100, 0),
      pct(core.weight * 100, 1),
      core.error ? pill('erro', 'down') : (core.notes && core.notes.length ? pill('observações', 'warn') : pill('ok', 'up'))
    ]);
    return table(['Núcleo', 'Retorno estimado', 'Incerteza (σ)', 'Confiança', 'Peso', 'Situação'], rows, 1);
  }

  function coreNotes(data) {
    const notes = [];
    (data.cores || []).forEach((core) => (core.notes || []).forEach((note) => notes.push(`${core.label}: ${note}`)));
    (data.adjustments || []).forEach((item) => notes.push(`Camada C — ${item.step}: ${item.value_pp} p.p. (${item.detail})`));
    if (!notes.length) return null;
    return el('ul', { class: 'note-list' }, notes.map((note) => el('li', { text: note })));
  }

  /* -------------------------------------------------------------- Renderização */
  function setStatus(text) { const node = document.getElementById('quant-status'); if (node) node.textContent = text || ''; }

  function render(data) {
    const body = document.getElementById('quant-body');
    if (!body) return;
    body.replaceChildren();
    const projection = data.projection || {};
    const grid = el('div', { class: 'quant-grid' });

    grid.append(block('span-7', 'Distribuição de cenários', 'Cinco faixas de preço com as probabilidades obtidas pela simulação recentrada na combinação dos núcleos.', scenarioBlock(data)));
    grid.append(block('span-5', 'Contribuição dos modelos', 'Quanto cada núcleo acrescentou ao preço esperado, já com os pesos dinâmicos.', contributionBlock(data)));
    grid.append(block('span-12', 'O que pressiona o preço', 'Fatores que mais empurraram a projeção para cima e para baixo, com o núcleo de origem.', pressureBlock(data)));
    grid.append(block('span-12', 'Matriz de interdependência', 'Correlações na mesma janela do horizonte, defasagens, estabilidade, cointegração e a sensibilidade estimada por regressão.', driverBlock(data)));
    grid.append(block('span-6', 'Descoberta automática de variáveis', 'Relações encontradas em variáveis que não estavam programadas, após correção de múltiplos testes.', discoveryBlock(data)));
    grid.append(block('span-6', 'Eventos classificados por regras', 'Acontecimentos recentes e o impacto estimado com decaimento no tempo.', eventsBlock(data)));
    grid.append(block('span-12', 'Painel macroeconômico', 'Situação atual, direção e velocidade de cada indicador, com o desvio em relação à própria média histórica.', macroBlock(data)));
    grid.append(block('span-12', 'Risco, confiabilidade e validação', 'Volatilidade, beta, cauda da distribuição, choque de variáveis externas e a nota de confiabilidade da projeção.', riskBlock(data)));
    grid.append(block('span-12', 'Aprendizado com os próprios erros', 'Cada projeção é registrada e depois confrontada com o preço realizado; o erro recalibra os pesos.', learningBlock(data)));
    const backtestNode = block('span-12', 'Backtesting walk-forward', 'O motor volta no tempo, projeta com os dados disponíveis em cada data e compara com o que aconteceu.', el('div', { id: 'quant-backtest-body' }, el('p', { class: 'hint', text: 'Clique em “Validar no passado” para executar. Pode levar alguns segundos.' })));
    grid.append(backtestNode);
    grid.append(block('span-12', 'Resumo dos oito núcleos', 'Estimativa individual, incerteza, confiança e peso de cada núcleo nesta execução.', coresBlock(data)));

    body.append(kpiRow(data));
    body.append(el('div', { style: 'height:16px' }));
    body.append(grid);

    const notes = coreNotes(data);
    const foot = el('div', { class: 'quant-foot' });
    foot.append(el('p', {}, [el('strong', { text: 'Como ler este painel. ' }), data.disclaimer || '']));
    foot.append(el('p', { text: `Fontes consultadas nesta execução: ${Object.entries(data.sources?.sources || {}).map(([name, item]) => `${name} (${item.status})`).join(', ') || 'não informado'}. Dados de preço: Yahoo Finance (fechamento ajustado). Fundamentos e lista de ativos: brapi.dev. Indicadores macroeconômicos: Banco Central do Brasil (SGS) e FRED.` }));
    foot.append(el('p', { text: `Contexto: ${data.context?.bars || 0} pregões · ${data.context?.drivers?.length || 0} variáveis externas · ${data.context?.macro_series?.length || 0} séries macro · ${data.context?.events || 0} eventos · gerado em ${data.generated_at?.replace('T', ' ').slice(0, 19) || '—'}.` }));
    if (data.demo) foot.append(el('p', {}, [el('span', { class: 'quant-demo', text: 'MODO DEMONSTRAÇÃO — DADOS SINTÉTICOS' }), ' Todos os preços, indicadores e notícias exibidos são fictícios, gerados para exercitar o motor sem acesso às fontes externas.']));
    if (notes) foot.append(notes);
    body.append(foot);
  }

  async function fetchJSON(url) {
    const response = await fetch(url);
    let payload = null;
    try { payload = await response.json(); } catch (error) { /* corpo não JSON */ }
    if (!response.ok) throw new Error((payload && payload.error) || 'Não foi possível concluir a solicitação.');
    return payload;
  }

  async function run() {
    if (!state.symbol || state.running) return;
    state.running = true;
    const body = document.getElementById('quant-body');
    body.replaceChildren(el('div', { class: 'quant-loading' }, [el('span', { class: 'spinner' }), el('span', { text: 'Executando os oito núcleos: macro, fundamentos, interdependência, eventos, fluxo, estatística, simulação e risco.' })]));
    setStatus('CALCULANDO…');
    try {
      const data = await fetchJSON(`/api/quant?symbol=${encodeURIComponent(state.symbol)}&horizon=${state.horizon}&mode=full`);
      state.data = data;
      render(data);
      setStatus(`ATUALIZADO ${new Date().toLocaleTimeString('pt-BR')}`);
    } catch (error) {
      body.replaceChildren(el('div', { class: 'quant-idle', text: `Não foi possível concluir a projeção quantitativa: ${error.message}` }));
      setStatus('ERRO');
    } finally {
      state.running = false;
    }
  }

  async function runBacktest() {
    if (!state.symbol) return;
    const button = document.getElementById('quant-backtest');
    const target = document.getElementById('quant-backtest-body');
    if (!target) return;
    button.disabled = true;
    target.replaceChildren(el('div', { class: 'quant-loading' }, [el('span', { class: 'spinner' }), el('span', { text: 'Reexecutando o motor em datas passadas…' })]));
    try {
      const result = await fetchJSON(`/api/backtest?symbol=${encodeURIComponent(state.symbol)}&horizon=${state.horizon}`);
      target.replaceChildren(backtestBlock(result));
    } catch (error) {
      target.replaceChildren(el('p', { class: 'hint', text: 'Falha no backtesting: ' + error.message }));
    } finally {
      button.disabled = false;
    }
  }

  function start(symbol, name) {
    state.symbol = symbol;
    state.name = name || symbol;
    const title = document.getElementById('quant-title');
    const sub = document.getElementById('quant-sub');
    const section = document.getElementById('quant');
    if (title) title.textContent = `Projeção quantitativa — ${state.name}`;
    if (sub) sub.textContent = 'Motor de oito núcleos: macroeconomia, fundamentos e valuation, interdependência entre mercados, eventos, fluxo financeiro, modelagem estatística, simulação de cenários e risco. Sem inteligência artificial: apenas matemática, estatística e regras programadas.';
    if (section) section.hidden = false;
    run();
  }

  function wire() {
    const horizons = document.getElementById('quant-horizons');
    if (horizons) {
      horizons.addEventListener('click', function (event) {
        const button = event.target.closest('button[data-qdays]');
        if (!button) return;
        horizons.querySelectorAll('button').forEach((item) => item.classList.toggle('active', item === button));
        state.horizon = Number(button.dataset.qdays);
        if (state.symbol) run();
      });
    }
    const refresh = document.getElementById('quant-refresh');
    if (refresh) refresh.addEventListener('click', run);
    const backtestButton = document.getElementById('quant-backtest');
    if (backtestButton) backtestButton.addEventListener('click', runBacktest);
    const macroButton = document.getElementById('quant-macro');
    if (macroButton) {
      macroButton.addEventListener('click', function () {
        const details = document.querySelectorAll('#quant-body details.quant-details');
        const target = Array.from(details).pop();
        if (target) { target.open = true; target.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
      });
    }
  }

  // O script pode ser carregado depois do DOMContentLoaded (ex.: defer/dynamic).
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire);
  else wire();

  window.AcaoClaraQuant = { start: start, run: run };
})();
