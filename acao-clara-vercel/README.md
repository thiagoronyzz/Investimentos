# Ação Clara

Protótipo educacional para analisar ações brasileiras a partir do histórico de preços; não é recomendação de investimento.

## Executar localmente

Requer Python 3.12+ e acesso à internet:

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Acesse http://localhost:8000. Para mudar a porta, defina `PORT`.

## Fontes, atualização e limitações

A pesquisa, cotação e histórico vêm dos endpoints públicos não oficiais do Yahoo Finance. A cotação mais recente exibe a hora informada pela fonte e pode ter atraso; não é necessariamente em tempo real. A tela tenta atualizar a cotação a cada cinco minutos enquanto a análise está aberta. O botão Atualizar busca novamente a cotação, o histórico e a projeção. Os pedidos incluem proteção contra cache; ainda assim, a fonte pode retornar dados atrasados, revisados, incorretos ou indisponíveis. O histórico usa o campo `adjclose` da fonte, normalmente ajustado para dividendos e desdobramentos sem verificação independente. Durante o pregão, o modelo usa até o último fechamento disponível antes do dia atual. O fechamento atual passa a ser aceito 30 minutos após o fim do pregão, conforme horário fornecido pelo Yahoo; em caso de falha no horário, usa-se o fechamento anterior.

O ranking obtém uma lista de ativos classificados como ações pela brapi.dev, filtrando códigos no padrão aceito pela aplicação e excluindo fracionários. Não se garante que a lista da fonte cubra todas as ações da B3. Cada ativo requer histórico disponível e recente e cerca de três anos de dados; o sistema informa o progresso, a quantidade sem resultado e se é parcial. Pode levar muitos minutos ou ser interrompido por limites da fonte. O processamento ocorre enquanto o navegador mantém o ranking aberto. Pode gerar centenas de requisições; respeite limites, custos e licenças da hospedagem e dos provedores. Verifique os termos do Yahoo Finance e da brapi antes de uso comercial ou publicação pública.

## Método

O modelo de direção é uma regressão logística padronizada com retornos de 5, 20 e 60 pregões, volatilidade de 20 e distâncias das médias de 20 e 60, sempre calculados com dados disponíveis na data de referência. O volume é obtido quando fornecido, mas não é usado. Rótulo de alta: fechamento ajustado futuro estritamente maior; igualdade é “não subiu”. O ranking usa o mesmo modelo de direção, porém sem calcular validação individual para cada ativo; analise cada ação para consultar seu teste.

A projeção de preço ajustado usa outro modelo, regressão Ridge com as mesmas variáveis, para estimar o retorno futuro no prazo selecionado. O preço estimado é o fechamento ajustado de referência multiplicado por um mais o retorno estimado. Não é cotação futura garantida; eventos corporativos podem fazer com que preço negociado futuro e valor ajustado sejam diferentes. A linha tracejada é interpolação reta entre o último fechamento e a projeção final; não estima preços para pregões intermediários. Os dois modelos podem divergir na direção indicada. O retorno estimado é limitado ao intervalo de -95% a +300% para evitar resultados extremos e preços negativos.

Validação progressiva em quatro blocos cronológicos. Em cada bloco, o treino usa apenas amostras cujo rótulo já era conhecido antes do primeiro dia do teste, sem atualização dentro do bloco. Métricas fora da amostra: acerto direcional, Brier, comparação com a frequência de alta no treino e tabela de calibração. A previsão de retorno é avaliada pelo erro absoluto médio de retorno, comparado à hipótese de preço ajustado constante. Observações no tempo podem se sobrepor e não são independentes. O modelo final usa os rótulos conhecidos até o último fechamento disponível; testes não escolhem hiperparâmetros. Resultados passados não garantem resultados futuros. Esta análise clássica não considera notícias nem fatores externos; o **motor quantitativo** descrito na próxima seção incorpora macroeconomia, fundamentos, eventos, fluxo e interdependência entre mercados.

## Motor quantitativo (oito núcleos)

Além da análise clássica, o sistema projeta preço esperado, faixa e probabilidades
nos horizontes de **7, 30, 90, 180 e 365 dias** combinando oito núcleos
independentes. Não há inteligência artificial generativa: só matemática,
estatística, economia e regras programadas, todas auditáveis no código.

| Núcleo | O que mede |
| --- | --- |
| 1. Macroeconomia | Selic, Fed Funds, IPCA/CPI, PIB, desemprego, câmbio, treasuries, dívida, confiança e PMI — nível, direção e velocidade (variação, aceleração, desvio da média histórica, z-score) |
| 2. Fundamentos e valuation | P/L, P/VP, EV/EBITDA, ROE, ROIC, margens, crescimento, dívida líquida/EBITDA, FCF, DY e payout comparados com a própria história e com pares do setor; dois modelos próprios: **fluxo de caixa descontado** e **valuation relativo (comps)** |
| 3. Interdependência | Matriz de dependência por empresa (ex.: PETR4 ↔ petróleo, câmbio; VALE3 ↔ minério, China) com Pearson, Spearman, correlação móvel, regressão multivariada, defasagens e sensibilidade histórica — separando correlação de causalidade |
| 4. Eventos externos | Classificador por regras (guerras, sanções, eleições, mudanças tributárias, regulação, decisões de banco central, tarifas, M&A, desastres, greves, trocas de gestão) com intensidade 0–10, direção, setores afetados, horizonte, certeza, vigência e desconto do que **já está no preço** |
| 5. Fluxo financeiro | Volume relativo à média histórica, pressão de compra/venda (OBV, acumulação, MFI, avanços/declínios), participação estrangeira e institucional, aluguel e short interest quando disponíveis |
| 6. Modelagem estatística | Regressão múltipla OLS, Ridge, Lasso, ARIMA, GARCH(1,1), cointegração, Monte Carlo, modelo fatorial e regressão logística — coeficientes em janelas móveis, teste fora da amostra e controle de sobreajuste |
| 7. Cenários futuros | Cinco cenários (muito otimista → muito pessimista) com probabilidade, preço e retorno esperados, probabilidade de alta/queda, ganho e perda potenciais, intervalo de confiança e assimetria risco-retorno |
| 8. Risco e confiabilidade | Volatilidade histórica e projetada, drawdown esperado, beta, sensibilidade a choques macro, risco de cauda (VaR/CVaR), qualidade dos dados, divergência entre modelos, erro histórico e backtesting walk-forward |

**Três camadas de processamento.** (A) cada núcleo produz estimativa, incerteza e
confiança próprios; (B) os pesos são definidos dinamicamente pela precisão
histórica, estabilidade, aderência ao setor e ao horizonte, qualidade dos dados e
desempenho fora da amostra, com piso e teto para que nenhum núcleo domine — o
preenchimento é do tipo *water-filling*, então o teto é relaxado quando há poucos
núcleos ativos; (C) ajustes finais de risco e cenário (limites de retorno,
limite em desvios-padrão do horizonte e encolhimento proporcional à qualidade).

**Descoberta automática de variáveis.** O motor testa candidatos contemporâneos e
com defasagens de 1, 5, 10, 20 e 60 dias, mede estabilidade e significância,
valida o poder explicativo fora da amostra e aplica **correção de testes múltiplos
(Benjamini–Hochberg)** antes de aceitar uma variável.

**Qualidade dos dados.** Normalização, uso do histórico ajustado por dividendos e
desdobramentos, controle de faltantes, corte *point-in-time* (nada de olhar para o
futuro), backtesting, pesos dinâmicos, controle de correlação entre modelos e
detecção de mudança de regime.

**Aprendizado com os próprios erros.** Cada projeção é gravada num livro SQLite
(`runtime/forecasts.db`); quando o prazo vence, o preço realizado é confrontado com
o previsto e o erro recalibra os pesos da Camada B e a precisão exibida na tela.
Em hospedagem sem disco persistente o livro é recriado a cada instância e o motor
passa a usar apenas o backtesting walk-forward.

### Endpoints do motor

| Rota | Conteúdo |
| --- | --- |
| `/api/quant?symbol=&horizon=&mode=full` | Projeção completa: preço esperado, faixa, cenários, contribuição de cada núcleo, drivers, eventos, macro, risco e aprendizado |
| `/api/backtest?symbol=&horizon=&count=` | Validação walk-forward (origens históricas, erro por núcleo e do conjunto) |
| `/api/macro` | Painel macroeconômico com nível, direção, velocidade e verificação de fonte |
| `/api/ledger` | Livro de previsões: precisão registrada e histórico |
| `/api/health` | Versão do motor, horizontes, estado das fontes e do livro |

Os endpoints clássicos continuam disponíveis: `/api/search`, `/api/analyze`,
`/api/quote`, `/api/universe` e `/api/ranking`.

### Variáveis de ambiente

| Variável | Efeito |
| --- | --- |
| `PORT` | Porta do servidor (padrão 8000) |
| `ACAO_CLARA_DEMO=1` | Modo demonstração: séries sintéticas determinísticas, sem acesso externo. Útil para testar sem internet |
| `ACAO_CLARA_LEDGER=0` | Desliga o livro de previsões |
| `ACAO_CLARA_LEDGER_PATH` | Caminho alternativo do SQLite |
| `BRAPI_TOKEN` | Token da brapi.dev para fundamentalos |

## Testes

```bash
python -m unittest discover -s tests
```

São duas suítes: `tests/test_model.py` (análise clássica) e `tests/test_engine.py`
(motor quantitativo — estatística, cenários, valuation, núcleos, Camadas A/B/C,
segurança contra *look-ahead*, projeção, livro de previsões, walk-forward, pesos
por horizonte e rotas da API). Ambas rodam offline com `ACAO_CLARA_DEMO=1`.

## Publicar na Vercel

Esta versão exporta uma aplicação Flask (`app` em `app.py`) para a API e usa `public/` para os arquivos públicos. A pasta `static/` atende o servidor local; mantenha as duas pastas sincronizadas se modificar o frontend. Faça upload do conteúdo deste pacote na raiz do repositório e importe-o na Vercel com o framework Flask. Se já estiver conectado, um novo commit normalmente dispara outro deploy. Teste todos os endpoints após publicar: o Yahoo ou a brapi podem limitar os acessos vindos da hospedagem e o ranking demanda centenas de requisições e tempo de execução. Confira os logs se houver erros.
