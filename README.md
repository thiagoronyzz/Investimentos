# Ação Clara

Ferramenta educacional para analisar ações brasileiras: projeções de preço, faixa
esperada, probabilidades, cenários e risco, sem recomendação de investimento.

## Estrutura do repositório

| Pasta | Conteúdo |
| --- | --- |
| `acao-clara/` | **Aplicação completa.** Servidor local (`app.py`, `http.server`), motor quantitativo (`engine/`), dados de apoio (`data/`), frontend (`static/`) e testes (`tests/`) |
| `acao-clara-vercel/` | Mesmo código empacotado para a Vercel: aplicação Flask (`app.py`), frontend em `public/`, e `static/` para o servidor local |
| `tools/sync-vercel.sh` | Copia `engine/`, `data/`, `server_core.py`, `static/`, `tests/` e `requirements.txt` de `acao-clara/` para `acao-clara-vercel/` |
| `app.py` (raiz) | Atalho: executa a aplicação de `acao-clara/` |

`server_core.py` concentra a lógica clássica (busca, histórico, regressão
logística e Ridge, ranking) e é compartilhada pelas duas versões; cada `app.py`
só expõe essa lógica no seu próprio servidor.

## Executar

Requer Python 3.10+ e acesso à internet:

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r acao-clara/requirements.txt
cd acao-clara && python app.py
```

Abra http://localhost:8000. Para trocar a porta, defina a variável `PORT`.
`python app.py` na raiz do repositório abre a mesma aplicação.

Sem internet, rode em modo demonstração com séries sintéticas determinísticas:

```bash
ACAO_CLARA_DEMO=1 python acao-clara/app.py
```

## O que o sistema faz

1. **Análise clássica** — regressão logística para a probabilidade de alta e
   regressão Ridge para o retorno esperado, com validação progressiva em quatro
   blocos cronológicos (acerto direcional, Brier, calibração, erro absoluto
   médio comparado à hipótese de preço constante).
2. **Motor quantitativo de oito núcleos** — macroeconomia, fundamentos e
   valuation (DCF + comps), interdependência entre mercados, eventos externos,
   fluxo financeiro, modelagem estatística, cenários e risco, combinados em três
   camadas (estimativas independentes → pesos dinâmicos → ajustes de risco) nos
   horizontes de 7, 30, 90, 180 e 365 dias.

A documentação completa do motor — núcleos, camadas, descoberta automática de
variáveis, qualidade dos dados, aprendizado com os próprios erros, endpoints e
variáveis de ambiente — está em [`acao-clara/README.md`](acao-clara/README.md).

## Testes

```bash
cd acao-clara && python -m unittest discover -s tests
```

## Fonte e limitações

Pesquisa e histórico diário obtidos dos endpoints públicos não oficiais do Yahoo
Finance; fundamentalos da brapi.dev; indicadores macro do BCB (SGS) e do FRED.
Disponibilidade, abrangência, qualidade e continuidade não são garantidas; sujeito
aos termos de cada provedor — verifique antes de publicar ou comercializar. O
histórico usa o campo `adjclose` fornecido pela fonte, normalmente ajustado para
dividendos e desdobramentos sem verificação independente. A fonte pode sofrer
atrasos ou revisões.

Este material é educacional e **não é recomendação de investimento**. Projeções
são estimativas estatísticas construídas com dados públicos, podem errar e não
garantem resultado futuro.
