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

Validação progressiva em quatro blocos cronológicos. Em cada bloco, o treino usa apenas amostras cujo rótulo já era conhecido antes do primeiro dia do teste, sem atualização dentro do bloco. Métricas fora da amostra: acerto direcional, Brier, comparação com a frequência de alta no treino e tabela de calibração. A previsão de retorno é avaliada pelo erro absoluto médio de retorno, comparado à hipótese de preço ajustado constante. Observações no tempo podem se sobrepor e não são independentes. O modelo final usa os rótulos conhecidos até o último fechamento disponível; testes não escolhem hiperparâmetros. Resultados passados não garantem resultados futuros. Notícias, balanços, custos e fatores externos não são considerados.

## Testes

```bash
python -m unittest discover -s tests
```
