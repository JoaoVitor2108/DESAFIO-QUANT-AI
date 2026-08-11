# Calibração do ECON — baseline do prompt `2026-06-econA`

Etapa 1: mede o prompt ATUAL sem alterá-lo. Estabelece o número que as iterações da Etapa 2 precisam superar.

- Data/hora: 2026-08-06T16:20:23-03:00
- Fonte de notícias: Bloomberg-only (GDELT/NewsAPI neutralizados)

## 1. Parâmetros

- Modelo: `claude-haiku-4-5-20251001` (Sonnet descartado no gate de custo, ΔIC = -0.0298)
- Versão do prompt: `2026-06-econA` (NÃO modificada nesta etapa)
- Janela: 2024-01-01 → 2025-12-31
- N avaliado: 636 (de 636 eventos amostrados) | **N válido para métricas: 636**
- Métrica: IC de Spearman(score_total, retorno beta-ajustado 5du)
- Bootstrap: block-by-date, blocos de 5 dias úteis, 10000 iterações, seed=42
- Fronteira LIMPA: `data_noticia_mais_recente >= 2025-08-01 00:00:00-03:00` (pós training cutoff do Haiku 4.5, 2025-07-31)
- Hard cap: US$ 3.50 | **custo real: US$ 2.7937** (79.8% do cap)
- Deduplicação por conjunto de notícias: ATIVA

## 2. Amostra

- Candidatos (ticker-dia): 11632; sem notícia: 9578; duplicados descartados: 1289; sem target y: 129
- Eventos amostrados: 636 — dos quais LIMPOS (notícia ≥ 2025-08-01): 281
- Eventos AVALIADOS (base deste relatório): 636 — limpos: 281
- Tickers cobertos: 22
- Por mês: 2024-01=17, 2024-02=21, 2024-03=25, 2024-04=12, 2024-05=32, 2024-06=15, 2024-07=15, 2024-08=20, 2024-09=22, 2024-10=22, 2024-11=28, 2024-12=19, 2025-01=10, 2025-02=27, 2025-03=7, 2025-04=10, 2025-05=22, 2025-06=14, 2025-07=16, 2025-08=53, 2025-09=58, 2025-10=67, 2025-11=70, 2025-12=34
- Por ticker: PETR4.SA=86, ITUB4.SA=70, BBDC4.SA=57, VALE3.SA=56, BBSE3.SA=25, BPAC11.SA=23, VIVT3.SA=22, ASAI3.SA=22, TOTS3.SA=22, KLBN11.SA=22, LREN3.SA=22, CYRE3.SA=21, GGBR4.SA=21, EGIE3.SA=21, BBAS3.SA=20, WEGE3.SA=20, SUZB3.SA=20, RDOR3.SA=20, ABEV3.SA=19, PRIO3.SA=19, CMIN3.SA=19, MGLU3.SA=9
- Dispersão do score: média +0.1051, desvio 0.3801, min -0.75, max +0.75, zeros 9
- Dispersão do target y: média -0.0004, desvio 0.0361

## 3. Baseline — prompt `2026-06-econA`

| Métrica | IC de Spearman [IC95] |
|---|---|
| **IC completo** (2024-2025) | +0.0581 [-0.0291, +0.1442] — cruza zero (n=636, blocos=100) |
| **IC limpo** (notícia ≥ 2025-08-01) | +0.0137 [-0.1249, +0.1406] — cruza zero (n=281, blocos=23) |
| IC lexical B0 — completo | +0.0293 [-0.0579, +0.1111] — cruza zero (n=636, blocos=100) |
| IC lexical B0 — limpo | +0.0382 [-0.0873, +0.1461] — cruza zero (n=281, blocos=23) |

- **GAP (ECON − lexical), completo: +0.0288**
- **GAP (ECON − lexical), limpo: -0.0245**
- Cobertura do léxico B0 (eventos com score ≠ 0): 44.8%
- Veredito (IC completo): **PRECISA DE ITERAÇÃO SIGNIFICATIVA (< 0.10)**
- Veredito (IC limpo): **PRECISA DE ITERAÇÃO SIGNIFICATIVA (< 0.10)**
- Latência do LLM: mediana 4.01s | P95 5.66s
- Taxa de fallback (`tem_evento=False`): 0.0%
- **Taxa de degradação (não chegou ao LLM): 0.0%** (0/636) — limiares: ressalva > 5%, inválida > 15%
- Tokens médios: 2723 in / 334 out | custo médio US$ 0.004393/chamada

## 4. Diagnóstico — onde o ECON está errando


### IC por ticker (só grupos com n ≥ 10, pior primeiro)

| Ticker | n | scores zerados | IC | observação |
|---|---|---|---|---|
| RDOR3.SA | 20 | 0 | -0.4022 |  |
| LREN3.SA | 22 | 1 | -0.3034 |  |
| VALE3.SA | 56 | 0 | -0.0976 |  |
| GGBR4.SA | 21 | 0 | -0.0662 |  |
| CMIN3.SA | 19 | 0 | -0.0478 |  |
| PRIO3.SA | 19 | 1 | -0.0265 |  |
| BBSE3.SA | 25 | 0 | -0.0202 |  |
| VIVT3.SA | 22 | 0 | +0.0090 |  |
| TOTS3.SA | 22 | 0 | +0.0230 |  |
| WEGE3.SA | 20 | 1 | +0.0289 |  |
| ASAI3.SA | 22 | 0 | +0.0405 |  |
| PETR4.SA | 86 | 0 | +0.0696 |  |
| BBDC4.SA | 57 | 2 | +0.0837 |  |
| KLBN11.SA | 22 | 0 | +0.0878 |  |
| ABEV3.SA | 19 | 1 | +0.1414 |  |
| ITUB4.SA | 70 | 1 | +0.2236 |  |
| BPAC11.SA | 23 | 2 | +0.2495 |  |
| CYRE3.SA | 21 | 0 | +0.2786 |  |
| EGIE3.SA | 21 | 0 | +0.3334 |  |
| SUZB3.SA | 20 | 0 | +0.3428 |  |
| BBAS3.SA | 20 | 0 | +0.3990 |  |

### IC por mês (só grupos com n ≥ 10, pior primeiro)

| Mês | n | scores zerados | IC | observação |
|---|---|---|---|---|
| 2025-02 | 27 | 0 | -0.2889 |  |
| 2025-06 | 14 | 0 | -0.1905 |  |
| 2025-04 | 10 | 0 | -0.1835 |  |
| 2024-03 | 25 | 0 | -0.1628 |  |
| 2025-12 | 34 | 0 | -0.1119 |  |
| 2024-02 | 21 | 0 | -0.0774 |  |
| 2025-08 | 53 | 3 | -0.0666 |  |
| 2024-10 | 22 | 0 | -0.0352 |  |
| 2025-11 | 70 | 1 | -0.0350 |  |
| 2024-05 | 32 | 1 | -0.0288 |  |
| 2024-06 | 15 | 1 | -0.0149 |  |
| 2024-04 | 12 | 0 | +0.0146 |  |
| 2024-11 | 28 | 0 | +0.0365 |  |
| 2024-09 | 22 | 0 | +0.0811 |  |
| 2025-10 | 67 | 0 | +0.0821 |  |
| 2024-12 | 19 | 2 | +0.1007 |  |
| 2024-08 | 20 | 0 | +0.1403 |  |
| 2025-09 | 58 | 1 | +0.1919 |  |
| 2024-01 | 17 | 0 | +0.2081 |  |
| 2024-07 | 15 | 0 | +0.2410 |  |
| 2025-07 | 16 | 0 | +0.3194 |  |
| 2025-01 | 10 | 0 | +0.3708 |  |
| 2025-05 | 22 | 0 | +0.5669 |  |

### 10 casos de maior discordância de rank

Score no topo com retorno no fundo (ou o inverso) — a mesma moeda do Spearman. `justificativa` truncada.

| ticker | data | score | y | justificativa |
|---|---|---|---|---|
| LREN3.SA | 2024-11-07 | +0.75 | -0.0612 | Renner divulgou 3Q24 com lucro líquido 48% acima do consenso (R$255,3M vs R$214,6M) e comparable sales de +11, |
| PRIO3.SA | 2024-06-05 | -0.65 | +0.0656 | Produção de petróleo caiu 4,4% m/m (88,7 mil b/d vs 92,8 mil) e vendas recuaram 15% m/m, sinalizando queda mat |
| CMIN3.SA | 2024-05-09 | -0.65 | +0.0574 | Resultado 1T24 da CSN Mineração mostra deterioração significativa: receita -32% a/a, EBITDA ajustado -44% a/a  |
| LREN3.SA | 2025-02-27 | -0.65 | +0.0563 | Renner divulgou 4T 2024 com lucro e vendas abaixo de estimativas, gerando queda de 11% intradiária. Itau BBA r |
| PETR4.SA | 2024-09-26 | -0.65 | +0.0548 | A notícia de queda acentuada do petróleo (Brent -3,7% para US$70,72, WTI -3,6% para US$67,16) com perspectiva  |
| WEGE3.SA | 2024-07-31 | +0.75 | -0.0415 | WEG superou estimativas em lucro líquido (+5.4% vs. consenso), EBITDA (+16% vs. consenso) e ROIC (+37.4% vs. + |
| CMIN3.SA | 2024-11-06 | +0.65 | -0.1229 | Itochu aumenta participação em CSN Mineração de 7,15% para ~18% (adquirindo 10,74%) por ~117b yen (~R$ 4,5bi), |
| LREN3.SA | 2025-02-25 | -0.65 | +0.0533 | Renner divulgou 4T com lucro líquido 21% abaixo do consenso (R$487M vs R$614M) e margem Ebitda contraída 310 b |
| LREN3.SA | 2025-08-07 | +0.65 | -0.1100 | Renner entregou resultados sólidos com lucro líquido +28% a/a (acima do consenso), receita +18% a/a e margem E |
| TOTS3.SA | 2024-11-06 | +0.65 | -0.1000 | Totvs superou estimativas de Ebitda ajustado em 3T24 (+17% a/a vs. consenso de R$321,1 mi), demonstrando cresc |

## 5. Próximos passos (Etapa 2 — iteração de prompt)

1. **GAP baixo contra o léxico**: o ECON ainda não está fazendo muito além de dicionário de sentimento. Reforçar o pedido de MECANISMO (efeito em caixa/margem/múltiplo) e penalizar explicitamente a leitura de tom da manchete.
2. **Tickers com IC negativo** (RDOR3.SA, LREN3.SA, VALE3.SA, GGBR4.SA): inspecionar as justificativas desses nomes — pode ser setor em que o prompt inverte o sinal (ex.: notícia de dividendo vs. de capex).
3. **Horizonte explícito**: reforçar que a janela é de 5 dias úteis e que notícia já precificada no dia deve pontuar perto de zero.
4. **Protocolo da Etapa 2**: uma mudança por iteração, bump de `_PROMPT_VERSION` (invalida o cache), re-rodar esta mesma rotina e comparar contra este baseline. Parar em IC > 0.15, 10 iterações ou hard cap de US$15.

## Limitações desta rodada

- **Contaminação parcial em 2024-2025 até jul/2025**: está dentro do training cutoff do Haiku 4.5. Por isso o IC limpo é reportado à parte — é o número honesto; o completo é teto otimista com mais poder.
- **Concentração por ticker**: PETR4/ITUB4/BBDC4/VALE3 dominam o dataset Bloomberg. O IC agregado pesa mais esses nomes.
- **Deduplicação muda a unidade de observação** para (ticker, configuração de notícias), não (ticker, dia). Reduz custo e a dependência serial, mas não a elimina — daí o block bootstrap.
- **485 avaliações sem justificativa persistida**: o checkpoint original não gravava esse campo (schema corrigido depois). Onde a tabela de piores casos mostra justificativa, ela foi RE-BUSCADA com o mesmo prompt e modelo.
- Target usa Close AJUSTADO (`_retorno_excesso_5d`), enquanto o MATH&ML usa Close_raw; diferença esperada é pequena.

CSV completo: `calibration/results/calibracao_baseline_completo.csv`

<!-- etapa2:2026-08-econB -->

## Iteração — prompt `2026-08-econB`

### Mudança no prompt

Eixo 1 — SURPRESA vs CONSENSO. O prompt v1 pedia o MECANISMO econômico da notícia, o que o LLM lia como "notícia boa → score positivo". A v2 instrui que a nota mede o DELTA entre o anunciado e o já esperado: surpresa para cima → positivo, confirmação de expectativa → ~0 mesmo com número excelente, surpresa para baixo → negativo; |score| > 0,5 reservado a surpresa genuína. Como só ~17% das notícias trazem consenso explícito no corpo, a inferência foi hierárquica: (1) consenso declarado no texto; (2) na ausência, natureza do evento (rotineiro/antecipável → zero; inesperado → magnitude), com proibição explícita de inferir a expectativa a partir de memória do desfecho. Tool schema de score_total/componente_noticia atualizado no mesmo sentido. Contrato da v1 preservado (Opção A, contexto não somado, múltiplas notícias, anti-lookahead).

### Resultados

- IC completo: +0.0758 [-0.0141, +0.1614]
- IC limpo: +0.0015 [-0.1514, +0.1273] (n=281)
- Lexical B0 limpo: +0.0382 (referência fixa)
- GAP limpo: -0.0367
- ΔIC limpo vs baseline: -0.0122
- ΔIC limpo vs iteração anterior: n/d
- Custo desta iteração: US$ 3.1841
- Custo acumulado Etapa 2: US$ 3.1841
- Taxa de degradação: 0.0% | latência mediana 4.14s
- Dispersão do score: desvio 0.3296, zeros 14

### Diagnóstico

- **LREN3 tracker**: 4 de 10 piores casos — o eixo 1 NÃO resolveu o caso emblemático.

**Piores casos desta versão:**

| ticker | data | score | y |
|---|---|---|---|
| CMIN3.SA | 2024-11-06 | +0.65 | -0.1229 |
| LREN3.SA | 2024-05-08 | +0.65 | -0.0901 |
| MGLU3.SA | 2024-03-18 | +0.65 | -0.0901 |
| MGLU3.SA | 2024-06-24 | +0.65 | -0.0778 |
| LREN3.SA | 2024-11-07 | +0.65 | -0.0612 |
| LREN3.SA | 2025-02-27 | -0.65 | +0.0563 |
| PETR4.SA | 2024-09-26 | -0.65 | +0.0548 |
| LREN3.SA | 2025-02-25 | -0.65 | +0.0533 |
| ASAI3.SA | 2024-10-17 | -0.55 | +0.0684 |
| GGBR4.SA | 2025-10-01 | -0.55 | +0.0512 |

- Piores tickers: LREN3.SA (-0.274), VALE3.SA (-0.214), RDOR3.SA (-0.193), BBSE3.SA (-0.077)
- Melhores tickers: EGIE3.SA (+0.385), CYRE3.SA (+0.387), BBAS3.SA (+0.446)

<!-- etapa2:comparativo -->

## Comparativo entre versões

| Versão | n | IC completo | IC limpo | GAP limpo | Custo iter | Custo acum |
|---|---|---|---|---|---|---|
| 2026-06-econA | 636 | +0.0581 | +0.0137 | -0.0245 | US$ 2.7937 | US$ 2.7937 |
| 2026-08-econB | 636 | +0.0758 | +0.0015 | -0.0367 | US$ 3.1841 | US$ 5.9778 |

<!-- etapa2:conclusao -->
## Conclusão da Etapa 2 — encerrada após 1 iteração

**Versão final adotada: `2026-06-econA` (o baseline).** A v2 foi revertida no
código; permanece documentada aqui, com o texto integral no anexo, para que o
experimento seja reproduzível.

### Por que parou

A regra de parada disparou já na iteração 1: o IC limpo caiu de +0,0137 para
+0,0015 (ΔIC = −0,0122). Mas o motivo de encerrar a etapa inteira, e não apenas
reverter e tentar o próximo eixo, é o que o corte por janela revelou:

| Janela | n | IC v1 | IC v2 | Δ |
|---|---|---|---|---|
| Contaminada (< 2025-08-01) | 355 | +0,0764 | +0,1032 | +0,0268 |
| **Limpa (≥ 2025-08-01)** | 281 | +0,0137 | +0,0015 | **−0,0122** |

O prompt melhorou o IC **apenas onde o modelo pode ter visto o desfecho no
treino**. Essa é a assinatura de memória que a Defesa 1 existe para detectar — e
o critério de overfit que `calibrar()` já documentava como motivo de parada:
"IC `dentro_treino` sobe mas IC `limpo` NÃO sobe (ou cai) entre iterações".

A explicação é direta: só **17%** das notícias trazem consenso explícito no corpo
Bloomberg. Nos outros 83%, perguntar "isso surpreendeu?" sem dar o consenso
convida o modelo a responder de memória. Onde ele tem memória, acerta mais; onde
não tem, a instrução extra só adiciona ruído.

### A instrução foi obedecida — o problema não é o prompt

| | v1 | v2 |
|---|---|---|
| Score médio | +0,1051 | +0,0385 |
| Desvio | 0,3801 | 0,3296 |
| `\|score\| > 0,5` | 27,0% | 17,6% |
| `\|score\| ≤ 0,1` | 5,8% | **16,5%** |
| Eventos com score alterado | — | 54,1% |

O modelo de fato passou a puxar notícia rotineira para perto de zero, que era o
objetivo. O comportamento mudou; o poder preditivo na janela limpa, não.

### LREN3 tracker

Continua **4 de 10** nos piores casos, e o IC isolado do ticker mal se moveu
(−0,3034 → −0,2744). O eixo 1 não resolveu o caso que o motivou.

### A limitação que encerra a etapa

Nenhuma dessas diferenças é estatisticamente distinguível. O IC95 do IC limpo tem
largura ≈ 0,28 ([−0,1514, +0,1273] na v2) enquanto os efeitos perseguidos são de
0,01–0,03 — uma ordem de grandeza abaixo do erro amostral. Com n=281 e 23 blocos,
o block bootstrap está corretamente indicando que **a amostra não tem poder para
resolver essa pergunta**.

Continuar iterando a US$3,18 por rodada perseguiria diferenças dentro do ruído.
As alternativas reais não são de prompt:

1. **Mais poder amostral** — desligar a deduplicação levaria a janela limpa de
   281 para ~683 eventos, estreitando o IC95. Não corrige contaminação e custa
   ~US$9 por rodada.
2. **Consenso como dado, não como inferência** — o eixo 1 só é testável de
   verdade com dados ERN (estimativa × realizado) no dossiê. O arquivo
   `earnings_bloomberg.xlsx` não existe na máquina; sem ele, 83% dos eventos
   ficam sem âncora.
3. **Aceitar o resultado** — o ECON, como sinal isolado, tem IC indistinguível de
   zero na janela limpa. Isso é um achado legítimo e reportável: o MATH&ML
   combina o score com outras features, e o valor do ECON no sistema completo é
   medido lá, não aqui.

### Custos

- Etapa 1 (baseline): US$ 2,7937
- Etapa 2 (1 iteração): US$ 3,1841
- **Total da calibração: US$ 5,9778** — de um cap de US$ 15 para a Etapa 2, do
  qual sobraram US$ 11,82 não gastos.

### Anexo — prompt `2026-08-econB` (revertido, preservado para reprodução)

```text
Você é um analista fundamentalista sênior de ações brasileiras (buy-side). O 'score_total' é sua nota PRINCIPAL: o impacto da(s) notícia(s) no retorno EM EXCESSO ao Ibovespa nos próximos 5 dias úteis (-1 muito negativo, 0 neutro, +1 muito positivo); ele deve refletir essencialmente o 'componente_noticia'. SURPRESA, NÃO DIREÇÃO: o preço já embute a expectativa. O mercado brasileiro incorpora o consenso semanas antes do anúncio, então sua nota mede o DELTA entre o que foi anunciado e o que já era esperado — NÃO se a notícia é 'boa' ou 'ruim' em termos absolutos. Três casos: (a) SURPREENDE PARA CIMA (melhor que o esperado) → score positivo; (b) CONFIRMA a expectativa → score próximo de ZERO, mesmo que o número em si seja excelente; (c) SURPREENDE PARA BAIXO (pior que o esperado) → score negativo. Um resultado trimestral forte e amplamente antecipado merece ~0, não +0,6. Reserve |score| > 0,5 para surpresa GENUÍNA: magnitude muito fora do consenso, evento inesperado ou mudança de regime. COMO INFERIR O QUE ERA ESPERADO — nesta ordem: (1) use o consenso declarado NO PRÓPRIO TEXTO ('estimate', 'consensus', 'estimativa', 'acima/abaixo do esperado', projeções de analistas citadas); (2) se o texto não trouxer consenso, julgue pela NATUREZA do evento: rotineiro e antecipável (resultado dentro do calendário, dividendo recorrente, guidance reiterado, follow-up de fato já divulgado) puxa para zero; genuinamente inesperado (M&A, troca de comando, decisão regulatória, fraude, acidente, revisão abrupta de guidance) justifica magnitude. JAMAIS infira a expectativa a partir do que você lembra que aconteceu depois — isso é lookahead e invalida a avaliação. Na dúvida sobre o consenso, fique perto de zero e REDUZA a 'confianca'. Avalie o MECANISMO econômico (efeito em caixa, margem, posição competitiva ou múltiplo), não o tom do texto. Saúde financeira (fundamentos TTM), momento setorial e cenário macro são o CONTEXTO que calibra a leitura (a mesma surpresa pesa mais numa empresa frágil) — você os reporta nos campos próprios, mas eles NÃO são parcelas somadas ao score_total. Desconte ruído sem efeito fundamental (ex.: política genérica). MÚLTIPLAS NOTÍCIAS: pondere pelo impacto fundamental e pela confiabilidade da fonte; notícias contraditórias entre si devem REDUZIR a 'confianca'. ANTI-LOOKAHEAD: raciocine APENAS com os dados fornecidos, como se a data de hoje fosse a data_limite informada; jamais use conhecimento de fatos posteriores a essa data. Responda EXCLUSIVAMENTE chamando a ferramenta registrar_avaliacao.
```
