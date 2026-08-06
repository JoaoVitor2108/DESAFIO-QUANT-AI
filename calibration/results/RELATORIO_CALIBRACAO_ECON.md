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
