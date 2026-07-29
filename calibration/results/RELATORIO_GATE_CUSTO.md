# Gate de custo — Haiku 4.5 vs Sonnet 4.6

- Data/hora de execução: 2026-07-29T03:20:38-03:00
- Fonte de notícias: Bloomberg-only

## Parâmetros do experimento

- N alvo: TODOS os eventos limpos | N mínimo: 40 | N efetivo: 318
- Janela: 2025-08-01 → 2025-12-31
- Fronteira LIMPA (exata): `data_noticia_mais_recente >= 2025-08-01 00:00:00-03:00` (estritamente pós training cutoff do Haiku 4.5, 2025-07-31). Jul/2025 EXCLUÍDO por estar no treino.
- Modelos: `claude-haiku-4-5-20251001` vs `claude-sonnet-4-6`
- Métrica: IC Spearman(score, retorno beta-ajustado 5du)
- Bootstrap: block-by-date, blocos de 5 dias úteis, 10000 iterações, seed=42
- Seed da amostra: 42

## Amostra

- Sondagens de notícia: 2100 (de 2100 candidatos); sem notícia: 1780; notícia pré-cutoff (< 2025-08-01, descartadas): 0; sem target y: 2
- Segmento (cutoff Haiku 2025-07-31): limpo=318
- Distribuição por mês: 2025-08=55, 2025-09=68, 2025-10=78, 2025-11=69, 2025-12=48
- Distribuição por ticker: PETR4.SA=90, ITUB4.SA=82, VALE3.SA=75, BBDC4.SA=71
- ⚠️ Tickers com > 5 eventos (concentração): PETR4.SA=90, ITUB4.SA=82, VALE3.SA=75, BBDC4.SA=71

## Resultado — Haiku 4.5 (`claude-haiku-4-5-20251001`)

- IC Spearman: +0.0468 [-0.1535, +0.2286]  (n_blocos=22)
- Custo total: US$ 1.4967 | médio/chamada: US$ 0.004707
- Latência LLM: mediana 4.02s | P90 5.12s | P95 5.71s
- Taxa de fallback: 0.6%

## Resultado — Sonnet 4.6 (`claude-sonnet-4-6`)

- IC Spearman: +0.0169 [-0.1937, +0.2111]  (n_blocos=22)
- Custo total: US$ 4.7650 | médio/chamada: US$ 0.014984
- Latência LLM: mediana 9.66s | P90 12.09s | P95 13.01s
- Taxa de fallback: 0.3%

## ΔIC (Sonnet − Haiku)

- ΔIC = -0.0298 [-0.1267, +0.0770] (bootstrap pareado, mesmos blocos)

## Decisão automatizada

- Critério: ΔIC > 0.05 → Sonnet; < 0.03 → Haiku; zona cinza → Haiku (custo-benefício).
- **DECISÃO: HAIKU**

## Custo real gasto

- **Custo real total: US$ 6.2617**
- Percentual do hard cap (US$ 8): 78.3%
- Crédito restante estimado: US$ 3.74 (de US$ 10 iniciais)

## Latência e implicação prática

Latência mediana isolada do LLM (excluindo coleta de notícias):
- Haiku 4.5: 4.02s por chamada (medido)
- Sonnet 4.6: 9.66s por chamada (medido)
- ΔLatência: 5.64s (Sonnet 2.4x mais lento)

Implicação pro backtest oficial estimado:
- Universo alvo: ~2500 chamadas (24 tickers × ~100 eventos)
- Haiku wall-clock: ~2.8h
- Sonnet wall-clock: ~6.7h

Além do resultado no IC (ΔIC = -0.0298), essa diferença de latência
reforça a escolha de Haiku para o backtest oficial: dobra o tempo de
iteração se problemas forem descobertos e o backtest precisar rerodar.

## Limitações

- **Amostra restrita a ago-dez/2025** (janela limpa do Haiku 4.5, estritamente pós training cutoff 2025-07-31; fronteira exata `data_noticia_mais_recente >= 2025-08-01 00:00:00-03:00`). N pequeno e uma única janela → não generaliza para outros regimes.
- **Cutoff do Sonnet 4.6 pode DIFERIR do Haiku 4.5.** Se for mais recente, ago-dez/2025 pode estar (parcialmente) no treino do Sonnet → risco de contaminação diferencial (Sonnet 'lembrar' de eventos que o Haiku não viu). Não há como corrigir dentro do gate.
- **Divergência metodológica com `econ_calibration.py::comparar_modelos()`**: lá o bootstrap é i.i.d. (reamostra linhas); aqui é block-by-date (blocos de 5 dias úteis). Motivo: retornos por data são correlacionados, então i.i.d. subestima o IC95. O ΔIC deste gate usa bootstrap PAREADO (mesmos blocos p/ os dois modelos).
- Sem iteração de prompt, sem placebo, sem auditoria de justificativas (fora do escopo deste gate — é decisão única).
- Target usa Close AJUSTADO (via `_retorno_excesso_5d`), enquanto o MATH&ML usa Close_raw; diferença de retorno esperada é pequena.
- Eventos se sobrepõem: `get_noticias` usa lookback de 7 dias, então a mesma notícia é pontuada em ticker-dias vizinhos (não são observações independentes; o block bootstrap por data mitiga).

## Anexo — dados brutos (amostra)

**Primeiros 10 eventos:**

| evento_id | ticker | data | y_realizado | score_haiku | score_sonnet |
|---|---|---|---|---|---|
| 0 | BBDC4.SA | 2025-08-01 | -0.0038 | +0.0500 | +0.0500 |
| 1 | ITUB4.SA | 2025-08-01 | +0.0301 | +0.0000 | +0.0000 |
| 2 | BBDC4.SA | 2025-08-04 | -0.0081 | +0.0500 | +0.0500 |
| 3 | ITUB4.SA | 2025-08-04 | +0.0325 | +0.3500 | +0.3500 |
| 4 | BBDC4.SA | 2025-08-05 | -0.0039 | +0.1500 | +0.0500 |
| 5 | ITUB4.SA | 2025-08-05 | +0.0234 | +0.1500 | +0.3500 |
| 6 | BBDC4.SA | 2025-08-06 | +0.0118 | +0.1500 | +0.0500 |
| 7 | ITUB4.SA | 2025-08-06 | +0.0236 | +0.6500 | +0.5500 |
| 8 | BBDC4.SA | 2025-08-07 | +0.0153 | +0.0500 | +0.0500 |
| 9 | ITUB4.SA | 2025-08-07 | +0.0243 | +0.6500 | +0.5500 |

**Últimos 10 eventos:**

| evento_id | ticker | data | y_realizado | score_haiku | score_sonnet |
|---|---|---|---|---|---|
| 308 | ITUB4.SA | 2025-12-18 | +0.0071 | -0.1500 | +0.1500 |
| 309 | PETR4.SA | 2025-12-18 | +0.0131 | +0.2500 | +0.3500 |
| 310 | BBDC4.SA | 2025-12-19 | -0.0180 | -0.1500 | -0.1500 |
| 311 | PETR4.SA | 2025-12-19 | +0.0121 | +0.2500 | +0.3500 |
| 312 | BBDC4.SA | 2025-12-22 | -0.0002 | -0.2500 | +0.1500 |
| 313 | PETR4.SA | 2025-12-22 | +0.0069 | +0.3500 | +0.2500 |
| 314 | PETR4.SA | 2025-12-26 | -0.0383 | +0.1500 | +0.0500 |
| 315 | BBDC4.SA | 2025-12-29 | +0.0295 | +0.1500 | -0.0500 |
| 316 | PETR4.SA | 2025-12-29 | -0.0362 | +0.1500 | +0.0500 |
| 317 | BBDC4.SA | 2025-12-30 | +0.0043 | +0.1500 | -0.1000 |

CSV completo: `calibration/results/gate_custo_resultado_completo.csv`
