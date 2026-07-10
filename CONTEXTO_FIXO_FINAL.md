# SISTEMA JEMPO — DESAFIO QUANT AI (ITAÚ ASSET MANAGEMENT)

## O que é
JEMPO é uma estratégia quantitativa multi-agente event-driven para ações
brasileiras (Ibovespa). Detecta oportunidades após eventos de notícias,
combinando análise fundamentalista via IA generativa com machine learning
quantitativo. Desenvolvido para o Desafio Quant AI 2026 do Itaú Asset
Management.

## Os 5 agentes
- JOURNAL (implementado, 70+ testes): provedor de dados puro. Coleta e
  organiza notícias, preços, fundamentos e macro. Fontes: Bloomberg CSV
  manual, GDELT (com backoff blindado), NewsAPI; yfinance; CVM; BCB SGS
  com fallback FRED.
- ECON (implementado, 36+ testes — commit 93316fa): analista
  fundamentalista qualitativo via Claude API. Devolve ScoreEcon.
  Método principal: `avaliar(ticker, data_limite)`.
  Calibração real pendente (depende de ANTHROPIC_API_KEY).
- MATH&ML (implementado e formalmente fechado, 35 testes): prevê
  retorno idiossincrático 5du à frente via GradientBoosting.
  Arquitetura em duas fases (pré-fetch + montagem em memória). Regra
  de platô com fallback para argmax. sample_weight=5x nos eventos
  como default. Run oficial de sensibilidade completo com relatório
  versionado (commits b7457e3 + 399145c). Mock estruturado permite
  treinar antes da API key do ECON. **Contrato de `prever_universo`
  expandido para 8 colunas — consumido pelo ORQUESTRADOR (ver seção
  específica abaixo).**
- ORQUESTRADOR (implementado, 56 testes): coordena os 3 agentes
  anteriores, toma decisão final e aplica gestão de risco. Agente
  central. Contrato público: `decidir(data, equity_hoje) → DecisaoDia`,
  `notificar_execucao`, `notificar_fechamento`, `status`. Não chama
  JOURNAL diretamente. Implementado via Claude Code em 7 etapas com
  approval gate por etapa.
- PROGRAM (não implementado): backtest com custos, Monte Carlo, e
  visualizações. Depende do ORQUESTRADOR pronto.

## Periodização do sistema
- Warmup: desde 2019 (features de momentum usam 252 dias úteis).
- Calibração ECON: 2020-2021 (teto otimista) + jul-dez/2025 (limpa).
- Treino MATH&ML: 2020-2023 (4 anos).
- Backtest OOS: 2024-2025.
- Walk-forward: retreina mensalmente.

## Regras do ORQUESTRADOR — TRAVADAS

**Todas as ambiguidades das regras originais foram resolvidas em
conversa arquitetural. Não abrir de novo sem motivo forte.**

### Regra de entrada — TOP-N DINÂMICO com filtro embutido
Pool de candidatas em cada dia D:
```
pool = df[(df.score_econ > 0.30) & (df.volume_relativo > 1.5)]
```
Não há threshold em `y_pred` — o ranking do MATH&ML é a ordenação.

Seleção com **N dinâmico = 3 − posições_abertas**, com limite setorial
embutido no loop:
1. Ordena pool por `rank` ascendente.
2. Percorre. Adiciona à lista "a comprar" se e somente se:
   - total (abertas + a comprar) ≤ 3
   - no mesmo setor (abertas + a comprar) ≤ 2
3. Para ao atingir 3 posições ou esgotar o pool.

Motivo do dinâmico: se top-3 é do mesmo setor, pega 1 e 2, pula 3,
desce até rank de outro setor. Maximiza uso do orçamento sem violar
limite setorial.

### Sizing — EQUAL WEIGHT 15%
Cada posição recebe 15% do capital corrente (`equity_hoje`). Não
pondera por força do sinal. Conviction weight e Kelly ficam como
flags experimentais fora de escopo.

### Limites de posição
- Máximo **3 posições simultâneas**.
- Máximo **2 posições no mesmo setor**.

### Regras de saída (primeiro critério atingido)
Prioridade: **stop > take > prazo > reversão**.
- Stop loss: preço ≤ 0.92 × preço_entrada.
- Take profit: preço ≥ 1.15 × preço_entrada.
- Prazo: 5 dias úteis (fecha na abertura do 6º).
- Reversão: `ECON.avaliar(ticker, data_limite=data).score_total < -0.30`.

Nota de honestidade metodológica (vai no relatório final): com alvo
de 5du, take-profit de +15% dispara raramente — saída dominante é
por prazo. Declarado abertamente.

### Circuit-breaker de drawdown
- Métrica: **trailing 21 dias úteis, peak-to-trough**, sobre a série
  de equity que o PROGRAM injeta em cada `decidir(data, equity_hoje)`.
- Se dd > 10%: pausa novas entradas por 5 dias úteis.
- Posições abertas continuam sob suas regras individuais de saída.
- Pausa não estende se novo drawdown durante pausa.
- Janela incompleta (t < 21): circuit-breaker inativo. Documentado.

### Lógica de timing D+1 vs D+2
- Decisão às 10h de D usa dados até fechamento de D-1.
- Execução padrão: abertura de D+1.
- Se `data_noticia_mais_recente > 17h05 de D-1`: execução em D+2.
- Regra vale por-ticker (candidatas diferentes podem ter datas
  diferentes no mesmo dia).
- Pula fins de semana ao somar dias úteis.

### Acoplamento com o PROGRAM
- **PROGRAM injeta `equity_hoje`** a cada `decidir(data, equity_hoje)`.
  ORQUESTRADOR não calcula P&L (respeita separação de
  responsabilidades) — só usa a equity para operar o circuit-breaker.
- **PROGRAM tem última palavra sobre fechamento.** No dia D, PROGRAM
  detecta stop/take intraday primeiro; se disparou, ignora
  `FechamentoOrdem` de prazo/reversão retornado pelo ORQUESTRADOR
  para o mesmo ticker. Isso preserva a prioridade
  `stop > take > prazo > reversão` mesmo com decisão distribuída
  entre dois agentes. Vai virar teste de integração quando PROGRAM
  existir.

### Custos no backtest (aplicados pelo PROGRAM, não pelo ORQUESTRADOR)
0.3% corretagem + 0.1% slippage por operação (0.8% round-trip).

## DECISÕES CONSOLIDADAS DO JOURNAL

### Anti-lookahead estrutural
- Timestamps TIMEZONE-AWARE em America/Sao_Paulo em todo lugar.
- Corte da B3: 17h05. Antes disso, D-1 é o último fechamento.
- ORQUESTRADOR decide às 10h e opera na abertura → toda decisão usa
  dados até D-1.
- `_assert_no_lookahead` em toda saída de dados (3 camadas de defesa).
- Fundamentos CVM: anti-lookahead via `data_recebimento_cvm`
  (DT_RECEB), não lag heurístico de 45 dias.
- IPCA: corte pela data de divulgação (~11 dias após ref), constante
  `_LAG_IPCA_DIAS`.

### JOURNAL como provedor de dados puro
- Coleta e entrega dados brutos. Não pontua, não pondera.
- `get_retornos_setor` agrega retornos setoriais (dado bruto). ECON faz
  a pontuação.

### Arquitetura modular das fontes
- Cada fonte externa em `agents/sources/` como classe própria:
  `CVMSource`, `GDELTSource`, `NewsAPISource`. Dataclass `Noticia`
  compartilhada em `noticia.py`.
- `JournalAgent` orquestra; parsing pesado nas classes específicas.

### Coleta de notícias em camadas
- Cascata em `get_noticias`: Bloomberg CSV (peso 1.0) > GDELT > NewsAPI.
- Whitelist rígida: bloomberg.com, reuters.com, valor.globo.com,
  valor.com.br, broadcast.com.br, estadao.com.br, infomoney.com.br.
- Pesos: Bloomberg 1.0, Reuters/Valor 0.95, Broadcast 0.90, Estadão
  0.85, InfoMoney 0.75.
- `TICKER_PARA_NOME` em `config.py` resolve ticker→nome antes da busca.
- Cache por fonte em pickle, TTL 24h.

### Deduplicação fuzzy entre fontes
- Similaridade de títulos via `difflib.SequenceMatcher` > 0.85 E
  diferença de publicação < 24h.
- `_DEDUP_SIM_MIN = 0.85`, `_DEDUP_HORAS_MAX = 24`.
- Processadas em ordem decrescente de peso; duplicatas de menor peso
  descartadas. Stdlib, sem dependência externa.

### GDELT blindado contra rate limit (Round 4)
- `GDELTRateLimitedError` e `GDELTUnavailableError` como exceções
  tipadas em `agents/sources/gdelt.py`.
- Backoff exponencial: 60s → 120s → 240s → 480s → 600s, 5 tentativas.
- `sleep_fn` injectável para testes determinísticos.
- Env var `GDELT_THROTTLE_SECONDS` (default 5). Rodadas oficiais usam 12s.
- Captura no `JournalAgent` com `gdelt_degradado_count` no
  `health_check()`. Se > 0 pós-run, magnitude é mensurável.

### Fonte de fundamentos: CVM
- `dados.cvm.gov.br/dados/CIA_ABERTA/DOC/`. Módulo em `cvm.py`.
- ITR trimestral + DFP anual como ZIPs, cache em `data/cvm/`.
- Encoding latin-1, sep ";", decimal ",", valores em milhares de R$.
- Sempre consolidado (`_con_`), nunca individual.
- Versões reapresentadas: filtra máxima por (CNPJ_CIA, DT_REFER).
- Bancos usam BPB (não BPA+BPP) — código ramifica por setor.
- yfinance é fallback APENAS para setor.
- Fluxo (receita, EBIT, lucro, D&A, EBITDA) em TTM:
  se DFP, TTM=anual; se ITR, TTM=ULTIMO_YTD + (DFP_ano_ant −
  PENULTIMO_YTD). Sem download extra.
- Estoque/balanço (ativo, patrimônio, dívida, caixa): point-in-time.
- Dicionário `periodicidade` no retorno explicita TTM vs point_in_time.
- P/L e P/VP best-effort: retorna None + aviso se ações em circulação
  ou preço falharem.
- Dataclass `Fundamentals` tem `data_recebimento_cvm` para
  rastreabilidade.

### Survivorship bias com membership por data
- `UNIVERSO_HISTORICO` cobre 2019-2025, com entrada/saida por ticker.
- Campos: setor, entrada, saida, confianca, fonte, cd_cvm, cnpj.
- `tickers_ativos(data_aware)` usada em TODOS os loops sobre universo.
- Casos emblemáticos: AMER3 (saída 12/jan/2023 — cai no treino, modelo
  aprende com quebra real); IRBR3 (saída após escândalo 2020).
- **JBSS3** saiu da B3 em 06/06/2025 (dupla listagem → BDR). Registrada
  com `saida=2025-06-06`.

### Fontes em camadas (resumo)
- Notícias: Bloomberg CSV (1.0) > GDELT (whitelist, backoff) > NewsAPI.
- Macro: BCB SGS primário, FRED fallback automático.
- Preços: yfinance com duas versões (ajustada e bruta).
- Fundamentos: CVM (primária), yfinance só para setor.

## DECISÕES CONSOLIDADAS DO ECON

### Função e contrato
- Analista fundamentalista qualitativo via Claude Haiku 4.5
  (`claude-haiku-4-5-20251001`). Modelo final pendente de
  `comparar_modelos` vs Sonnet 4.6.
- Recebe dossiê do JOURNAL (notícia + fundamentos CVM + macro +
  setoriais) e devolve `ScoreEcon`.
- **Método principal:** `avaliar(ticker, data_limite)`.
  `data_limite` deve ser passado explícito (kwarg) por consumidores
  (ORQUESTRADOR inclusive) — é defesa anti-lookahead.

### ScoreEcon — campos principais
- `score_total` [-1, +1]: IMPACTO DA NOTÍCIA no excesso ao Ibov em 5d.
  Opção A — NÃO combina saúde financeira / setor / macro.
- `comp_noticia`: base do `score_total`.
- `comp_saude_financeira`, `comp_setorial`, `comp_macro`: CONTEXTO
  considerado — NÃO somados ao total.
- `confianca` [0, 1], `tem_evento`, `n_noticias`.
- `noticias_hashes`: rastreabilidade.
- `data_noticia_mais_recente`: tz-aware `America/Sao_Paulo` ou NaT.
  Consumido pelo ORQUESTRADOR para decidir D+1 vs D+2 (regra 17h05
  da B3) e projetado na saída do `prever_universo` do MATH&ML.
- `justificativa`, `modelo`, `avisos`.

### Decisão "Opção A" sobre o score
- ECON pontua o MECANISMO da notícia; MATH&ML otimiza pesos.
- Reduz colinearidade EXPLÍCITA com features cruas que MATH&ML recebe
  do JOURNAL.
- Colinearidade IMPLÍCITA permanece (score_total é condicionado ao
  contexto fundamental) — diagnosticada em
  `diagnosticar_colinearidade` na calibração.

### Integração com MATH&ML
- `score_total` entra como feature principal.
- Saúde/setor/macro entram no MATH&ML como features CRUAS do JOURNAL,
  NÃO `comp_*` do ECON (evita duplicar sinal).
- `comp_*` disponíveis para interpretabilidade + features opcionais
  validadas por CV.

### Arquitetura técnica
- Tool use forçado + temperature=0. Reprodutibilidade vem do CACHE
  VERSIONADO (`_PROMPT_VERSION` na chave), não do temperature.
- Event-driven: sem notícia → `ScoreEcon` neutro sem chamar Claude.
- Degradação graciosa: nunca levanta exceção; devolve neutro + aviso;
  NÃO cacheia falha.
- Reusa `_DiskCache` e `_validate_aware` do JOURNAL.

### Anti-lookahead do LLM (3 defesas)
- Cutoffs do Haiku 4.5 (fim de mês como fronteira conservadora):
  - Reliable knowledge: fev/2025
  - Training data: jul/2025
- Janela genuinamente LIMPA = jul-dez/2025 (pós-training).
- Defesa 1 — IC segmentado: fronteira no TRAINING cutoff. Dois rótulos:
  `dentro_treino` (teto otimista) e `limpo` (confiável).
- Defesa 2 — Placebo com dois modos:
  - `swap`: notícia atribuída a par B do setor; contexto de B.
    ΔIC alto → sinal vem de memória sobre a empresa.
  - `identidade_pura`: anonimiza nome E oculta contexto fundamental.
- Defesa 3 — Auditoria regex de justificativas buscando linguagem
  ex-post.

### Calibração (estrutura pronta, execução pendente)
- Métrica primária: IC Spearman, meta > 0.15 com IC95 bootstrap não
  cruzando zero.
- Métrica secundária: IC com retorno ajustado por beta setorial.
- Baseline lexical + GAP para justificar custo do LLM.
- Degradação: >5% = ressalva; >15% = inválida.
- `contar_eventos_por_segmento` ANTES do IC; alerta se N_limpo < 30.
- Loop de prompt ≤ 10 iterações. Parar se IC_dentro sobe e IC_limpo
  não sobe (cola). Cada iteração bumpa `_PROMPT_VERSION`.

### Próximos passos do ECON (em ordem)
1. Smoke test com `ANTHROPIC_API_KEY` (~5-10 chamadas).
2. `comparar_modelos` Haiku 4.5 vs Sonnet 4.6 em 50-100 eventos limpos.
   Critério: gap > 0.05 → Sonnet; < 0.03 → Haiku.
3. Amostra de eventos (GDELT 2020-2021 + jul-dez/2025).
4. `calibrar` completo; ler relatório.
5. Loop de prompt engineering ≤ 10 iterações.
6. Congelar prompt + `RELATORIO_CALIBRACAO_ECON.md` final.

## DECISÕES CONSOLIDADAS DO MATH&ML

### Filosofia central: hipótese antes de padrão
- Cada feature mapeia 1-para-1 a hipótese testável da literatura, com
  sinal esperado explícito. Feature sem hipótese vira ruído.
- `importancia_features()` cruza ganho observado com sinal teórico;
  direção invertida → red flag reportado.

### Regra 1: target = retorno beta-ajustado 5du
- `y_i(t) = r_i_fwd − beta_i(t) × r_ibov_fwd`, sobre `Close_raw`
  (split-only) para ação e Ibov.
- Beta por `np.polyfit(r_ibov, r_i, deg=1)` sobre 252du terminando em t.
- Fallback `beta=1.0` se janela tiver < 200 observações; flag
  `beta_fallback=True` na linha.
- Refinamento opcional `beta_contra_setor` levanta
  `NotImplementedError` (spec futura).
- Motivação: retorno absoluto contamina com beta de mercado; usando
  idiossincrático, modelo aprende alfa.

### Regra 2: fundamentos por DT_RECEB
- MATH&ML consome `journal.get_fundamentals(ticker, data_limite)` que
  já corta por `data_recebimento_cvm <= data_limite`.
- Não reimplementar lag — DT_RECEB é estritamente superior.

### Regra 3: sinal = ranking cross-sectional
- `MathMLAgent` entrega **previsão contínua + ranking**; seleção final
  (top-N, limites setoriais) é do ORQUESTRADOR.
- Motivo: threshold absoluto tem problema de regime (bull dispara
  tudo, bear zera). Ranking é invariante a regime.
- Gate do ECON: pool = `score_econ > 0.3`; ranking DENTRO do pool.

### Regra 4: linhas de treino = todos os dias-ticker do universo ativo
- Não apenas dias com evento (overfit certo).
- Em dias sem evento, `score_econ=0`.
- **Sample weight de 5.0 nos dias de evento como default** (Round 7).
  Alinha o modelo com a hipótese econômica sem sacrificar IC líquido.
- IC reportado em DUAS agregações: total e subset `tem_evento=True`.

### Regra 5: períodos
- Warmup desde 2019-01-01.
- Treino 2020-2023. Backtest OOS 2024-2025.
- Walk-forward com janela expansiva. `freq='MS'` (mensal) default.

### Modelo: GradientBoosting raso de propósito
- `GradientBoostingRegressor(max_depth=3, learning_rate=0.05,
  subsample=0.8, random_state=42)`.
- `n_estimators` por regra de platô **com fallback para argmax**: se
  `n_platau < 0.3 × n_argmax`, usa `n_argmax`. Sem o fallback, platô
  colapsava para n=1 quando IC de CV era ruidoso.
- `cv_report` inclui `n_platau`, `n_argmax`, `n_escolhido`, `fonte`
  (`"platau"` ou `"argmax_fallback (...)"`).
- Justificativa metodológica: boosting corrige erro sequencialmente com
  regularização explícita (lr, subsample) + track record em painel
  financeiro. NÃO justificar como "RF é pior em sequência" — argumento
  fraco.

### Vetor de features (17 features + 3 flags de auditoria)

**Momentum e reversão:**
- `mom_12_1`: retorno 12m pulando último mês (Jegadeesh-Titman 1993). **+**
- `rev_1m`: retorno do último mês (Jegadeesh 1990; Lehmann 1990). **−**

**Fluxo de resultados (growth, não PEAD):**
- `dias_desde_resultado`: dias desde `data_recebimento_cvm`. Modula regime.
- `crescimento_lucro_yoy`: YoY do lucro TTM. **+** (Lakonishok-Shleifer-
  Vishny; growth, não SUE). **Renomeada de `surpresa_lucro`** — YoY de
  TTM não é PEAD clássico.
- `pead_window` foi **removida** (sem SUE de verdade).

**Qualidade/valor (Fama-French 1992; Novy-Marx 2013):**
- `pl`: P/L. **−** (baixo P/L supera)
- `pvp`: P/VP. **−** (book-to-market alto = HML)
- `roe`: **+** (qualidade)
- `margem`: **+**
- `divida_ebitda`: **−**

**Volume (Karpoff 1987):**
- `volume_relativo`: `Vol[t] / média(Vol[t-20:t])`. **+**
- **Duas semânticas convivem no código:** o modelo consome o valor
  **imputado** (mediana cross-sectional do dia). A projeção externa
  em `prever_universo` retorna o valor **cru** (NaN preservado se
  histórico insuficiente). Detalhes na seção "Contrato de saída do
  `prever_universo`" abaixo.

**Estado macro (regime):**
- `selic_nivel`, `selic_var_21d`, `cambio_var_21d`. Condicionantes;
  GBM usa em interações.
- **Não incluir retorno contemporâneo do Ibov** — alvo já é
  beta-ajustado.

**Notícia:**
- `score_econ`: `ScoreEcon.score_total`. **+**
- `econ_confianca`, `econ_n_noticias`.

**Flags de auditoria:**
- `beta_fallback`, `fundamental_imputado` (era `pl_imputado`),
  `econ_degradado`.

**Uso das componentes do ECON:** MATH&ML **não** consome
`comp_saude_financeira`, `comp_setorial`, `comp_macro` como features —
usa brutos do JOURNAL. Preserva "Opção A" do ECON e evita duplicação.

### Regras anti-lookahead (defesa em profundidade)
1. Toda feature em t usa só dados `<= t`.
2. Label usa `t+1..t+5` (forward — regras 3 e 4 obrigatórias).
3. Drop de label incompleto ANTES de `_montar_features`.
4. Purge + embargo López de Prado: `PurgedTimeSeriesSplit`, embargo 5du
   entre treino e teste, purge de amostras cujo `[t, t+5]` invade teste.
5. Imputação cross-sectional do dia; fallback global só do subset de
   treino.
6. Walk-forward respeita o mesmo embargo.
7. `_assert_no_lookahead` levanta `LookaheadError` em 4 fontes:
   preço, fundamento, macro, ECON.

### Arquitetura de pré-fetch (Round 5)
- **Bug medido:** padrão anterior chamava JOURNAL ~10k vezes por run.
  Backtest oficial (24 tickers × 6 anos) ≈ 130k chamadas → BCB SGS
  rate-limitava, run não terminava.
- **Solução em duas fases:**
  1. `_prefetch(tickers, data_inicio, data_fim)`: cada fonte chamada
     UMA vez cobrindo range inteiro. Popula `_DatasetCache` com
     precos, ibov, macro, fundamentos, setores.
  2. `_montar_features(ticker, t, cache)` e `_alvo(ticker, t, cache)`:
     lêem SÓ do cache — zero I/O na montagem.
- `_prefetch` tolera tickers sem dados no yfinance (ex.: ELET3.SA,
  JBSS3.SA pós-jun/2025 → 404). Skipa do cache; `construir_dataset`
  filtra em Fase 2.
- `get_macro`: uma chamada (série inteira).
- `get_retornos_setor`: REMOVIDO do cache (nenhuma feature consome).
- Fundamentos em dois modos:
  - `indexed` (produção, `DT_RECEB` populado): âncoras mensais +
    `_fund_em_t`.
  - `passthrough` (fixtures sem `DT_RECEB`): chamada por-t. Apenas em
    unit tests com `FakeJournal`; nunca em produção.
- Cache é fast-path puro: acessores caem no journal para
  ticker/data fora do cache.
- Teste guardião `test_prefetch_chamadas_unicas` no modo indexed.
- Cold-cache 24 tickers × 6 anos ~90 min; reruns ~5s via disk-cache.

### Ajustes do run oficial (Round 7)
Durante o run, três bugs foram corrigidos cirurgicamente:

**Fix 1 — Fallback do platô.** Com IC de CV ruidoso, `tol=0.5σ`
retornava `n_estimators=1` → predição constante. Solução: se
`n_platau < 0.3 × n_argmax`, usa `n_argmax`. `cv_report` registra
`fonte="argmax_fallback (...)"`.

**Fix 2 — sample_weight_eventos=5.0 default no MathMLConfig.** No mock,
não aumenta IC líquido, mas quadruplica ganho da feature-tese
`score_econ` (0.018 → 0.26 na importância). Alinha modelo com hipótese
econômica.

**Fix 3 — Snapshot do cv_report em sensibilidade_econ.py.** Script lia
`agent.cv_report` DEPOIS do walk_forward, capturando último fold em
vez do modelo principal. Snapshot antes corrigiu.

### Mock estruturado do ECON
- Substitui ECON real até API key + calibração.
- Sinal controlado: `α × z(y) + ruído`; `z(y)` é z-score cross-sectional
  dinâmico do dia.
- Quatro modos por IC alvo: ruído (0.00), fraco (0.10), meta (0.15),
  forte (0.20).
- Sem fallback tanh; distribuição unimodal (Round 2).
- Achado do Round 5: `α` na `amostra_calibracao` também fazia n+1 no
  JOURNAL. Solução: `z(y)` in-memory via `groupby("data")["y"]`. Tempo
  de calibração 9 min → 5s.

### Avaliação e métricas
- IC de Spearman total e no subset `tem_evento=True`.
- IC95 por block bootstrap por data (Round 2 — substituiu i.i.d. que
  subestimava incerteza).
- Três baselines competitivos:
  - B1: só `score_econ`
  - B2: só `mom_12_1`
  - B3: intercepto puro
- GAP = IC_modelo − max(B1, B2, B3).
- Checagem de sinal em `importancia_features()`.

### Contrato de saída do `prever_universo` (pós-expansão para ORQUESTRADOR)

Após a expansão feita antes do início da implementação do ORQUESTRADOR,
o `MathMLAgent.prever_universo(tickers, data_limite)` projeta **8
colunas** no DataFrame retornado. Este é o contrato público consumido
pelo ORQUESTRADOR — qualquer alteração aqui exige revisão do prompt do
ORQUESTRADOR.

| Coluna | Tipo | Semântica |
|---|---|---|
| `ticker` | str | Ex.: "PETR4.SA" |
| `y_pred` | float | Retorno idiossincrático 5du previsto pelo GBM |
| `score_econ` | float ∈ [-1, +1] | Score da notícia (via ECON ou mock) |
| `tem_evento` | bool | Se houve notícia relevante no dia |
| `rank` | int | Ranking cross-sectional (1 = melhor) |
| `volume_relativo` | float | `Vol[t]/média(Vol[t-20:t])` — **CRU**, pré-imputação |
| `data_noticia_mais_recente` | datetime tz-aware SP ou NaT | Do ScoreEcon; NaT se `tem_evento=False` |
| `setor` | str | Do `UNIVERSO_HISTORICO` via `journal.get_setor` |

**Notas essenciais:**

1. **`volume_relativo` é cru na projeção, imputado internamente para
   o modelo.** O GBM continua consumindo o valor imputado
   cross-sectional (mediana do dia) para features. A projeção externa
   preserva NaN. Motivação: `NaN > 1.5 = False` é o filtro correto no
   ORQUESTRADOR — ticker com histórico insuficiente **não deve**
   passar no filtro de volume. Imputar à mediana introduziria
   vazamento transversal em dias de evento setorial (o ticker novo
   passaria por causa do que os outros fizeram no dia). Volume cru é
   honesto e auditável.

2. **Duas fontes distintas de NaN em `volume_relativo`:**
   (a) `pos - janela_volume < 0` (histórico <20du);
   (b) `media_vol <= 0` na janela. Ambas caem no mesmo comportamento:
   NaN → não passa no filtro do ORQUESTRADOR.

3. **`data_noticia_mais_recente` é tz-aware `America/Sao_Paulo` ou
   `pd.NaT`.** Nunca naive. O ORQUESTRADOR usa esse campo para decidir
   D+1 vs D+2 (regra da 17h05 da B3). NaT → sem notícia → regra da
   17h não se aplica → execução em D+1.

4. **`setor` propaga `DadoIndisponivel` como erro de programação.** Se
   disparar, indica inconsistência entre `config.tickers_ativos(data)`
   e `UNIVERSO_HISTORICO` — bug de invariante, não caso de borda a
   tratar. Sem `try/except` no ORQUESTRADOR.

5. **Retorno-vazio tem o mesmo shape.** DataFrame vazio com as 8
   colunas nomeadas. Consumidor não precisa checar existência de
   colunas.

**Consequência arquitetural:** o ORQUESTRADOR **não chama JOURNAL
diretamente**. Toda informação necessária chega via `prever_universo`
do MATH&ML ou via `econ.avaliar` do ECON (esta última só para verificar
reversão de sinal em posições abertas).

### Resultados observados — Run oficial de sensibilidade
(24 tickers, 2020-2023 treino + 2024-2025 OOS, 4 modos do mock,
duração 90.8 min; relatório completo em
`calibration/results/RELATORIO_CALIBRACAO_MATHML.md`, commit 399145c).

| Modo | IC_alvo | n_platau→n_est | IC_evento | GAP vs B1 |
|---|---|---|---|---|
| ruído | 0.00 | 1→4 (fallback) | +0.029 | +0.029 |
| fraco | 0.10 | 1→4 (fallback) | +0.095 | +0.001 |
| **meta** | **0.15** | **1→4 (fallback)** | **+0.147** | **+0.0037** |
| forte | 0.20 | 3→17 (fallback) | +0.189 | −0.0004 |

**Sanidades — todas passaram:**
- Modo ruído: IC_evento ∈ [−0.05, +0.05] ✓
- Modo meta: `n_source=argmax_fallback` ✓
- Progressão monotônica ruído→forte ✓
- Modo meta: GAP ∈ [−0.05, +0.05] (empate honesto, lado positivo) ✓

**Importância no modo meta:**
- `score_econ` #1 com ganho 0.256 (era 0.008 no run com bug).
- `mom_12_1` #2 (0.236) — NÃO invertida.
- `rev_1m` #3.
- **Zero features com sinal invertido.**

**Nota metodológica registrada no §8 do relatório:** GAP=+0.0037 é
empate estatístico dentro do IC95 — mock é linear por construção, e B1
(ótimo linear) é matematicamente difícil de bater sem interações
não-lineares. Espera-se GBM abrir vantagem quando ECON real trouxer
sinal contextual não-linear.

### Determinismo e reprodutibilidade
- `PYTHONHASHSEED` ∈ {0, 1, 12345} — determinismo cross-process
  validado.
- `_stable_seed(ticker, t)` para RNG do mock.

### Suíte de testes (35 testes math_ml)
Cobertura: alvo beta-ajustado, anti-lookahead 4 fontes, drop de label
incompleto, purge+embargo, imputação cross-sectional, mock nos 4 modos,
walk-forward sem vazamento, `_prefetch` tolerante,
`test_prefetch_chamadas_unicas` (guardião n+1), regra de platô com
fallback, sample_weight default afeta fit. Testes novos da expansão do
contrato: `test_prever_universo_projeta_colunas_extras` (8 colunas
presentes, tipos corretos, tz-aware) e `test_volume_relativo_cru_preserva_nan`
(ticker com <20du sai NaN, ticker completo sai float finito).

## Stack técnica
Python, Claude API (Anthropic), scikit-learn, pandas, numpy, yfinance,
pyarrow (parquet), requests, matplotlib. Macro via BCB SGS (sem chave)
com FRED fallback. Fundamentos via CVM aberto. Dev no Mac (M1) via
Claude Code.

## Estrutura de arquivos
```
agents/
- journal.py
- econ.py             (EconAgent + ScoreEcon; método principal avaliar)
- math_ml.py          (MathMLAgent, MathMLConfig com
                       sample_weight_eventos=5.0 default,
                       _DatasetCache, _prefetch, make_econ_mock,
                       regra de platô com fallback,
                       prever_universo com 8 colunas)
- orchestrator.py     (implementado, 56 testes; contrato do §5 do
                       prompt do ORQUESTRADOR + privados
                       _atualizar_pausa, _selecionar_ordens,
                       _resolver_data_execucao, _verificar_fechamentos)
- program.py          (stub — depois do orchestrator)
- sources/
  - noticia.py        (Noticia + helpers de whitelist)
  - cvm.py            (CVMSource)
  - gdelt.py          (GDELTSource com backoff + exceções tipadas)
  - newsapi.py        (NewsAPISource)
calibration/
- econ_calibration.py
- results/
  - RELATORIO_CALIBRACAO_MATHML.md   (versionado, commit 399145c)
  - RELATORIO_CALIBRACAO_ECON.md     (gitignored, WIP)
backtest/
- engine.py           (não implementado — parte do PROGRAM)
- monte_carlo.py      (não implementado — parte do PROGRAM)
data/
- cache/              (pickle, TTL 24h)
- bloomberg/, cvm/raw/, cvm/processed/
tests/
- test_journal.py, test_cvm.py, test_gdelt.py, test_newsapi.py
- test_econ.py, test_econ_calibration.py
- test_math_ml.py     (35 testes)
- test_orchestrator.py (56 testes, incluindo 3 de integração)
scripts/
- smoke_test_mathml.py, smoke_test_v2.py
- debug_dataset.py
- diagnostico_gbm.py
- sensibilidade_econ.py   (run oficial dos 4 modos)
```

## Decisões pendentes

### ORQUESTRADOR — implementação concluída
Todas as 4 ambiguidades das regras originais foram resolvidas e
implementadas:
1. **Regra de entrada:** resolvida com top-N dinâmico
   (3 − posições_abertas) e filtro setorial embutido no loop. ✓
2. **Sizing:** resolvido com equal weight 15%. ✓
3. **Stop/take:** mantidos −8%/+15%/5du honestos (dominância de saída
   por prazo declarada abertamente no relatório). ✓
4. **Circuit-breaker:** implementado como trailing 21du peak-to-trough
   > 10%, pausando entradas por 5 dias úteis. ✓

Implementado em 7 etapas via Claude Code, com approval gate entre cada
uma. 56 testes determinísticos (incluindo 3 de integração), acima da
meta de 25. Anti-lookahead auditado explicitamente no §10 do prompt do
ORQUESTRADOR.

### Outros pendentes
- **PROGRAM (motor de backtest):** após ORQUESTRADOR estar pronto.
  Consome decisões do ORQUESTRADOR. Aplica custos reais (0.3% +
  0.1%), gera equity, Sharpe, drawdown, Monte Carlo 10k. **Contrato
  de acoplamento com ORQUESTRADOR já definido:** PROGRAM injeta
  `equity_hoje` em `decidir(data, equity_hoje)`; PROGRAM tem última
  palavra sobre fechamento (stop/take intraday) e ignora
  `FechamentoOrdem` de prazo/reversão em caso de conflito no mesmo
  dia. Teste de integração desse contrato é TODO do PROGRAM.

  **TODOs de integração PROGRAM+ORQUESTRADOR** (obrigatórios quando
  o PROGRAM for implementado):
  1. Teste de integração cobrindo o cenário em que stop/take
     intraday dispara no mesmo dia em que o ORQUESTRADOR retorna
     `FechamentoOrdem(motivo="prazo")` ou `"reversao"` para o mesmo
     ticker. Assert que PROGRAM fecha por stop/take e ignora o
     fechamento retornado pelo ORQUESTRADOR. Preserva a prioridade
     `stop > take > prazo > reversão` na fronteira dos dois agentes.
  2. Convenção de `equity_hoje`: PROGRAM deve documentar
     explicitamente se passa equity de fechamento de t-1
     (mark-to-market ao início do dia D) ou equity de fechamento de
     D (após executar ordens do dia). ORQUESTRADOR aceita ambos
     mas exige consistência ao longo do backtest — mudança de
     convenção no meio quebra o cálculo de drawdown.
- **Calibração real do ECON:** Haiku 4.5 vs Sonnet 4.6 empiricamente
  via `comparar_modelos` quando API key chegar. Substitui mock.
- **Cobertura completa do IBOV 2025** (TODOs em UNIVERSO_HISTORICO).
- **DFP anual vs ITRs no 4º trimestre** (resultado anual sai na DFP).

## Limitações conhecidas (documentadas, não bloqueantes)
- NewsAPI gratuito: 30 dias. Compensado pelas outras camadas.
- Bloomberg sem API; integração via CSV.
- `publishedAt` ≠ momento do evento (proxy).
- Ações em circulação na CVM têm gaps em reorganização.
- B3 sem API limpa para composição histórica do IBOV.
- GDELT com rate limit por IP — mitigado por backoff (Round 4).
  `gdelt_degradado_count` reporta magnitude.
- Paginação GDELT/NewsAPI não implementada (maxrecords=100 basta para
  janelas de 7 dias).
- Lookahead do LLM: MITIGADO (IC segmentado + placebo + auditoria),
  não eliminado. IC de 2020-2021 = teto otimista.
- **PEAD clássico não testado** — usamos `crescimento_lucro_yoy` como
  proxy de growth. PEAD legítimo exigiria SUE (surprise vs consenso).
- **yfinance frágil em runtime:** reporta "possibly delisted" para
  tickers ativos (ELET3 pós-privatização) e não serve tickers hoje
  delistados (JBSS3 pós-jun/2025). `_prefetch` tolera; documentado no
  relatório.
- **BCB SGS rate-limita por IP** silenciosamente após centenas de
  chamadas em rajada. Pré-fetch do MATH&ML elimina o gatilho.
- **MATH&ML empata com B1 no mock** — GAP=+0.0037 no modo meta é
  empate estatístico dentro do IC95. Esperado por construção (mock
  linear). Expectativa: GBM abre vantagem com ECON real.
- **Take-profit de +15% dispara raramente** com alvo de 5du. Saída
  dominante do ORQUESTRADOR é por prazo. Declarado abertamente no
  relatório final — não é bug, é consequência do alvo curto.