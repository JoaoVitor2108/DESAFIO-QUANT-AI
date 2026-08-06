# SISTEMA JEMPO — DESAFIO QUANT AI (ITAÚ ASSET MANAGEMENT)

## O que é
JEMPO é uma estratégia quantitativa multi-agente event-driven para ações
brasileiras (Ibovespa). Detecta oportunidades após eventos de notícias,
combinando análise fundamentalista via IA generativa com machine learning
quantitativo. Desenvolvido para o Desafio Quant AI 2026 do Itaú Asset
Management.

## Os 5 agentes
- **JOURNAL** (implementado, 76+ testes): provedor de dados puro. Coleta e
  organiza notícias, preços, fundamentos e macro. Fontes: **Bloomberg CSV
  (primária, integrada via BloombergCSVSource)**, GDELT (suplementar,
  com backoff blindado, IP-flaky), NewsAPI (fallback); yfinance; CVM;
  BCB SGS com fallback FRED.
- **ECON** (implementado, 36+ testes; validado com API real em 17/07/2026):
  analista fundamentalista qualitativo via Claude Haiku 4.5.
  Devolve ScoreEcon. Método principal: `avaliar(ticker, data_limite)`.
  **Modelo decidido empiricamente via gate de custo (29/07/2026):
  Haiku 4.5 dominou Sonnet 4.6** (ΔIC = −0.0298, 3x mais barato,
  2.4x mais rápido). Custo real ~US$0.0047/chamada.
  Calibração real de prompt pendente (próxima ação).
- **MATH&ML** (implementado e formalmente fechado, 35 testes): prevê
  retorno idiossincrático 5du à frente via GradientBoosting.
  Arquitetura em duas fases (pré-fetch + montagem em memória). Regra
  de platô com fallback para argmax. sample_weight=5x nos eventos
  como default. Run oficial de sensibilidade completo com relatório
  versionado (commits b7457e3 + 399145c). Mock estruturado permite
  treinar antes da API key do ECON. Contrato de `prever_universo`
  expandido para 8 colunas — consumido pelo ORQUESTRADOR.
- **ORQUESTRADOR** (implementado, 56 testes): coordena os 3 agentes
  anteriores, toma decisão final e aplica gestão de risco. Agente
  central. Contrato público: `decidir(data, equity_hoje) → DecisaoDia`,
  `notificar_execucao`, `notificar_fechamento`, `status`. Não chama
  JOURNAL diretamente. Implementado via Claude Code em 7 etapas com
  approval gate por etapa.
- **PROGRAM** (Etapas 1, 2, 3 fechadas — 98 testes): motor de backtest
  event-driven com Monte Carlo e métricas de performance. Etapas 4-5
  pendentes (visualizações/notebook, integração ECON real).

## Estado da suíte de testes
- **Total: 396 testes verdes** (após parser Bloomberg Etapa 2 +
  gate de custo + universo atualizado + dataset unificado).
- Zero regressão em nenhuma etapa desde o início.

## Periodização do sistema
- Warmup: desde 2019 (features de momentum usam 252 dias úteis).
- **Calibração ECON: ago-dez/2025** (janela limpa pós-training cutoff
  Haiku em 2025-07-31; jul/2025 removido por estar dentro do treino).
  Suplementar opcional: 2020-2021 (teto otimista, dentro de treino).
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

### Sizing — EQUAL WEIGHT 15%
Cada posição recebe 15% do capital corrente (`equity_hoje`). Não
pondera por força do sinal.

### Limites de posição
- Máximo **3 posições simultâneas**.
- Máximo **2 posições no mesmo setor**.

### Regras de saída (primeiro critério atingido)
Prioridade: **stop > take > prazo > reversão**.
- Stop loss: preço ≤ 0.92 × preço_entrada.
- Take profit: preço ≥ 1.15 × preço_entrada.
- Prazo: 5 dias úteis (fecha na abertura do 6º).
- Reversão: `ECON.avaliar(ticker, data_limite=data).score_total < -0.30`.

Nota de honestidade metodológica: com alvo de 5du, take-profit de +15%
dispara raramente — saída dominante é por prazo. **Confirmado empiricamente
no smoke test do PROGRAM: 0 takes em 9 trades (mar-jun/2024)**. Declarado
abertamente no relatório final.

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
- Regra vale por-ticker.
- Pula fins de semana ao somar dias úteis.

### Acoplamento com o PROGRAM
- **PROGRAM injeta `equity_hoje` = MTM ao fim de D-1** a cada
  `decidir(data, equity_hoje)`. Convenção travada: usa preços de
  fechamento de D-1, não de D. Anti-lookahead + evita reflexividade
  do circuit-breaker.
- **PROGRAM tem última palavra sobre fechamento** — validado por
  invariante testada: se stop/take intraday dispara no dia D, o
  ORQUESTRADOR não deve ter o ticker em `_posicoes` no momento de
  `_verificar_fechamentos` (senão o engine levanta `RuntimeError`).

### Custos no backtest (aplicados pelo PROGRAM)
0.3% corretagem + 0.1% slippage por operação = **0.4% por perna, 0.8% round-trip**.

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
- **Contratos `get_precos`, `get_noticias`, etc. exigem tz-aware SP
  em datas de fronteira** (`_validate_aware`). Consumidores precisam
  respeitar (ver adaptador do PROGRAM abaixo).

### JOURNAL como provedor de dados puro
- Coleta e entrega dados brutos. Não pontua, não pondera.
- `get_retornos_setor` agrega retornos setoriais (dado bruto). ECON faz
  a pontuação.

### Arquitetura modular das fontes
- Cada fonte externa em `agents/sources/` como classe própria:
  `CVMSource`, `GDELTSource`, `NewsAPISource`, **`BloombergCSVSource`**.
  Dataclass `Noticia` compartilhada em `noticia.py`.
- `JournalAgent` orquestra; parsing pesado nas classes específicas.

### Coleta de notícias — cascata com Bloomberg como primária
- **Cascata em `get_noticias`: Bloomberg CSV (1.0) → GDELT (suplementar) →
  NewsAPI (fallback).**
- **Bloomberg promovida a fonte primária** (17/07/2026) após descoberta
  de que GDELT e NewsAPI são insuficientes para janela ago-dez/2025:
  GDELT com IP penalizado (retries 65s → 138s → connection reset),
  NewsAPI free tier retorna 426 (Upgrade Required) em janelas históricas.
- Bloomberg é curado via Bloomberg Terminal na biblioteca da FGV,
  exportado para Excel bruto e parseado via `bloomberg_parser.py`
  gerando `data/bloomberg/parsed/noticias.csv`. Sem rate limit, sem
  dependência de API externa em runtime. Determinístico.
- Whitelist rígida: bloomberg.com, reuters.com, valor.globo.com,
  valor.com.br, broadcast.com.br, estadao.com.br, infomoney.com.br.
- Pesos: Bloomberg 1.0, Reuters/Valor 0.95, Broadcast 0.90, Estadão
  0.85, InfoMoney 0.75.
- `TICKER_PARA_NOME` em `config.py` resolve ticker→nome antes da busca.
- Cache por fonte em pickle, TTL 24h.

### Parser Bloomberg (implementado — commits 423914b + 5a4b8e4 + 91846cc)
- `agents/sources/bloomberg_parser.py`: Excel bruto → CSV limpo, 28 testes.
- Detecção dinâmica de coluna de conteúdo (A ou B) — algumas abas do
  Bloomberg exportam com índice tabular residual que empurra conteúdo
  pra coluna B.
- Segmentação por date-line (formato `MM/DD/YYYY HH:MM:SS[CODIGO]`),
  não por delimitador `<Back> Voltar` (mais robusto).
- Códigos de fonte mapeados: BN, BFW, PBN, BI. Códigos desconhecidos →
  "Bloomberg" genérico + aviso.
- **TICKER_ALIAS**: `AXIA3 → ELET3` (rebrand pós-privatização 2024;
  universo mantém nome histórico ELET3 para consistência com 2019-2023).
- Limpeza de chrome: rodapés PT+EN ("Para entrar em contato com",
  "To contact the reporter/editor/translator"), "Notícias recomendadas",
  tags "TICKER BZ" sem "Equity", runs finais de "NNN)". Preserva
  NOTE:/RELATED: (referências reais).
- Fallback de corpo para notícias sem markers `(Bloomberg) --` / `By`.
- Deduplicação por `(ticker, data, titulo)` dentro do arquivo.
- Determinismo: ordenação estável, dois runs byte-a-byte iguais.

### Integração via BloombergCSVSource (commit 5a4b8e4)
- `agents/sources/bloomberg_csv.py`: source primário no schema Noticia.
- Reconciliação de schema: `corpo + resumo_ia → conteudo` (resumo IA
  primeiro por ser mais denso), `data → publicado_em`, `peso → peso_fonte`.
- Lazy load do CSV inteiro na primeira chamada, sem TTL (fonte
  determinística local).
- Filtro por ticker exato + janela inclusiva (`<= data_limite`, casa
  com GDELT/NewsAPI).
- Loader legado do JournalAgent removido (-113 linhas). Uma única
  fonte de verdade sobre Bloomberg.

### Dataset unificado (commit 91846cc)
- `data/bloomberg/parsed/noticias.csv`: **653 notícias** cobrindo
  **24 tickers × 2024-2025**.
- Composição:
  - Excel do gate (ago-dez/2025, 4 tickers com dados): 164 notícias
  - trab.xlsx (2024-2025, 6 tickers): 125 notícias
  - trab-2.xlsx (2024-2025, 18 tickers): 366 notícias
- Total antes de dedup: 655. Duplicatas cross-arquivo removidas: 2.
- Distribuição por ano: 2024=241, 2025=412.
- Universo OOS e CSV alinhados: zero órfãos nos dois sentidos.
- ELET3 recebe 21 notícias via alias AXIA3→ELET3.
- Densidade por ticker: mediana ~20, PETR4/ITUB4/VALE3/BBDC4 mais
  densos (aparecem tanto no gate quanto no trab-2), RDOR3 com 16 é
  o menor.
- SHA-256 estável entre runs (determinismo verificado).
- Backup `.bak` das 164 originais preservado, gitignored.

### Densidade real dos dados (achado empírico)
- Média ~27 notícias/ticker em 2 anos = ~1 notícia/mês por ticker.
- **Estimativa: 20-40 trades no backtest oficial** (assumindo filtros
  atuais de score_econ > 0.30 e volume_relativo > 1.5).
- Aceitável mas apertado. IC95 do hit rate vai ficar largo (±20%).
- Se backtest der problema estatístico específico, aí sim vale
  segunda visita à FGV pra reforçar cobertura. Antes disso não.

### Deduplicação fuzzy entre fontes
- Similaridade de títulos via `difflib.SequenceMatcher` > 0.85 E
  diferença de publicação < 24h.
- `_DEDUP_SIM_MIN = 0.85`, `_DEDUP_HORAS_MAX = 24`.
- Processadas em ordem decrescente de peso; duplicatas de menor peso
  descartadas.

### GDELT blindado contra rate limit (Round 4)
- `GDELTRateLimitedError` e `GDELTUnavailableError` como exceções
  tipadas em `agents/sources/gdelt.py`.
- Backoff exponencial: 60s → 120s → 240s → 480s → 600s, 5 tentativas.
- Env var `GDELT_THROTTLE_SECONDS` (default 5). Rodadas oficiais usam 12s.
- Captura no `JournalAgent` com `gdelt_degradado_count` no
  `health_check()`.
- **Achado 17/07/2026:** mesmo com backoff, GDELT pode ficar
  inacessível por dias em janelas específicas devido a penalização de
  IP. Não é bug do backoff — é limitação estrutural da fonte gratuita.
  Bloomberg cobre essa lacuna.

### Cache negativo — TODO aberto
- **Problema descoberto no smoke test do PROGRAM (48 min cold run):**
  tickers com falha permanente no yfinance (JBSS3 pós-delistagem em
  06/06/2025, ELET3 flaky) são re-tentados a cada run porque falhas
  não entram no disk-cache. Cria overhead significativo.
- **Solução proposta (não implementada):** cache negativo com TTL longo
  (~7 dias) para respostas 404/vazias do yfinance. Antes do backtest
  oficial de produção.

### Fonte de fundamentos: CVM
- `dados.cvm.gov.br/dados/CIA_ABERTA/DOC/`. Módulo em `cvm.py`.
- ITR trimestral + DFP anual como ZIPs, cache em `data/cvm/`.
- Encoding latin-1, sep ";", decimal ",", valores em milhares de R$.
- Sempre consolidado (`_con_`), nunca individual.
- Versões reapresentadas: filtra máxima por (CNPJ_CIA, DT_REFER).
- Bancos usam BPB (não BPA+BPP) — código ramifica por setor.
- yfinance é fallback APENAS para setor.
- Fluxo TTM: DFP → TTM=anual; ITR → TTM=ULTIMO_YTD + (DFP_ano_ant −
  PENULTIMO_YTD).
- Estoque/balanço: point-in-time.
- `Fundamentals.data_recebimento_cvm` para rastreabilidade.

### Survivorship bias com membership por data
- `UNIVERSO_HISTORICO` cobre 2019-2025, com entrada/saida por ticker.
- Campos: setor, entrada, saida, confianca, fonte, cd_cvm (int), cnpj.
- `tickers_ativos(data_aware)` em `config.py:277`. Usada em TODOS
  os loops sobre universo.
- Casos emblemáticos: AMER3 (saída 12/jan/2023), **IRBR3 (saída
  02/jan/2023)**, JBSS3 (saída 06/06/2025).

### Universo atualizado (commit 229e91a)
- **AMER3 e IRBR3 mantidos** em `UNIVERSO_HISTORICO` com data_saida
  registrada (usados no warmup e treino MATH&ML 2020-2023). Não
  aparecem no backtest OOS 2024-2025 via `tickers_ativos()`.
- **Substituições no OOS:**
  - AMER3 → **ASAI3** (setor `varejo`): Assai Atacadista, spin-off
    do GPA. Entrada Ibov: estimativa `2021-05-03` — confirmar via
    B3/Bloomberg em passe futuro. `cd_cvm=25372`, CNPJ
    `06.057.223/0001-71`.
  - IRBR3 → **BBSE3** (setor `outros`): BB Seguridade. Entrada
    Ibov: `None` (assumido ativo desde 2019). `cd_cvm=23159`,
    CNPJ `17.344.597/0001-94`. Não criei setor "seguros" —
    IRBR3 também era "outros", mantém consistência taxonômica.
- **Total ativo em 2024-06-01: 24 tickers.**
- Setores em risco de sub-representação (Varejo com 2 e Seguros com 0)
  agora preservados.

### Fontes em camadas (resumo)
- **Notícias: Bloomberg CSV (primária, 1.0) > GDELT (suplementar,
  whitelist, backoff, IP-flaky) > NewsAPI (fallback, 30d gratuito).**
- Macro: BCB SGS primário, FRED fallback automático.
- Preços: yfinance com duas versões (ajustada e bruta).
- Fundamentos: CVM (primária), yfinance só para setor.

## DECISÕES CONSOLIDADAS DO ECON

### Função e contrato
- Analista fundamentalista qualitativo via Claude Haiku 4.5
  (`claude-haiku-4-5-20251001`). **Modelo escolhido empiricamente
  no gate de custo (29/07/2026).**
- Recebe dossiê do JOURNAL (notícia + fundamentos CVM + macro +
  setoriais) e devolve `ScoreEcon`.
- **Método principal:** `avaliar(ticker, data_limite)`.
  `data_limite` deve ser passado explícito (kwarg) por consumidores.

### API key validada (17/07/2026)
- Chave configurada no `.env` (fora do git via `.gitignore`).
- Teste de sanidade em 1 chamada real (PETR4, data_limite 2025-10-15):
  - `ScoreEcon` retornado com shape esperado.
  - `score_total=-0.25`, `tem_evento=True`, `n_noticias=6`,
    `confianca=0.65`, `modelo=claude-haiku-4-5-20251001`.
  - Justificativa coerente (Ibama/Foz do Amazonas + queda setorial
    vs P/L baixo).
  - Custo real: US$0.0045/chamada (2917 tokens input, 311 output).
  - Latência dominada por coleta de notícias (148s), não LLM (~5s).

### ScoreEcon — campos principais
- `score_total` [-1, +1]: IMPACTO DA NOTÍCIA no excesso ao Ibov em 5d.
  Opção A — NÃO combina saúde financeira / setor / macro.
- `comp_noticia`: base do `score_total`.
- `comp_saude_financeira`, `comp_setorial`, `comp_macro`: CONTEXTO
  considerado — NÃO somados ao total.
- `confianca` [0, 1], `tem_evento`, `n_noticias`.
- `noticias_hashes`: rastreabilidade.
- `data_noticia_mais_recente`: tz-aware `America/Sao_Paulo` ou NaT.
- `justificativa`, `modelo`, `avisos`.

### Decisão "Opção A" sobre o score
- ECON pontua o MECANISMO da notícia; MATH&ML otimiza pesos.
- Reduz colinearidade EXPLÍCITA com features cruas que MATH&ML recebe
  do JOURNAL.

### Integração com MATH&ML
- `score_total` entra como feature principal.
- Saúde/setor/macro entram no MATH&ML como features CRUAS do JOURNAL,
  NÃO `comp_*` do ECON.

### Arquitetura técnica
- Tool use forçado + temperature=0. Reprodutibilidade vem do CACHE
  VERSIONADO (`_PROMPT_VERSION` na chave), não do temperature.
- Event-driven: sem notícia → `ScoreEcon` neutro sem chamar Claude.
- Degradação graciosa: nunca levanta exceção; devolve neutro + aviso;
  NÃO cacheia falha.

### Anti-lookahead do LLM (3 defesas)
- Cutoffs do Haiku 4.5 (fim de mês como fronteira conservadora):
  - Reliable knowledge: fev/2025
  - Training data: **jul/2025** (inclusive — jul entra em treino)
- **Janela genuinamente LIMPA = ago-dez/2025** (estritamente pós-training).
- Defesa 1 — IC segmentado: fronteira no TRAINING cutoff.
- Defesa 2 — Placebo com dois modos: `swap` e `identidade_pura`.
- Defesa 3 — Auditoria regex de justificativas.

### Gate de custo Haiku 4.5 vs Sonnet 4.6 — EXECUTADO (commit 85c69f9)
- **Data:** 29/07/2026, 03:20 SP.
- **N efetivo:** 318 eventos limpos em janela ago-dez/2025 (fronteira:
  `data_noticia_mais_recente >= 2025-08-01`).
- **Fonte:** Bloomberg-only (GDELT/NewsAPI neutralizados pra evitar
  IP-flakiness e viés de cobertura).
- **Amostra por ticker:** PETR4=90, ITUB4=82, VALE3=75, BBDC4=71.
- **Amostra por mês:** ago=55, set=68, out=78, nov=69, dez=48.
- **Método:** IC Spearman(score, retorno beta-ajustado 5du); block
  bootstrap por data (blocos 5du, 10k iterações, seed=42).

**Resultados:**
- **Haiku 4.5:** IC=+0.0468 [-0.1535, +0.2286]; custo US$1.4967;
  latência mediana 4.02s (P90=5.12, P95=5.71); fallback 0.6%.
- **Sonnet 4.6:** IC=+0.0169 [-0.1937, +0.2111]; custo US$4.7650;
  latência mediana 9.66s (P90=12.09, P95=13.01); fallback 0.3%.
- **ΔIC (Sonnet − Haiku) = −0.0298 [-0.1267, +0.0770]** pareado.

**Decisão: HAIKU 4.5.**
- Critério: ΔIC > 0.05 → Sonnet; < 0.03 → Haiku; zona cinza → Haiku.
- Sonnet ficou **marginalmente pior** que Haiku (não zona cinza —
  decisão empírica, não por default).
- Haiku é ~3.2x mais barato E ~2.4x mais rápido.
- Implicação pro backtest oficial (~2500 chamadas): Haiku ~2.8h de
  wall-clock; Sonnet teria sido ~6.7h.
- **Custo real do gate: US$6.26 (78% do hard cap US$8).**
  Crédito restante estimado: US$3.74.
- Relatório: `calibration/results/RELATORIO_GATE_CUSTO.md`.
- CSV bruto: `calibration/results/gate_custo_resultado_completo.csv`.

### Calibração real do ECON — próxima ação
- **Bloqueio removido.** Dataset unificado (653 notícias em 24 tickers,
  2024-2025) permite calibração completa.
- Métrica primária: IC Spearman, meta > 0.15 com IC95 bootstrap não
  cruzando zero.
- Baseline lexical + GAP para justificar custo do LLM.
- Loop de prompt ≤ 10 iterações. Cada iteração bumpa `_PROMPT_VERSION`.
- Estimativa de custo: US$5-15.

### Divergência metodológica registrada (a reconciliar)
- `calibration/econ_calibration.py::comparar_modelos()` usa bootstrap
  i.i.d.
- Gate de custo usa block bootstrap por data (correto para retornos
  correlacionados por data).
- **TODO:** migrar `comparar_modelos()` para block bootstrap na
  calibração completa.

### Próximos passos do ECON (em ordem)
1. ~~Smoke test com API key~~ ✓ (17/07/2026).
2. ~~Popular Bloomberg CSV~~ ✓ (2 visitas FGV, 653 notícias).
3. ~~Executar gate de custo~~ ✓ (29/07/2026, Haiku venceu).
4. **Calibração real: amostra de eventos + prompt engineering.**
5. Loop de prompt engineering ≤ 10 iterações.
6. Congelar prompt + `RELATORIO_CALIBRACAO_ECON.md` final.

## DECISÕES CONSOLIDADAS DO MATH&ML

### Filosofia central: hipótese antes de padrão
- Cada feature mapeia 1-para-1 a hipótese testável da literatura, com
  sinal esperado explícito.
- `importancia_features()` cruza ganho observado com sinal teórico;
  direção invertida → red flag reportado.

### Regra 1: target = retorno beta-ajustado 5du
- `y_i(t) = r_i_fwd − beta_i(t) × r_ibov_fwd`, sobre `Close_raw`
  (split-only) para ação e Ibov.
- Beta por `np.polyfit(r_ibov, r_i, deg=1)` sobre 252du terminando em t.
- Fallback `beta=1.0` se janela tiver < 200 observações.

### Regra 2: fundamentos por DT_RECEB
- `journal.get_fundamentals(ticker, data_limite)` já corta por
  `data_recebimento_cvm <= data_limite`.

### Regra 3: sinal = ranking cross-sectional
- `MathMLAgent` entrega **previsão contínua + ranking**; seleção
  final é do ORQUESTRADOR.

### Regra 4: linhas de treino = todos os dias-ticker do universo ativo
- Em dias sem evento, `score_econ=0`.
- **Sample weight de 5.0 nos dias de evento como default**.
- IC reportado em DUAS agregações: total e subset `tem_evento=True`.

### Regra 5: períodos
- Warmup desde 2019-01-01.
- Treino 2020-2023. Backtest OOS 2024-2025.
- Walk-forward com janela expansiva, `freq='MS'` mensal default.

### Modelo: GradientBoosting raso de propósito
- `GradientBoostingRegressor(max_depth=3, learning_rate=0.05,
  subsample=0.8, random_state=42)`.
- `n_estimators` por regra de platô **com fallback para argmax**.
- `cv_report` inclui `n_platau`, `n_argmax`, `n_escolhido`, `fonte`.

### Vetor de features (17 features + 3 flags de auditoria)

**Momentum e reversão:**
- `mom_12_1` (+), `rev_1m` (−).

**Fluxo de resultados (growth, não PEAD):**
- `dias_desde_resultado`, `crescimento_lucro_yoy` (+).

**Qualidade/valor:**
- `pl` (−), `pvp` (−), `roe` (+), `margem` (+), `divida_ebitda` (−).

**Volume:**
- `volume_relativo` (+). Duas semânticas: modelo consome imputado
  (mediana cross-sectional); `prever_universo` retorna cru (NaN
  preservado).

**Estado macro:**
- `selic_nivel`, `selic_var_21d`, `cambio_var_21d`.

**Notícia:**
- `score_econ` (+), `econ_confianca`, `econ_n_noticias`.

**Flags de auditoria:**
- `beta_fallback`, `fundamental_imputado`, `econ_degradado`.

### Regras anti-lookahead (defesa em profundidade)
1. Toda feature em t usa só dados `<= t`.
2. Label usa `t+1..t+5` (forward — regras 3 e 4 obrigatórias).
3. Drop de label incompleto ANTES de `_montar_features`.
4. Purge + embargo López de Prado: `PurgedTimeSeriesSplit`, embargo 5du.
5. Imputação cross-sectional do dia; fallback global só do subset de treino.
6. Walk-forward respeita o mesmo embargo.
7. `_assert_no_lookahead` levanta `LookaheadError` em 4 fontes.

### Arquitetura de pré-fetch (Round 5)
- **Bug medido:** padrão anterior chamava JOURNAL ~10k vezes por run.
- **Solução em duas fases:**
  1. `_prefetch`: cada fonte chamada UMA vez cobrindo range inteiro.
  2. `_montar_features` e `_alvo` lêem SÓ do cache — zero I/O na montagem.
- `_prefetch` tolera tickers sem dados no yfinance.
- Cold-cache 24 tickers × 6 anos ~90 min; reruns ~5s via disk-cache.

### Ajustes do run oficial (Round 7)
- Fix 1: Fallback do platô (n_platau < 0.3 × n_argmax → usa argmax).
- Fix 2: sample_weight_eventos=5.0 default.
- Fix 3: Snapshot do cv_report antes do walk_forward.

### Mock estruturado do ECON
- Sinal controlado: `α × z(y) + ruído`.
- Quatro modos: ruído (0.00), fraco (0.10), meta (0.15), forte (0.20).
- **Achado do smoke test end-to-end:** `set_cache(None)` após treino
  para evitar envenenar avaliações no serve-time.

### Avaliação e métricas
- IC de Spearman total e no subset `tem_evento=True`.
- IC95 por block bootstrap por data.
- Três baselines: B1 (score_econ), B2 (mom_12_1), B3 (intercepto).
- GAP = IC_modelo − max(B1, B2, B3).

### Contrato de saída do `prever_universo` (8 colunas)

| Coluna | Tipo | Semântica |
|---|---|---|
| `ticker` | str | Ex.: "PETR4.SA" |
| `y_pred` | float | Retorno idiossincrático 5du previsto |
| `score_econ` | float ∈ [-1, +1] | Score da notícia |
| `tem_evento` | bool | Se houve notícia relevante |
| `rank` | int | Ranking cross-sectional (1 = melhor) |
| `volume_relativo` | float | Cru, pré-imputação, NaN preservado |
| `data_noticia_mais_recente` | datetime tz-aware SP ou NaT | Do ScoreEcon |
| `setor` | str | Do `UNIVERSO_HISTORICO` |

### Resultados observados — Run oficial de sensibilidade
(24 tickers, 2020-2023 + 2024-2025, 4 modos, 90.8 min; commit 399145c).

| Modo | IC_alvo | n_platau→n_est | IC_evento | GAP vs B1 |
|---|---|---|---|---|
| ruído | 0.00 | 1→4 (fallback) | +0.029 | +0.029 |
| fraco | 0.10 | 1→4 (fallback) | +0.095 | +0.001 |
| **meta** | **0.15** | **1→4 (fallback)** | **+0.147** | **+0.0037** |
| forte | 0.20 | 3→17 (fallback) | +0.189 | −0.0004 |

**Importância no modo meta:** `score_econ` #1 (0.256), `mom_12_1` #2
(0.236), `rev_1m` #3. Zero features com sinal invertido.

### Determinismo e reprodutibilidade
- `PYTHONHASHSEED` ∈ {0, 1, 12345}.
- `_stable_seed(ticker, t)` para RNG do mock.

## DECISÕES CONSOLIDADAS DO ORQUESTRADOR

### Estado da implementação
- 564 linhas em `agents/orchestrator.py`.
- 771 linhas em `tests/test_orchestrator.py`.
- 56 testes determinísticos + 3 de integração.

### Contrato público
```python
class OrchestratorAgent:
    def __init__(self, journal, econ, math_ml, config, tickers_ativos=None)
    def decidir(self, data, equity_hoje) -> DecisaoDia
    def notificar_execucao(self, ticker, setor, preco_execucao, data_execucao) -> None
    def notificar_fechamento(self, ticker, data_fechamento) -> None  # agnóstico ao motivo
    def status(self) -> dict
```

Dataclasses: `OrchestratorConfig` (frozen), `Ordem`, `FechamentoOrdem`
(motivo Literal["prazo", "reversao"]), `DecisaoDia`.

### Métodos privados
- `_atualizar_pausa(data, equity_hoje) → float`: drawdown + circuit-breaker.
- `_selecionar_ordens(df, data)`: pool + top-N + limite setorial.
- `_resolver_data_execucao(row, data)`: D+1 vs D+2 com corte 17h05.
- `_verificar_fechamentos(data)`: prazo antes de reversão.

### Padrões-chave preservados
- `tickers_ativos` injetável para testes determinísticos.
- Anti-lookahead: `data_limite=data` sempre explícito nas chamadas ao ECON.
- Não chama JOURNAL diretamente.
- `_pausado_ate` não é limpo ao expirar — "estar pausado" é derivado.
- **Todas as datas de fronteira exigem tz-aware SP** (`_exigir_aware`).

## DECISÕES CONSOLIDADAS DO PROGRAM (Etapas 1, 2, 3 fechadas)

### Estado da implementação
- **Etapa 1 completa** (BacktestEngine core), commits 03d9799 (1.1),
  84429c6 (1.2), e869767 (1.3). ~560 linhas em `backtest/engine.py`.
- **Etapa 2 completa** (Monte Carlo), commit da Etapa 2 (bootstrap +
  permutação com determinismo). ~350 linhas em `backtest/monte_carlo.py`.
- **Etapa 3 completa** (métricas de performance + atribuição). ~460 linhas
  em `backtest/metrics.py`.
- **Total: 98 testes determinísticos** do PROGRAM (40 engine + 33 metrics
  + 25 monte carlo).
- **Smoke test de integração** com 4 agentes reais (JOURNAL/MATH&ML/
  ORQUESTRADOR + mock estruturado do ECON) executando janela mar-jun/2024,
  83 dias úteis BMF, exit 0.

### Contrato público (Etapa 1 — engine)
```python
@dataclass(frozen=True)
class BacktestConfig:
    capital_inicial: float = 100_000.0
    corretagem: float = 0.003
    slippage: float = 0.001
    stop_pct: float = 0.08
    take_pct: float = 0.15
    sizing_pct: float = 0.15

@dataclass(frozen=True)
class ResultadoBacktest:
    trades: pd.DataFrame
    equity_diario: pd.Series
    avisos: list[dict]
    config: BacktestConfig
    data_inicio: pd.Timestamp
    data_fim: pd.Timestamp
    n_dias_uteis: int
    n_trades: int
    capital_final: float

class BacktestEngine:
    def __init__(self, journal, orquestrador, config: Optional[BacktestConfig] = None)
    def rodar_backtest(self, data_inicio, data_fim) -> ResultadoBacktest
```

### Dataclasses internas (engine)
- `PosicaoInterna` (mutável): estado de posição aberta com stop/take,
  custo_entrada, snapshots de y_pred, score_econ, rank.
- `TradeRegistro` (frozen): registro completo do trade fechado com
  `motivo: Literal["stop", "take", "prazo", "reversao", "fim_backtest"]`.

### Fluxo diário (Ordem B — travada)
```
Passo 0: equity_hoje = MTM ao fim de D-1
Passo 1: detectar stop/take intraday (Low/High do dia D)
         - Prioridade STOP > TAKE
         - preço = stop_price ou take_price exato (sem slippage extra)
         - Chama orq.notificar_fechamento(ticker, D)
Passo 2: dec = orq.decidir(D, equity_hoje)  ← equity é do fim de D-1
Passo 3: Executar dec.fechamentos (Open[D+1] ou próximo pregão)
         - INVARIANTE: nenhum ticker aqui pode ter sido fechado no Passo 1
           (RuntimeError se violar)
Passo 4: Executar dec.novas_ordens (Open[data_execucao])
         - qtd = int((0.15 × equity_hoje) / preco_entrada)
         - Custo entrada 0.4%, degradação graciosa (aviso)
         - Chama orq.notificar_execucao
Passo 5: Registra equity_fim_D (mark-to-market com Close[D])
```

### Adaptador tz na fronteira (decisão travada)
- **Interior do engine 100% naive.** Datas dentro do PROGRAM são
  `pd.Timestamp` sem timezone.
- **Fronteira com ORQUESTRADOR e JOURNAL:** ambos exigem tz-aware SP
  (`_exigir_aware`, `_validate_aware`).
- **Solução:**
  - `_para_sp(data)`: função pura, idempotente, converte naive → SP
    preservando hora de parede (`tz_localize`, não `tz_convert`).
  - `_get_precos(ticker, data_inicio, data_limite)`: choke-point único
    de acesso ao JOURNAL. Localiza limites em SP na entrada, captura
    `DadoIndisponivel`/`LookaheadError` → DataFrame vazio, normaliza
    índice a naive na saída.
- **Regra dura:** nenhum outro método chama `self._journal.get_precos`
  diretamente; nenhuma chamada ao ORQUESTRADOR sem `_para_sp(data)`
  inline.
- Fakes estritos em `tests/fakes.py` replicam `_exigir_aware`/
  `_validate_aware` — rejeitam naive. Se algum call-site esquecer o
  adaptador, teste quebra em vez de passar em "naive-land".

### Calendário B3
- `pandas_market_calendars` com calendário `'BMF'`.
- NÃO usar `pd.bdate_range` (ignora feriados nacionais).
- Validado: 2024-03-29 (Sexta-feira Santa) e 2024-05-30 (Corpus Christi)
  ausentes do calendário retornado.
- `_calendario_bmf` retorna DatetimeIndex naive (após `tz_localize(None)`).

### Tensão R8×R1 — comportamento aceito, documentado, testado
Duas regras corretas geram diferença numérica esperada:
- **R1** (Passo 5): MTM ao fim de D é contabilidade sem custo, marca a
  mercado pelo Close.
- **R8** (fim do backtest): se há posição aberta em data_fim, força
  fechamento pelo Close[data_fim] com custo de saída de 0.4% aplicado.

**Consequência:** quando há posições abertas no último dia,
`capital_final ≤ equity_diario.iloc[-1]` — a diferença é a soma dos
custos de saída forçada. Igualdade só quando não há posições abertas.

**Por que essa política em vez de "sempre igual":** honestidade
metodológica. Não cobrar custo na liquidação forçada maquiaria o
resultado. Banca de asset management pega.

**Testes dedicados:** `test_capital_final_menor_que_equity_final_quando_
fim_backtest_dispara` valida a diferença exata (equity[-1]=100690,
capital_final=100627, Δ=63 = custo de saída).

### Determinismo em produção (evidência de qualidade)
- Smoke test cold vs warm: byte-a-byte idênticos (mesmos 9 trades,
  mesma decomposição, mesmo `capital_final = R$ 96.396,32`).
- Corta suspeita de cherry-picking na banca pela raiz.
- Fundamentos: `PYTHONHASHSEED` fixado, `_stable_seed` no mock ECON,
  iteração alfabética sobre dicionários, adaptador tz determinístico,
  monkeypatch de TTL do JOURNAL na janela histórica.

### Fallback do `_close_marcacao` (documentado + testado)
- Se `Close[D]` não existe (ticker sem barra no dia), usa Close mais
  recente ≤ D disponível (carry-forward).
- Documentado em docstring; teste dedicado `test_close_marcacao_
  fallback_para_close_anterior_quando_barra_ausente` (asserta
  carry-forward: `Close[03-05]=107` → equity `100.990`).

### Achados do smoke test do PROGRAM (mar-jun/2024)
1. **9 trades gerados:** 1 stop, 5 prazo, 3 reversão, 0 take, 0 fim_backtest.
   Confirma que prazo é saída dominante (regra R1 do relatório final).
2. **Cold run 48 min vs warm 2 min.** Causa: yfinance throttling + falta
   de cache negativo para tickers com falha permanente (JBSS3, ELET3).
   Loop do engine é rápido; overhead vem do JOURNAL.
3. **Determinismo cold==warm byte-a-byte** validado.
4. **Zero crashes de fronteira tz, invariante de sincronização,
   ou anti-lookahead.** Adaptador funciona em produção.

## PROGRAM Etapa 3 (métricas de performance) — fechada

### Contrato público (metrics)
```python
@dataclass(frozen=True)
class MetricasBacktest:
    # Retorno
    retorno_total: float
    retorno_anualizado: float

    # Risco
    volatilidade_anualizada: float
    sharpe_anualizado: float
    sortino_anualizado: float
    maximum_drawdown: float          # negativo
    duracao_mdd_dias: int
    dias_ate_recuperacao_mdd: Optional[int]

    # Trades
    n_trades_total: int
    n_trades_vencedores: int
    n_trades_perdedores: int
    hit_rate: Optional[float]
    payoff_medio: Optional[float]
    profit_factor: Optional[float]

    # Custos
    custo_total_pago: float
    custo_como_pct_capital_inicial: float

    # Metadados
    taxa_livre_risco_anualizada: float
    fonte_taxa_livre_risco: str  # "cdi" | "selic" | "override" | "zero_por_falha"

def calcular_metricas(
    resultado: ResultadoBacktest,
    journal,
    taxa_livre_risco_override: Optional[float] = None,
) -> MetricasBacktest: ...

def atribuir_por_motivo(resultado: ResultadoBacktest) -> pd.DataFrame: ...
def atribuir_por_setor(resultado: ResultadoBacktest) -> pd.DataFrame: ...
```

### Convenções travadas (metrics)
- **Volatilidade e Sharpe:** std amostral (ddof=1), padrão de asset
  pricing empírico.
- **Sortino:** downside deviation dividida por **N total** de observações
  (Sortino 1994, Nawrocki 1999), não por n_negativos.
  Fórmula: `downside_dev = sqrt(sum(min(r-MAR,0)²) / N)`.
- **Taxa livre de risco:** Selic realizada (SGS 11, `selic_diaria`) via
  `journal.get_macro`. Fallback: override → selic → zero_por_falha.
  CDI não existe no `get_macro`; Selic realizada é proxy aceitável.
- **Percentis:** `np.percentile` default (linear interpolation), convenção
  Fama-French / MATLAB / R.
- **MDD:** duração pico-a-vale + dias até recuperação (None se não
  recuperou).

### Convenções degeneradas (documentadas)
- `n_trades == 0` → hit_rate/payoff/profit_factor = `None`.
- `n_vencedores == 0` (só perdedores) → payoff = 0.0, profit_factor = 0.0.
- `n_perdedores == 0` (só vencedores) → payoff = `+inf`, profit_factor = `+inf`.

### Atribuição de retorno
- `atribuir_por_motivo`: agrupa por `motivo` do TradeRegistro. Colunas:
  n_trades, pnl_liquido_total, pnl_liquido_medio, hit_rate, pct_do_total_pnl.
  Ordenado por pnl_liquido_total descendente.
- `atribuir_por_setor`: idem, agrupado por `setor`.
- Determinismo via mergesort estável + tiebreak alfabético.
- `pct_do_total_pnl` soma 1.0 mesmo com grupos negativos.

### Testes (33 verdes)
Grupos A (retorno/vol, 5), B (Sharpe/Sortino + Ajuste 1, 6),
C (drawdown, 5), D (trades + Ajuste 3, 6), E (atribuição, 5),
F (custos, 3), G (robustez, 3).

## PROGRAM Etapa 2 (Monte Carlo) — fechada

### Contrato público (monte carlo)
```python
@dataclass(frozen=True)
class ResultadoMonteCarlo:
    tecnica: str  # "bootstrap_retornos" | "permutacao_ordem"
    n_simulacoes: int
    seed: int

    # Distribuição de retorno total
    retornos_simulados: np.ndarray
    retorno_p05, retorno_p25, retorno_p50, retorno_p75, retorno_p95: float
    retorno_medio, retorno_desvio: float

    # Distribuição de MDD reamostrado
    mdds_simulados: np.ndarray  # todos negativos ou zero
    mdd_p05, mdd_p25, mdd_p50, mdd_p75, mdd_p95: float

    # Probabilidades condicionais
    prob_retorno_positivo: float
    prob_supera_ibov: Optional[float]  # None se retorno_ibov=None
    prob_mdd_melhor_que_20pct: float

    # Metadados
    retorno_ibov_referencia: Optional[float]

def rodar_bootstrap_retornos_trades(
    resultado: ResultadoBacktest,
    n_simulacoes: int = 10_000,
    seed: int = 12345,
    retorno_ibov: Optional[float] = None,
) -> ResultadoMonteCarlo: ...

def rodar_permutacao_ordem_trades(
    resultado: ResultadoBacktest,
    n_simulacoes: int = 10_000,
    seed: int = 12345,
    retorno_ibov: Optional[float] = None,
) -> ResultadoMonteCarlo: ...
```

### Duas técnicas independentes
- **Bootstrap com reposição:** sorteia N trades da lista realizada
  com reposição, 10k simulações. Testa "distribuição de retornos por
  trade é robusta?".
- **Permutação sem reposição:** reembaralha ordem cronológica sem
  reposição. Testa "risco de sequência é robusto?".
- Ambas retornam distribuições SEPARADAS. Não combinadas.

### Convenções travadas (monte carlo)
- **Vetorização batched:** matriz (B, N) + cumsum + running-max
  vectorizado. 10k simulações em milissegundos, não loop Python.
- **Núcleo `_rodar` compartilhado** entre bootstrap e permutação (DRY).
- **`_mdd_reamostrado` local vetorizado** (não acopla a
  `metrics._drawdown` privado; permite batch).
- **Determinismo:** `np.random.default_rng(seed)` por chamada, sem
  `np.random.seed` global. Dois runs idênticos byte-a-byte.
- **Comparação com Ibov dentro do MC:** `P(JEMPO > Ibov)` via parâmetro
  escalar `retorno_ibov`. Ibov NÃO reamostrado — é benchmark passivo
  fixo do mesmo período.
- **MDD reamostrado documentado como aproximação:** equity sintética
  sem gaps temporais. Distinto do MDD real (que usa equity diária).
  Reportado separadamente nos gráficos.
- **Permutação preserva retorno total** (soma comutativa em float64).
  O que varia é MDD, não retorno. Bootstrap testa retorno; permutação
  testa risco de sequência.

### Testes (25 verdes)
Grupos A (determinismo, 4), B (bootstrap, 5), C (permutação, 4),
D (MDD reamostrado, 4), E (comparação Ibov, 3), F (robustez, 3),
G (contrato, 2).

**Testes críticos com asserts analíticos:**
- Teste 13 (`test_permutacao_retorno_final_igual_em_todas_simulacoes`):
  `retorno_desvio == 0.0` exato via P&L inteiros (soma comutativa em
  float64).
- Teste 15 (`test_mdd_permutacao_mesmo_conjunto_trades_produz_mdds_
  diferentes`): min/max MDD derivados analiticamente enumerando as 24
  permutações de `[+1000,+1000,-500,-500]`. Assert de extremos exatos:
  `min(mdds) = -0.01`, `max(mdds) = -500/101000`.

### O que fica fora do escopo das Etapas 1-3
- **Etapa 4:** Visualizações e notebook de apresentação
  (`backtest/plots.py`, `notebooks/apresentacao_banca.ipynb`).
- **Etapa 5:** Integração com ECON real (após calibração).

## SMOKE TESTS (validação de integração)

### `scripts/smoke_test_e2e.py` — 4 agentes originais
- Valida JOURNAL/ECON mock/MATH&ML/ORQUESTRADOR reais em cadeia.
- mar-abr/2024, 43 dias úteis. Executável, exit 0, ~2 min.
- Não é backtest — equity constante, sem P&L nem custos.
- 5 ordens, 5 fechamentos (3 prazo, 2 reversão). Cache JOURNAL 78% hit.

### `scripts/smoke_test_program.py` — 4 agentes + BacktestEngine
- Valida 4 agentes reais (JOURNAL/MATH&ML/ORQUESTRADOR + mock ECON) +
  BacktestEngine em cadeia.
- mar-jun/2024, 83 dias úteis BMF. Executável, exit 0.
- Cold: 48 min. Warm: 2 min. **Cold==warm byte-a-byte.**
- 9 trades. Reusa `_EconMockAdapter` do smoke_test_e2e sem duplicação.

## Stack técnica
Python, Claude API (Anthropic, Haiku 4.5), scikit-learn, pandas,
numpy, yfinance, pyarrow (parquet), requests, matplotlib,
`pandas_market_calendars`, openpyxl. Macro via BCB SGS (sem chave) com
FRED fallback. Fundamentos via CVM aberto. Dev no Mac (M1) via Claude Code.

## Estrutura de arquivos
```
agents/
- journal.py
- econ.py             (EconAgent + ScoreEcon; método principal avaliar)
- math_ml.py          (MathMLAgent com _DatasetCache, _prefetch,
                       make_econ_mock, prever_universo com 8 colunas)
- orchestrator.py     (56 testes; contratos exigem tz-aware SP)
- sources/
  - noticia.py, cvm.py, gdelt.py, newsapi.py
  - bloomberg_parser.py    (Excel bruto -> CSV limpo; 28 testes)
  - bloomberg_csv.py       (fonte primaria; 18 testes)
backtest/
- __init__.py         (exporta BacktestEngine, BacktestConfig,
                       ResultadoBacktest, TradeRegistro, PosicaoInterna,
                       MetricasBacktest, calcular_metricas,
                       atribuir_por_motivo, atribuir_por_setor,
                       ResultadoMonteCarlo,
                       rodar_bootstrap_retornos_trades,
                       rodar_permutacao_ordem_trades)
- engine.py           (~560 linhas, Etapa 1 completa)
- metrics.py          (~460 linhas, Etapa 3 completa)
- monte_carlo.py      (~350 linhas, Etapa 2 completa)
- plots.py            (não implementado — Etapa 4)
calibration/
- econ_calibration.py                (bootstrap i.i.d. — TODO migrar)
- gate_custo_haiku_vs_sonnet.py      (executado 29/07/2026)
- results/
  - RELATORIO_CALIBRACAO_MATHML.md   (versionado, commit 399145c)
  - RELATORIO_CALIBRACAO_ECON.md     (WIP, próxima ação)
  - RELATORIO_GATE_CUSTO.md          (executado, versionado 85c69f9)
  - gate_custo_resultado_completo.csv
  - gate_custo_intermediario.csv     (checkpoint incremental)
data/
- cache/              (pickle, TTL 24h)
- bloomberg/
  - raw/              (Excel brutos, gitignored)
  - parsed/
    - noticias.csv    (653 notícias, 24 tickers, 2024-2025 + gate)
    - noticias.csv.bak (backup das 164 originais, gitignored)
- cvm/raw/, cvm/processed/
scripts/
- smoke_test_mathml.py, smoke_test_v2.py
- smoke_test_orquestrador.py       (fluxo 10 dias com Fakes)
- smoke_test_e2e.py                (4 agentes reais, mar-abr/2024)
- smoke_test_program.py            (4 agentes + BacktestEngine, mar-jun/2024)
- unificar_bloomberg.py            (parser + concat + dedup dos 3 Excel)
- debug_dataset.py, diagnostico_gbm.py, sensibilidade_econ.py
tests/
- test_journal.py, test_cvm.py, test_gdelt.py, test_newsapi.py
- test_econ.py, test_econ_calibration.py
- test_math_ml.py     (35 testes)
- test_orchestrator.py (56 testes + 3 de integração)
- test_engine.py      (40 testes, incluindo adaptador tz e fallback)
- test_metrics.py     (33 testes)
- test_monte_carlo.py (25 testes)
- test_gate_custo.py  (5 testes)
- test_bloomberg_parser.py (28 testes)
- test_bloomberg_csv.py    (18 testes)
- fakes.py            (Fakes estritos com _exigir_aware/_validate_aware)
```

## Timeline dos commits recentes (últimas 24h da última sessão)

```
91846cc journal(bloomberg): dataset unificado 2024-2025 (24 tickers, 653 noticias)
229e91a config(universo): AMER3/IRBR3 substituidos por ASAI3/BBSE3 no OOS
85c69f9 calibration(gate): Etapa 3 — gate de custo Haiku 4.5 vs Sonnet 4.6
5a4b8e4 journal(bloomberg): parser Etapa 2 — integracao BloombergCSVSource
423914b journal(bloomberg): parser Etapa 1 — Excel bruto -> CSV limpo
```

## Decisões pendentes e ordem prática

### Bloqueio removido: dados Bloomberg populados
Dataset unificado disponível: 653 notícias em 24 tickers cobrindo
2024-2025. Gate de custo executado. Backtest oficial pronto pra rodar
tecnicamente — só falta ECON calibrado.

### Roadmap ajustado (~3 semanas até apresentação)

**Semana 1 (concluída):**
- ✅ Etapas 1-3 do PROGRAM (engine, Monte Carlo, métricas)
- ✅ Gate de custo (Haiku venceu, US$6.26 real)
- ✅ Coleta Bloomberg (2 visitas FGV, 653 notícias)
- ✅ Universo atualizado (ASAI3/BBSE3)

**Semana 2 (próxima):**
- 🔲 **Calibração real do ECON** com prompt engineering (≤10 iterações).
  Meta: IC > 0.15, US$5-15 estimado.
- 🔲 **PROGRAM Etapa 4** (visualizações + notebook estrutural) —
  autônoma, pode paralelizar.

**Semana 3:**
- 🔲 Backtest oficial 2024-2025 com ECON real (~US$10-15).
- 🔲 Integração do backtest oficial no notebook.
- 🔲 Relatório final.

**Semana 4:**
- 🔲 Revisão + polimento.
- 🔲 Apresentação.

### Plano B se calibração real estourar
Se calibração real der ruim (custo, iterações, IC baixo), manter ECON
mock no relatório final, declarar abertamente como limitação, apresentar
sistema com sensibilidade completa em 4 modos (que já existe).

### TODOs abertos (não bloqueiam)
- **Cache negativo no JOURNAL** para tickers com falha permanente no
  yfinance (JBSS3, AXIA3/ELET3). Reduz cold run do backtest oficial.
- **Migrar `comparar_modelos()` para block bootstrap por data**
  (reconciliação com o gate de custo).
- **`_health_bloomberg` com glob antigo**: reporta "vazio" mesmo com
  parsed populado. Cosmético.
- **Confirmar `entrada` correta de ASAI3 no Ibov** via B3/Bloomberg
  (usei estimativa 2021-05-03).
- **Confirmar `entrada` de BBSE3 no Ibov** (assumi None; se entrou
  depois de 2019, ajustar).
- **Teste extra `test_capital_final_menor_com_multiplas_posicoes_no_fim`**
  para fortalecer suíte do PROGRAM no caso N>1.
- **Mini-gate Sonnet 4.6 vs Sonnet 5** — descartado, Haiku venceu
  o gate 1.
- **DFP anual vs ITRs no 4º trimestre** (resultado anual sai na DFP).
- **Cobertura completa do IBOV 2025** (TODOs em UNIVERSO_HISTORICO).
- **Sensibilidade take-profit** (5%, 8%, 12%, 15%) como análise
  complementar se sobrar tempo.
- **Expansão do universo** de 24 para 60-80 tickers Ibov, condicional
  a Bloomberg cobrir os novos tickers na janela do backtest.
- **Coleta adicional de notícias na FGV**: só se backtest oficial
  mostrar problema estatístico específico que dados adicionais
  resolveriam. Antes disso não vale ida à FGV.

## Limitações conhecidas (documentadas, não bloqueantes)
- NewsAPI gratuito: 30 dias + retorna 426 em janelas históricas.
  **Não é mais fonte primária.**
- Bloomberg sem API; integração via CSV manual (Terminal FGV).
  **Fonte primária desde 17/07/2026, dataset unificado desde 03/08/2026.**
- `publishedAt` ≠ momento do evento (proxy).
- Ações em circulação na CVM têm gaps em reorganização.
- B3 sem API limpa para composição histórica do IBOV.
- **GDELT com IP-flakiness em janelas específicas** — mesmo com backoff,
  pode ficar inacessível por dias. Bloomberg cobre.
- Paginação GDELT/NewsAPI não implementada.
- Lookahead do LLM: MITIGADO, não eliminado.
- **PEAD clássico não testado** — usamos `crescimento_lucro_yoy` como
  proxy de growth. Consenso Bloomberg (surpresa vs estimado) coletado
  em earnings ERN mas não integrado como feature ainda.
- **yfinance frágil em runtime:** ELET3/AXIA3 pós-privatização, JBSS3
  pós-delistagem. `_prefetch` tolera.
- **BCB SGS rate-limita por IP.** Pré-fetch do MATH&ML elimina o gatilho.
- MATH&ML empata com B1 no mock (GAP=+0.0037). Esperado por construção.
- **Take-profit de +15% dispara raramente** com alvo 5du. Confirmado
  no smoke test do PROGRAM: 0 takes em 9 trades.
- **Calendário B3 ausente no `pd.bdate_range`.** Fix no PROGRAM via
  `pandas_market_calendars`.
- **Cobertura de fontes brasileiras via GDELT é desigual.** Bloomberg
  cobre.
- **Cold-run do PROGRAM ~48 min** por yfinance throttling + falta de
  cache negativo. Warm ~2 min. Determinismo cold==warm validado.
- **Contrato de fronteira tz assimétrico:** ORQUESTRADOR/JOURNAL exigem
  tz-aware SP, PROGRAM opera naive internamente. Adaptador
  `_para_sp`/`_get_precos` gerencia a fronteira.
- **CDI não disponível no `journal.get_macro`.** Metrics usa Selic
  realizada (SGS 11) como proxy de risk-free. Aceito como padrão em
  Fama-French empírico brasileiro.
- **MDD reamostrado (Monte Carlo) é aproximação.** Equity sintética
  sem gaps temporais; distinto do MDD real do backtest. Reportado
  separadamente.
- **Densidade Bloomberg no dataset unificado ~1 notícia/ticker/mês.**
  Estimativa: 20-40 trades no backtest oficial. Aceitável mas apertado —
  IC95 do hit rate vai ficar largo.
- **AXIA3 renomeado para ELET3 no parser via TICKER_ALIAS.**
  Rebrand pós-privatização; universo mantém nome histórico.
- **Cutoff Haiku (jul/2025 fim de mês).** Janela ago-dez/2025 é a
  única genuinamente limpa. Cobertura pra 2024 dentro do treino
  (viés de contaminação possível, aceito como limitação).
## O que é
JEMPO é uma estratégia quantitativa multi-agente event-driven para ações
brasileiras (Ibovespa). Detecta oportunidades após eventos de notícias,
combinando análise fundamentalista via IA generativa com machine learning
quantitativo. Desenvolvido para o Desafio Quant AI 2026 do Itaú Asset
Management.

## Os 5 agentes
- **JOURNAL** (implementado, 70+ testes): provedor de dados puro. Coleta e
  organiza notícias, preços, fundamentos e macro. Fontes: **Bloomberg CSV
  (primária)**, GDELT (suplementar, com backoff blindado), NewsAPI
  (fallback); yfinance; CVM; BCB SGS com fallback FRED.
- **ECON** (implementado, 36+ testes; validado com API real em 17/07/2026):
  analista fundamentalista qualitativo via Claude API. Devolve ScoreEcon.
  Método principal: `avaliar(ticker, data_limite)`.
  API key configurada e funcional. Custo ~US$0.0045/chamada (Haiku 4.5).
  Calibração real pendente de dados Bloomberg + resolução do gate de custo.
- **MATH&ML** (implementado e formalmente fechado, 35 testes): prevê
  retorno idiossincrático 5du à frente via GradientBoosting.
  Arquitetura em duas fases (pré-fetch + montagem em memória). Regra
  de platô com fallback para argmax. sample_weight=5x nos eventos
  como default. Run oficial de sensibilidade completo com relatório
  versionado (commits b7457e3 + 399145c). Mock estruturado permite
  treinar antes da API key do ECON. Contrato de `prever_universo`
  expandido para 8 colunas — consumido pelo ORQUESTRADOR.
- **ORQUESTRADOR** (implementado, 56 testes): coordena os 3 agentes
  anteriores, toma decisão final e aplica gestão de risco. Agente
  central. Contrato público: `decidir(data, equity_hoje) → DecisaoDia`,
  `notificar_execucao`, `notificar_fechamento`, `status`. Não chama
  JOURNAL diretamente. Implementado via Claude Code em 7 etapas com
  approval gate por etapa.
- **PROGRAM** (Etapas 1, 2, 3 fechadas — 98 testes): motor de backtest
  event-driven com Monte Carlo e métricas de performance. Etapas 4-5
  pendentes (visualizações/notebook, integração ECON real).

## Estado da suíte de testes
- **Total: 342 testes verdes** (após Etapa 2 do PROGRAM).
- Zero regressão em nenhuma etapa desde o início.

## Periodização do sistema
- Warmup: desde 2019 (features de momentum usam 252 dias úteis).
- **Calibração ECON: ago-dez/2025** (janela limpa pós-training cutoff Haiku
  em 2025-07-31; jul/2025 removido por estar dentro do treino).
  Suplementar opcional: 2020-2021 (teto otimista, dentro de treino).
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

### Sizing — EQUAL WEIGHT 15%
Cada posição recebe 15% do capital corrente (`equity_hoje`). Não
pondera por força do sinal.

### Limites de posição
- Máximo **3 posições simultâneas**.
- Máximo **2 posições no mesmo setor**.

### Regras de saída (primeiro critério atingido)
Prioridade: **stop > take > prazo > reversão**.
- Stop loss: preço ≤ 0.92 × preço_entrada.
- Take profit: preço ≥ 1.15 × preço_entrada.
- Prazo: 5 dias úteis (fecha na abertura do 6º).
- Reversão: `ECON.avaliar(ticker, data_limite=data).score_total < -0.30`.

Nota de honestidade metodológica: com alvo de 5du, take-profit de +15%
dispara raramente — saída dominante é por prazo. **Confirmado empiricamente
no smoke test do PROGRAM: 0 takes em 9 trades (mar-jun/2024)**. Declarado
abertamente no relatório final.

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
- Regra vale por-ticker.
- Pula fins de semana ao somar dias úteis.

### Acoplamento com o PROGRAM
- **PROGRAM injeta `equity_hoje` = MTM ao fim de D-1** a cada
  `decidir(data, equity_hoje)`. Convenção travada: usa preços de
  fechamento de D-1, não de D. Anti-lookahead + evita reflexividade
  do circuit-breaker.
- **PROGRAM tem última palavra sobre fechamento** — validado por
  invariante testada: se stop/take intraday dispara no dia D, o
  ORQUESTRADOR não deve ter o ticker em `_posicoes` no momento de
  `_verificar_fechamentos` (senão o engine levanta `RuntimeError`).

### Custos no backtest (aplicados pelo PROGRAM)
0.3% corretagem + 0.1% slippage por operação = **0.4% por perna, 0.8% round-trip**.

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
- **Contratos `get_precos`, `get_noticias`, etc. exigem tz-aware SP
  em datas de fronteira** (`_validate_aware`). Consumidores precisam
  respeitar (ver adaptador do PROGRAM abaixo).

### JOURNAL como provedor de dados puro
- Coleta e entrega dados brutos. Não pontua, não pondera.
- `get_retornos_setor` agrega retornos setoriais (dado bruto). ECON faz
  a pontuação.

### Arquitetura modular das fontes
- Cada fonte externa em `agents/sources/` como classe própria:
  `CVMSource`, `GDELTSource`, `NewsAPISource`. Dataclass `Noticia`
  compartilhada em `noticia.py`.
- `JournalAgent` orquestra; parsing pesado nas classes específicas.

### Coleta de notícias — cascata com Bloomberg como primária
- **Cascata em `get_noticias`: Bloomberg CSV (1.0) → GDELT (suplementar) →
  NewsAPI (fallback).**
- **Bloomberg promovida a fonte primária** (17/07/2026) após descoberta
  de que GDELT e NewsAPI são insuficientes para janela ago-dez/2025:
  GDELT com IP penalizado (retries 65s → 138s → connection reset),
  NewsAPI free tier retorna 426 (Upgrade Required) em janelas históricas.
- Bloomberg é curado via Bloomberg Terminal na biblioteca da FGV,
  exportado para `data/bloomberg/noticias.csv`. Sem rate limit, sem
  dependência de API externa em runtime. Determinístico.
- Whitelist rígida: bloomberg.com, reuters.com, valor.globo.com,
  valor.com.br, broadcast.com.br, estadao.com.br, infomoney.com.br.
- Pesos: Bloomberg 1.0, Reuters/Valor 0.95, Broadcast 0.90, Estadão
  0.85, InfoMoney 0.75.
- `TICKER_PARA_NOME` em `config.py` resolve ticker→nome antes da busca.
- Cache por fonte em pickle, TTL 24h.

### Janelas de Bloomberg necessárias (backlog)
- **ago-dez/2025** — para gate de custo e calibração real do ECON.
- **2024-01-01 a 2025-12-31** — para backtest oficial.
- Bloomberg CSV pré-Bloomberg era vazio (só `.gitkeep`). Popular antes
  de qualquer nova execução do gate/calibração/backtest.

### Deduplicação fuzzy entre fontes
- Similaridade de títulos via `difflib.SequenceMatcher` > 0.85 E
  diferença de publicação < 24h.
- `_DEDUP_SIM_MIN = 0.85`, `_DEDUP_HORAS_MAX = 24`.
- Processadas em ordem decrescente de peso; duplicatas de menor peso
  descartadas.

### GDELT blindado contra rate limit (Round 4)
- `GDELTRateLimitedError` e `GDELTUnavailableError` como exceções
  tipadas em `agents/sources/gdelt.py`.
- Backoff exponencial: 60s → 120s → 240s → 480s → 600s, 5 tentativas.
- Env var `GDELT_THROTTLE_SECONDS` (default 5). Rodadas oficiais usam 12s.
- Captura no `JournalAgent` com `gdelt_degradado_count` no
  `health_check()`.
- **Achado 17/07/2026:** mesmo com backoff, GDELT pode ficar
  inacessível por dias em janelas específicas devido a penalização de
  IP. Não é bug do backoff — é limitação estrutural da fonte gratuita.
  Bloomberg cobre essa lacuna.

### Cache negativo — TODO aberto
- **Problema descoberto no smoke test do PROGRAM (48 min cold run):**
  tickers com falha permanente no yfinance (JBSS3 pós-delistagem em
  06/06/2025, ELET3 flaky) são re-tentados a cada run porque falhas
  não entram no disk-cache. Cria overhead significativo.
- **Solução proposta (não implementada):** cache negativo com TTL longo
  (~7 dias) para respostas 404/vazias do yfinance. Antes do backtest
  oficial de produção.

### Fonte de fundamentos: CVM
- `dados.cvm.gov.br/dados/CIA_ABERTA/DOC/`. Módulo em `cvm.py`.
- ITR trimestral + DFP anual como ZIPs, cache em `data/cvm/`.
- Encoding latin-1, sep ";", decimal ",", valores em milhares de R$.
- Sempre consolidado (`_con_`), nunca individual.
- Versões reapresentadas: filtra máxima por (CNPJ_CIA, DT_REFER).
- Bancos usam BPB (não BPA+BPP) — código ramifica por setor.
- yfinance é fallback APENAS para setor.
- Fluxo TTM: DFP → TTM=anual; ITR → TTM=ULTIMO_YTD + (DFP_ano_ant −
  PENULTIMO_YTD).
- Estoque/balanço: point-in-time.
- `Fundamentals.data_recebimento_cvm` para rastreabilidade.

### Survivorship bias com membership por data
- `UNIVERSO_HISTORICO` cobre 2019-2025, com entrada/saida por ticker.
- Campos: setor, entrada, saida, confianca, fonte, cd_cvm, cnpj.
- `tickers_ativos(data_aware)` usada em TODOS os loops sobre universo.
- Casos emblemáticos: AMER3 (saída 12/jan/2023), IRBR3 (saída 2020),
  JBSS3 (saída 06/06/2025).

### Fontes em camadas (resumo)
- **Notícias: Bloomberg CSV (primária, 1.0) > GDELT (suplementar,
  whitelist, backoff, IP-flaky) > NewsAPI (fallback, 30d gratuito).**
- Macro: BCB SGS primário, FRED fallback automático.
- Preços: yfinance com duas versões (ajustada e bruta).
- Fundamentos: CVM (primária), yfinance só para setor.

## DECISÕES CONSOLIDADAS DO ECON

### Função e contrato
- Analista fundamentalista qualitativo via Claude Haiku 4.5
  (`claude-haiku-4-5-20251001`). **Modelo final pendente do gate de
  custo Haiku vs Sonnet 4.6 (bloqueado por dados Bloomberg).**
- Recebe dossiê do JOURNAL (notícia + fundamentos CVM + macro +
  setoriais) e devolve `ScoreEcon`.
- **Método principal:** `avaliar(ticker, data_limite)`.
  `data_limite` deve ser passado explícito (kwarg) por consumidores.

### API key validada (17/07/2026)
- Chave configurada no `.env` (fora do git via `.gitignore`).
- Teste de sanidade em 1 chamada real (PETR4, data_limite 2025-10-15):
  - `ScoreEcon` retornado com shape esperado.
  - `score_total=-0.25`, `tem_evento=True`, `n_noticias=6`,
    `confianca=0.65`, `modelo=claude-haiku-4-5-20251001`.
  - Justificativa coerente (Ibama/Foz do Amazonas + queda setorial
    vs P/L baixo).
  - Custo real: US$0.0045/chamada (2917 tokens input, 311 output).
  - Latência dominada por coleta de notícias (148s), não LLM (~5s).
    GDELT com backoff de 67s explicou o wall-clock.

### ScoreEcon — campos principais
- `score_total` [-1, +1]: IMPACTO DA NOTÍCIA no excesso ao Ibov em 5d.
  Opção A — NÃO combina saúde financeira / setor / macro.
- `comp_noticia`: base do `score_total`.
- `comp_saude_financeira`, `comp_setorial`, `comp_macro`: CONTEXTO
  considerado — NÃO somados ao total.
- `confianca` [0, 1], `tem_evento`, `n_noticias`.
- `noticias_hashes`: rastreabilidade.
- `data_noticia_mais_recente`: tz-aware `America/Sao_Paulo` ou NaT.
- `justificativa`, `modelo`, `avisos`.

### Decisão "Opção A" sobre o score
- ECON pontua o MECANISMO da notícia; MATH&ML otimiza pesos.
- Reduz colinearidade EXPLÍCITA com features cruas que MATH&ML recebe
  do JOURNAL.

### Integração com MATH&ML
- `score_total` entra como feature principal.
- Saúde/setor/macro entram no MATH&ML como features CRUAS do JOURNAL,
  NÃO `comp_*` do ECON.

### Arquitetura técnica
- Tool use forçado + temperature=0. Reprodutibilidade vem do CACHE
  VERSIONADO (`_PROMPT_VERSION` na chave), não do temperature.
- Event-driven: sem notícia → `ScoreEcon` neutro sem chamar Claude.
- Degradação graciosa: nunca levanta exceção; devolve neutro + aviso;
  NÃO cacheia falha.

### Anti-lookahead do LLM (3 defesas)
- Cutoffs do Haiku 4.5 (fim de mês como fronteira conservadora):
  - Reliable knowledge: fev/2025
  - Training data: **jul/2025** (inclusive — jul entra em treino)
- **Janela genuinamente LIMPA = ago-dez/2025** (estritamente pós-training).
  Correção 17/07/2026: era jul-dez, mas jul/2025 é dentro do treino
  do Haiku por fim-de-mês conservador.
- Defesa 1 — IC segmentado: fronteira no TRAINING cutoff.
- Defesa 2 — Placebo com dois modos: `swap` e `identidade_pura`.
- Defesa 3 — Auditoria regex de justificativas.

### Calibração (estrutura pronta, execução bloqueada por Bloomberg)
- Métrica primária: IC Spearman, meta > 0.15 com IC95 bootstrap não
  cruzando zero.
- Baseline lexical + GAP para justificar custo do LLM.
- Loop de prompt ≤ 10 iterações. Cada iteração bumpa `_PROMPT_VERSION`.
- **Bloqueio operacional:** requer dados de notícia em janela
  ago-dez/2025 (limpa) + 2020-2021 (dentro treino, opcional).
  Bloomberg é a única fonte viável nessa janela.

### Gate de custo Haiku vs Sonnet — script pronto, execução bloqueada
- **Script:** `calibration/gate_custo_haiku_vs_sonnet.py` implementado
  e testado (5 testes verdes: block bootstrap por data, ΔIC pareado,
  critério de decisão).
- **Parâmetros travados:**
  - N=100 eventos, janela **ago-dez/2025** (limpa).
  - Modelos: `claude-haiku-4-5-20251001` vs `claude-sonnet-4-6`.
  - Critério: ΔIC > 0.05 → Sonnet; < 0.03 → Haiku; zona cinza → Haiku.
  - Block bootstrap por data (blocos 5du, 10k iterações). NÃO i.i.d.
  - N_min = 40 (piso; abaixo disso aborta).
  - Hard cap US$8 (dobro da estimativa US$1.79).
- **Bloqueio:** primeira execução (17/07/2026) travou na Etapa A por
  falta de fonte de notícia em ago-dez/2025 (Bloomberg vazio, GDELT
  IP-penalizado, NewsAPI 426). 5 horas de wall-clock, ~1 evento
  coletado. US$0 gasto. **Requer Bloomberg populado para destravar.**

### Baseline da calibração (Etapa 1 — EXECUTADO em 06/08/2026)
- **Script:** `calibration/baseline_econ.py`. Prompt `2026-06-econA` medido
  SEM alteração. Relatório: `results/RELATORIO_CALIBRACAO_ECON.md`.
- **Amostra:** dataset Bloomberg unificado, 2024-2025, dedup por conjunto
  de notícias → **636 eventos, todos avaliados** (22 tickers).
- **Custo real: US$ 2,7937** (hard cap US$ 3,50). Latência LLM mediana
  4,01s / P95 5,66s. Degradação final 0%.
- **RESULTADO (a régua da Etapa 2):**
  - IC completo (2024-2025, n=636): **+0,0581** [-0,0291, +0,1442] —
    cruza zero.
  - IC limpo (notícia ≥ 2025-08-01, n=281): **+0,0137** [-0,1249, +0,1406]
    — cruza zero.
  - IC lexical B0: +0,0293 (completo) / +0,0382 (limpo).
  - **GAP: +0,0288 (completo) e −0,0245 (LIMPO — ECON PERDE do
    dicionário na janela sem contaminação).**
  - Veredito: PRECISA DE ITERAÇÃO SIGNIFICATIVA nas duas janelas.
- **Padrão dominante do erro:** os piores casos são resultado trimestral
  em que o ECON lê a direção fundamental CORRETA e o mercado já
  precificou (LREN3 recorrente). Eixo #1 da Etapa 2.

### Armadilha descoberta na Etapa 1 — degradação silenciosa
- **O EconAgent degrada graciosamente por contrato** (neutro + aviso, sem
  exceção). O retry da calibração só capturava EXCEÇÃO → 119 de 610
  avaliações (19,5%) nunca chegaram ao LLM e entraram no dataset como
  score 0,0 legítimo, acima do limiar de invalidez de 15%.
- Detecção: `tokens_in == 0`. Correções: `avaliar_com_retry(...,
  degradou=...)` aceita predicado; degradadas são excluídas das métricas;
  o relatório declara a taxa com veredito de validade.
- Os 119 foram reavaliados (US$ 0,4936) → taxa final 0%.
- **Regra para a Etapa 2:** sempre conferir a taxa de degradação ANTES de
  comparar IC entre iterações. Um IC pode cair só porque a API piorou.

### Rate-limit de macro na calibração — resolvido com pré-fetch
- `get_macro` cacheia por DATA: centenas de eventos = centenas de buscas
  da série inteira no BCB SGS → rate-limit por IP, rodada travada ~8h.
- `exec_infra.prefetch_macro` busca uma vez e serve fatias por
  `data_limite` (anti-lookahead testado). Mesmo padrão do `_prefetch` do
  MATH&ML.
- yfinance também entrou em throttling na retomada. Por isso
  `baseline_econ.py --finalizar` fecha o relatório a partir do
  checkpoint, sem rede — re-amostrar num dia ruim devolve outro conjunto
  de eventos e desalinha o `evento_id` (que é posicional; há guarda de
  identidade ticker+data na retomada).

### Divergência metodológica — RECONCILIADA (Etapa 1 da calibração)
- **Era:** `econ_calibration.py::comparar_modelos()` usava bootstrap
  i.i.d. (reamostra linhas); o gate de custo usava block bootstrap por
  data. Retornos de datas próximas são correlacionados, então i.i.d.
  subestima a largura do IC95.
- **Agora:** `adicionar_blocos` e `bootstrap_ic_bloco` são canônicos em
  `econ_calibration.py` (camada estatística compartilhada);
  `gate_custo_haiku_vs_sonnet.py` importa e reexporta de lá — uma única
  implementação, sem cópia. `comparar_modelos()` migrado: reporta
  `n_blocos` junto do IC95.
- Parâmetros travados: blocos de **5 dias úteis**, **10.000** iterações,
  **seed=42** — os mesmos do gate, agora em constantes únicas
  (`BLOCO_DIAS_UTEIS`, `N_BOOTSTRAP`, `SEED_BOOTSTRAP`).
- Coberto por `tests/test_econ_calibration.py::TestBlockBootstrap`,
  incluindo o teste que mostra o IC95 por bloco mais largo que o i.i.d.
  em dados com correlação intra-bloco.

### Infra de execução paga extraída (`calibration/exec_infra.py`)
- Captura de custo/latência via `usage` do SDK, checkpoint incremental,
  retry com backoff e calendário de pregões saíram do gate para um
  módulo próprio, reusado pela calibração. O gate reexporta os nomes
  antigos — `tests/test_gate_custo.py` segue verde sem alteração.

### Baseline lexical B0 — léxico bilíngue
- O dataset Bloomberg é majoritariamente em INGLÊS: o léxico só-PT
  tocava 12,9% dos títulos, o que deixaria o B0 quase todo zero e o GAP
  sem significado (vitória de régua torta).
- `_PALAVRAS_POS/_NEG` passaram a incluir termos EN de manchete
  (beats/misses/drops/upgrade/...): cobertura foi a **37,1%**.
- `baseline_sentimento_simples(..., apenas_titulo=True)` é a forma usada
  como B0: o corpo Bloomberg traz o artigo inteiro, com comentário de
  analista dos dois lados, e afogaria o sinal da manchete.

### Próximos passos do ECON (em ordem)
1. ~~Smoke test com API key~~ ✓ (17/07/2026).
2. **Popular Bloomberg CSV para ago-dez/2025 (via Terminal FGV).**
3. Executar gate de custo (script pronto, ~15 min sem GDELT rate-limit).
4. Amostra de eventos + `calibrar` completo.
5. Loop de prompt engineering ≤ 10 iterações.
6. Congelar prompt + `RELATORIO_CALIBRACAO_ECON.md` final.
7. **Mini-gate opcional Sonnet 4.6 vs Sonnet 5** — apenas se gate 1
   escolher Sonnet. Mesmo preço nominal, decisão sobre qualidade.

## DECISÕES CONSOLIDADAS DO MATH&ML

### Filosofia central: hipótese antes de padrão
- Cada feature mapeia 1-para-1 a hipótese testável da literatura, com
  sinal esperado explícito.
- `importancia_features()` cruza ganho observado com sinal teórico;
  direção invertida → red flag reportado.

### Regra 1: target = retorno beta-ajustado 5du
- `y_i(t) = r_i_fwd − beta_i(t) × r_ibov_fwd`, sobre `Close_raw`
  (split-only) para ação e Ibov.
- Beta por `np.polyfit(r_ibov, r_i, deg=1)` sobre 252du terminando em t.
- Fallback `beta=1.0` se janela tiver < 200 observações.

### Regra 2: fundamentos por DT_RECEB
- `journal.get_fundamentals(ticker, data_limite)` já corta por
  `data_recebimento_cvm <= data_limite`.

### Regra 3: sinal = ranking cross-sectional
- `MathMLAgent` entrega **previsão contínua + ranking**; seleção
  final é do ORQUESTRADOR.

### Regra 4: linhas de treino = todos os dias-ticker do universo ativo
- Em dias sem evento, `score_econ=0`.
- **Sample weight de 5.0 nos dias de evento como default**.
- IC reportado em DUAS agregações: total e subset `tem_evento=True`.

### Regra 5: períodos
- Warmup desde 2019-01-01.
- Treino 2020-2023. Backtest OOS 2024-2025.
- Walk-forward com janela expansiva, `freq='MS'` mensal default.

### Modelo: GradientBoosting raso de propósito
- `GradientBoostingRegressor(max_depth=3, learning_rate=0.05,
  subsample=0.8, random_state=42)`.
- `n_estimators` por regra de platô **com fallback para argmax**.
- `cv_report` inclui `n_platau`, `n_argmax`, `n_escolhido`, `fonte`.

### Vetor de features (17 features + 3 flags de auditoria)

**Momentum e reversão:**
- `mom_12_1` (+), `rev_1m` (−).

**Fluxo de resultados (growth, não PEAD):**
- `dias_desde_resultado`, `crescimento_lucro_yoy` (+).

**Qualidade/valor:**
- `pl` (−), `pvp` (−), `roe` (+), `margem` (+), `divida_ebitda` (−).

**Volume:**
- `volume_relativo` (+). Duas semânticas: modelo consome imputado
  (mediana cross-sectional); `prever_universo` retorna cru (NaN
  preservado).

**Estado macro:**
- `selic_nivel`, `selic_var_21d`, `cambio_var_21d`.

**Notícia:**
- `score_econ` (+), `econ_confianca`, `econ_n_noticias`.

**Flags de auditoria:**
- `beta_fallback`, `fundamental_imputado`, `econ_degradado`.

### Regras anti-lookahead (defesa em profundidade)
1. Toda feature em t usa só dados `<= t`.
2. Label usa `t+1..t+5` (forward — regras 3 e 4 obrigatórias).
3. Drop de label incompleto ANTES de `_montar_features`.
4. Purge + embargo López de Prado: `PurgedTimeSeriesSplit`, embargo 5du.
5. Imputação cross-sectional do dia; fallback global só do subset de treino.
6. Walk-forward respeita o mesmo embargo.
7. `_assert_no_lookahead` levanta `LookaheadError` em 4 fontes.

### Arquitetura de pré-fetch (Round 5)
- **Bug medido:** padrão anterior chamava JOURNAL ~10k vezes por run.
- **Solução em duas fases:**
  1. `_prefetch`: cada fonte chamada UMA vez cobrindo range inteiro.
  2. `_montar_features` e `_alvo` lêem SÓ do cache — zero I/O na montagem.
- `_prefetch` tolera tickers sem dados no yfinance.
- Cold-cache 24 tickers × 6 anos ~90 min; reruns ~5s via disk-cache.

### Ajustes do run oficial (Round 7)
- Fix 1: Fallback do platô (n_platau < 0.3 × n_argmax → usa argmax).
- Fix 2: sample_weight_eventos=5.0 default.
- Fix 3: Snapshot do cv_report antes do walk_forward.

### Mock estruturado do ECON
- Sinal controlado: `α × z(y) + ruído`.
- Quatro modos: ruído (0.00), fraco (0.10), meta (0.15), forte (0.20).
- **Achado do smoke test end-to-end:** `set_cache(None)` após treino
  para evitar envenenar avaliações no serve-time.

### Avaliação e métricas
- IC de Spearman total e no subset `tem_evento=True`.
- IC95 por block bootstrap por data.
- Três baselines: B1 (score_econ), B2 (mom_12_1), B3 (intercepto).
- GAP = IC_modelo − max(B1, B2, B3).

### Contrato de saída do `prever_universo` (8 colunas)

| Coluna | Tipo | Semântica |
|---|---|---|
| `ticker` | str | Ex.: "PETR4.SA" |
| `y_pred` | float | Retorno idiossincrático 5du previsto |
| `score_econ` | float ∈ [-1, +1] | Score da notícia |
| `tem_evento` | bool | Se houve notícia relevante |
| `rank` | int | Ranking cross-sectional (1 = melhor) |
| `volume_relativo` | float | Cru, pré-imputação, NaN preservado |
| `data_noticia_mais_recente` | datetime tz-aware SP ou NaT | Do ScoreEcon |
| `setor` | str | Do `UNIVERSO_HISTORICO` |

### Resultados observados — Run oficial de sensibilidade
(24 tickers, 2020-2023 + 2024-2025, 4 modos, 90.8 min; commit 399145c).

| Modo | IC_alvo | n_platau→n_est | IC_evento | GAP vs B1 |
|---|---|---|---|---|
| ruído | 0.00 | 1→4 (fallback) | +0.029 | +0.029 |
| fraco | 0.10 | 1→4 (fallback) | +0.095 | +0.001 |
| **meta** | **0.15** | **1→4 (fallback)** | **+0.147** | **+0.0037** |
| forte | 0.20 | 3→17 (fallback) | +0.189 | −0.0004 |

**Importância no modo meta:** `score_econ` #1 (0.256), `mom_12_1` #2
(0.236), `rev_1m` #3. Zero features com sinal invertido.

### Determinismo e reprodutibilidade
- `PYTHONHASHSEED` ∈ {0, 1, 12345}.
- `_stable_seed(ticker, t)` para RNG do mock.
## DECISÕES CONSOLIDADAS DO ORQUESTRADOR

### Estado da implementação
- 564 linhas em `agents/orchestrator.py`.
- 771 linhas em `tests/test_orchestrator.py`.
- 56 testes determinísticos + 3 de integração.

### Contrato público
```python
class OrchestratorAgent:
    def __init__(self, journal, econ, math_ml, config, tickers_ativos=None)
    def decidir(self, data, equity_hoje) -> DecisaoDia
    def notificar_execucao(self, ticker, setor, preco_execucao, data_execucao) -> None
    def notificar_fechamento(self, ticker, data_fechamento) -> None  # agnóstico ao motivo
    def status(self) -> dict
```

Dataclasses: `OrchestratorConfig` (frozen), `Ordem`, `FechamentoOrdem`
(motivo Literal["prazo", "reversao"]), `DecisaoDia`.

### Métodos privados
- `_atualizar_pausa(data, equity_hoje) → float`: drawdown + circuit-breaker.
- `_selecionar_ordens(df, data)`: pool + top-N + limite setorial.
- `_resolver_data_execucao(row, data)`: D+1 vs D+2 com corte 17h05.
- `_verificar_fechamentos(data)`: prazo antes de reversão.

### Padrões-chave preservados
- `tickers_ativos` injetável para testes determinísticos.
- Anti-lookahead: `data_limite=data` sempre explícito nas chamadas ao ECON.
- Não chama JOURNAL diretamente.
- `_pausado_ate` não é limpo ao expirar — "estar pausado" é derivado.
- **Todas as datas de fronteira exigem tz-aware SP** (`_exigir_aware`).

## DECISÕES CONSOLIDADAS DO PROGRAM (Etapas 1, 2, 3 fechadas)

### Estado da implementação
- **Etapa 1 completa** (BacktestEngine core), commits 03d9799 (1.1),
  84429c6 (1.2), e869767 (1.3). ~560 linhas em `backtest/engine.py`.
- **Etapa 2 completa** (Monte Carlo), commit da Etapa 2 (bootstrap +
  permutação com determinismo). ~350 linhas em `backtest/monte_carlo.py`.
- **Etapa 3 completa** (métricas de performance + atribuição). ~460 linhas
  em `backtest/metrics.py`.
- **Total: 98 testes determinísticos** do PROGRAM (40 engine + 33 metrics
  + 25 monte carlo).
- **Smoke test de integração** com 4 agentes reais (JOURNAL/MATH&ML/
  ORQUESTRADOR + mock estruturado do ECON) executando janela mar-jun/2024,
  83 dias úteis BMF, exit 0.

### Contrato público (Etapa 1 — engine)
```python
@dataclass(frozen=True)
class BacktestConfig:
    capital_inicial: float = 100_000.0
    corretagem: float = 0.003
    slippage: float = 0.001
    stop_pct: float = 0.08
    take_pct: float = 0.15
    sizing_pct: float = 0.15

@dataclass(frozen=True)
class ResultadoBacktest:
    trades: pd.DataFrame
    equity_diario: pd.Series
    avisos: list[dict]
    config: BacktestConfig
    data_inicio: pd.Timestamp
    data_fim: pd.Timestamp
    n_dias_uteis: int
    n_trades: int
    capital_final: float

class BacktestEngine:
    def __init__(self, journal, orquestrador, config: Optional[BacktestConfig] = None)
    def rodar_backtest(self, data_inicio, data_fim) -> ResultadoBacktest
```

### Dataclasses internas (engine)
- `PosicaoInterna` (mutável): estado de posição aberta com stop/take,
  custo_entrada, snapshots de y_pred, score_econ, rank.
- `TradeRegistro` (frozen): registro completo do trade fechado com
  `motivo: Literal["stop", "take", "prazo", "reversao", "fim_backtest"]`.

### Fluxo diário (Ordem B — travada)
```
Passo 0: equity_hoje = MTM ao fim de D-1
Passo 1: detectar stop/take intraday (Low/High do dia D)
         - Prioridade STOP > TAKE
         - preço = stop_price ou take_price exato (sem slippage extra)
         - Chama orq.notificar_fechamento(ticker, D)
Passo 2: dec = orq.decidir(D, equity_hoje)  ← equity é do fim de D-1
Passo 3: Executar dec.fechamentos (Open[D+1] ou próximo pregão)
         - INVARIANTE: nenhum ticker aqui pode ter sido fechado no Passo 1
           (RuntimeError se violar)
Passo 4: Executar dec.novas_ordens (Open[data_execucao])
         - qtd = int((0.15 × equity_hoje) / preco_entrada)
         - Custo entrada 0.4%, degradação graciosa (aviso)
         - Chama orq.notificar_execucao
Passo 5: Registra equity_fim_D (mark-to-market com Close[D])
```

### Adaptador tz na fronteira (decisão travada)
- **Interior do engine 100% naive.** Datas dentro do PROGRAM são
  `pd.Timestamp` sem timezone.
- **Fronteira com ORQUESTRADOR e JOURNAL:** ambos exigem tz-aware SP
  (`_exigir_aware`, `_validate_aware`).
- **Solução:**
  - `_para_sp(data)`: função pura, idempotente, converte naive → SP
    preservando hora de parede (`tz_localize`, não `tz_convert`).
  - `_get_precos(ticker, data_inicio, data_limite)`: choke-point único
    de acesso ao JOURNAL. Localiza limites em SP na entrada, captura
    `DadoIndisponivel`/`LookaheadError` → DataFrame vazio, normaliza
    índice a naive na saída.
- **Regra dura:** nenhum outro método chama `self._journal.get_precos`
  diretamente; nenhuma chamada ao ORQUESTRADOR sem `_para_sp(data)`
  inline.
- Fakes estritos em `tests/fakes.py` replicam `_exigir_aware`/
  `_validate_aware` — rejeitam naive. Se algum call-site esquecer o
  adaptador, teste quebra em vez de passar em "naive-land".

### Calendário B3
- `pandas_market_calendars` com calendário `'BMF'`.
- NÃO usar `pd.bdate_range` (ignora feriados nacionais).
- Validado: 2024-03-29 (Sexta-feira Santa) e 2024-05-30 (Corpus Christi)
  ausentes do calendário retornado.
- `_calendario_bmf` retorna DatetimeIndex naive (após `tz_localize(None)`).

### Tensão R8×R1 — comportamento aceito, documentado, testado
Duas regras corretas geram diferença numérica esperada:
- **R1** (Passo 5): MTM ao fim de D é contabilidade sem custo, marca a
  mercado pelo Close.
- **R8** (fim do backtest): se há posição aberta em data_fim, força
  fechamento pelo Close[data_fim] com custo de saída de 0.4% aplicado.

**Consequência:** quando há posições abertas no último dia,
`capital_final ≤ equity_diario.iloc[-1]` — a diferença é a soma dos
custos de saída forçada. Igualdade só quando não há posições abertas.

**Por que essa política em vez de "sempre igual":** honestidade
metodológica. Não cobrar custo na liquidação forçada maquiaria o
resultado. Banca de asset management pega.

**Testes dedicados:** `test_capital_final_menor_que_equity_final_quando_
fim_backtest_dispara` valida a diferença exata (equity[-1]=100690,
capital_final=100627, Δ=63 = custo de saída).

### Determinismo em produção (evidência de qualidade)
- Smoke test cold vs warm: byte-a-byte idênticos (mesmos 9 trades,
  mesma decomposição, mesmo `capital_final = R$ 96.396,32`).
- Corta suspeita de cherry-picking na banca pela raiz.
- Fundamentos: `PYTHONHASHSEED` fixado, `_stable_seed` no mock ECON,
  iteração alfabética sobre dicionários, adaptador tz determinístico,
  monkeypatch de TTL do JOURNAL na janela histórica.

### Fallback do `_close_marcacao` (documentado + testado)
- Se `Close[D]` não existe (ticker sem barra no dia), usa Close mais
  recente ≤ D disponível (carry-forward).
- Documentado em docstring; teste dedicado `test_close_marcacao_
  fallback_para_close_anterior_quando_barra_ausente` (asserta
  carry-forward: `Close[03-05]=107` → equity `100.990`).

### Achados do smoke test do PROGRAM (mar-jun/2024)
1. **9 trades gerados:** 1 stop, 5 prazo, 3 reversão, 0 take, 0 fim_backtest.
   Confirma que prazo é saída dominante (regra R1 do relatório final).
2. **Cold run 48 min vs warm 2 min.** Causa: yfinance throttling + falta
   de cache negativo para tickers com falha permanente (JBSS3, ELET3).
   Loop do engine é rápido; overhead vem do JOURNAL.
3. **Determinismo cold==warm byte-a-byte** validado.
4. **Zero crashes de fronteira tz, invariante de sincronização,
   ou anti-lookahead.** Adaptador funciona em produção.

## PROGRAM Etapa 3 (métricas de performance) — fechada

### Contrato público (metrics)
```python
@dataclass(frozen=True)
class MetricasBacktest:
    # Retorno
    retorno_total: float
    retorno_anualizado: float

    # Risco
    volatilidade_anualizada: float
    sharpe_anualizado: float
    sortino_anualizado: float
    maximum_drawdown: float          # negativo
    duracao_mdd_dias: int
    dias_ate_recuperacao_mdd: Optional[int]

    # Trades
    n_trades_total: int
    n_trades_vencedores: int
    n_trades_perdedores: int
    hit_rate: Optional[float]
    payoff_medio: Optional[float]
    profit_factor: Optional[float]

    # Custos
    custo_total_pago: float
    custo_como_pct_capital_inicial: float

    # Metadados
    taxa_livre_risco_anualizada: float
    fonte_taxa_livre_risco: str  # "cdi" | "selic" | "override" | "zero_por_falha"

def calcular_metricas(
    resultado: ResultadoBacktest,
    journal,
    taxa_livre_risco_override: Optional[float] = None,
) -> MetricasBacktest: ...

def atribuir_por_motivo(resultado: ResultadoBacktest) -> pd.DataFrame: ...
def atribuir_por_setor(resultado: ResultadoBacktest) -> pd.DataFrame: ...
```

### Convenções travadas (metrics)
- **Volatilidade e Sharpe:** std amostral (ddof=1), padrão de asset
  pricing empírico.
- **Sortino:** downside deviation dividida por **N total** de observações
  (Sortino 1994, Nawrocki 1999), não por n_negativos.
  Fórmula: `downside_dev = sqrt(sum(min(r-MAR,0)²) / N)`.
- **Taxa livre de risco:** Selic realizada (SGS 11, `selic_diaria`) via
  `journal.get_macro`. Fallback: override → selic → zero_por_falha.
  CDI não existe no `get_macro`; Selic realizada é proxy aceitável.
- **Percentis:** `np.percentile` default (linear interpolation), convenção
  Fama-French / MATLAB / R.
- **MDD:** duração pico-a-vale + dias até recuperação (None se não
  recuperou).

### Convenções degeneradas (documentadas)
- `n_trades == 0` → hit_rate/payoff/profit_factor = `None`.
- `n_vencedores == 0` (só perdedores) → payoff = 0.0, profit_factor = 0.0.
- `n_perdedores == 0` (só vencedores) → payoff = `+inf`, profit_factor = `+inf`.

### Atribuição de retorno
- `atribuir_por_motivo`: agrupa por `motivo` do TradeRegistro. Colunas:
  n_trades, pnl_liquido_total, pnl_liquido_medio, hit_rate, pct_do_total_pnl.
  Ordenado por pnl_liquido_total descendente.
- `atribuir_por_setor`: idem, agrupado por `setor`.
- Determinismo via mergesort estável + tiebreak alfabético.
- `pct_do_total_pnl` soma 1.0 mesmo com grupos negativos.

### Testes (33 verdes)
Grupos A (retorno/vol, 5), B (Sharpe/Sortino + Ajuste 1, 6),
C (drawdown, 5), D (trades + Ajuste 3, 6), E (atribuição, 5),
F (custos, 3), G (robustez, 3).

## PROGRAM Etapa 2 (Monte Carlo) — fechada

### Contrato público (monte carlo)
```python
@dataclass(frozen=True)
class ResultadoMonteCarlo:
    tecnica: str  # "bootstrap_retornos" | "permutacao_ordem"
    n_simulacoes: int
    seed: int

    # Distribuição de retorno total
    retornos_simulados: np.ndarray
    retorno_p05, retorno_p25, retorno_p50, retorno_p75, retorno_p95: float
    retorno_medio, retorno_desvio: float

    # Distribuição de MDD reamostrado
    mdds_simulados: np.ndarray  # todos negativos ou zero
    mdd_p05, mdd_p25, mdd_p50, mdd_p75, mdd_p95: float

    # Probabilidades condicionais
    prob_retorno_positivo: float
    prob_supera_ibov: Optional[float]  # None se retorno_ibov=None
    prob_mdd_melhor_que_20pct: float

    # Metadados
    retorno_ibov_referencia: Optional[float]

def rodar_bootstrap_retornos_trades(
    resultado: ResultadoBacktest,
    n_simulacoes: int = 10_000,
    seed: int = 12345,
    retorno_ibov: Optional[float] = None,
) -> ResultadoMonteCarlo: ...

def rodar_permutacao_ordem_trades(
    resultado: ResultadoBacktest,
    n_simulacoes: int = 10_000,
    seed: int = 12345,
    retorno_ibov: Optional[float] = None,
) -> ResultadoMonteCarlo: ...
```

### Duas técnicas independentes
- **Bootstrap com reposição:** sorteia N trades da lista realizada
  com reposição, 10k simulações. Testa "distribuição de retornos por
  trade é robusta?".
- **Permutação sem reposição:** reembaralha ordem cronológica sem
  reposição. Testa "risco de sequência é robusto?".
- Ambas retornam distribuições SEPARADAS. Não combinadas.

### Convenções travadas (monte carlo)
- **Vetorização batched:** matriz (B, N) + cumsum + running-max
  vectorizado. 10k simulações em milissegundos, não loop Python.
- **Núcleo `_rodar` compartilhado** entre bootstrap e permutação (DRY).
- **`_mdd_reamostrado` local vetorizado** (não acopla a
  `metrics._drawdown` privado; permite batch).
- **Determinismo:** `np.random.default_rng(seed)` por chamada, sem
  `np.random.seed` global. Dois runs idênticos byte-a-byte.
- **Comparação com Ibov dentro do MC:** `P(JEMPO > Ibov)` via parâmetro
  escalar `retorno_ibov`. Ibov NÃO reamostrado — é benchmark passivo
  fixo do mesmo período.
- **MDD reamostrado documentado como aproximação:** equity sintética
  sem gaps temporais. Distinto do MDD real (que usa equity diária).
  Reportado separadamente nos gráficos.
- **Permutação preserva retorno total** (soma comutativa em float64).
  O que varia é MDD, não retorno. Bootstrap testa retorno; permutação
  testa risco de sequência.

### Testes (25 verdes)
Grupos A (determinismo, 4), B (bootstrap, 5), C (permutação, 4),
D (MDD reamostrado, 4), E (comparação Ibov, 3), F (robustez, 3),
G (contrato, 2).

**Testes críticos com asserts analíticos:**
- Teste 13 (`test_permutacao_retorno_final_igual_em_todas_simulacoes`):
  `retorno_desvio == 0.0` exato via P&L inteiros (soma comutativa em
  float64).
- Teste 15 (`test_mdd_permutacao_mesmo_conjunto_trades_produz_mdds_
  diferentes`): min/max MDD derivados analiticamente enumerando as 24
  permutações de `[+1000,+1000,-500,-500]`. Assert de extremos exatos:
  `min(mdds) = -0.01`, `max(mdds) = -500/101000`.

### O que fica fora do escopo das Etapas 1-3
- **Etapa 4:** Visualizações e notebook de apresentação
  (`backtest/plots.py`, `notebooks/apresentacao_banca.ipynb`).
- **Etapa 5:** Integração com ECON real (após calibração).

## SMOKE TESTS (validação de integração)

### `scripts/smoke_test_e2e.py` — 4 agentes originais
- Valida JOURNAL/ECON mock/MATH&ML/ORQUESTRADOR reais em cadeia.
- mar-abr/2024, 43 dias úteis. Executável, exit 0, ~2 min.
- Não é backtest — equity constante, sem P&L nem custos.
- 5 ordens, 5 fechamentos (3 prazo, 2 reversão). Cache JOURNAL 78% hit.

### `scripts/smoke_test_program.py` — 4 agentes + BacktestEngine
- Valida 4 agentes reais (JOURNAL/MATH&ML/ORQUESTRADOR + mock ECON) +
  BacktestEngine em cadeia.
- mar-jun/2024, 83 dias úteis BMF. Executável, exit 0.
- Cold: 48 min. Warm: 2 min. **Cold==warm byte-a-byte.**
- 9 trades. Reusa `_EconMockAdapter` do smoke_test_e2e sem duplicação.

## Stack técnica
Python, Claude API (Anthropic, key configurada), scikit-learn, pandas,
numpy, yfinance, pyarrow (parquet), requests, matplotlib,
`pandas_market_calendars`. Macro via BCB SGS (sem chave) com FRED
fallback. Fundamentos via CVM aberto. Dev no Mac (M1) via Claude Code.

## Estrutura de arquivos
```
agents/
- journal.py
- econ.py             (EconAgent + ScoreEcon; método principal avaliar)
- math_ml.py          (MathMLAgent com _DatasetCache, _prefetch,
                       make_econ_mock, prever_universo com 8 colunas)
- orchestrator.py     (56 testes; contratos exigem tz-aware SP)
- sources/
  - noticia.py, cvm.py, gdelt.py, newsapi.py
backtest/
- __init__.py         (exporta BacktestEngine, BacktestConfig,
                       ResultadoBacktest, TradeRegistro, PosicaoInterna,
                       MetricasBacktest, calcular_metricas,
                       atribuir_por_motivo, atribuir_por_setor,
                       ResultadoMonteCarlo,
                       rodar_bootstrap_retornos_trades,
                       rodar_permutacao_ordem_trades)
- engine.py           (~560 linhas, Etapa 1 completa)
- metrics.py          (~460 linhas, Etapa 3 completa)
- monte_carlo.py      (~350 linhas, Etapa 2 completa)
- plots.py            (não implementado — Etapa 4)
calibration/
- econ_calibration.py                (bootstrap i.i.d. — TODO migrar)
- gate_custo_haiku_vs_sonnet.py      (script pronto, bloqueado por Bloomberg)
- results/
  - RELATORIO_CALIBRACAO_MATHML.md   (versionado, commit 399145c)
  - RELATORIO_CALIBRACAO_ECON.md     (WIP, gitignored)
  - RELATORIO_GATE_CUSTO.md          (pendente de execução)
data/
- cache/              (pickle, TTL 24h)
- bloomberg/          (CSV curado — POPULAR ANTES DE GATE/CALIBRAÇÃO)
- cvm/raw/, cvm/processed/
tests/
- test_journal.py, test_cvm.py, test_gdelt.py, test_newsapi.py
- test_econ.py, test_econ_calibration.py
- test_math_ml.py     (35 testes)
- test_orchestrator.py (56 testes + 3 de integração)
- test_engine.py      (40 testes, incluindo adaptador tz e fallback)
- test_metrics.py     (33 testes)
- test_monte_carlo.py (25 testes)
- test_gate_custo.py  (5 testes)
- fakes.py            (Fakes estritos com _exigir_aware/_validate_aware)
scripts/
- smoke_test_mathml.py, smoke_test_v2.py
- smoke_test_orquestrador.py       (fluxo 10 dias com Fakes)
- smoke_test_e2e.py                (4 agentes reais, mar-abr/2024)
- smoke_test_program.py            (4 agentes + BacktestEngine, mar-jun/2024)
- debug_dataset.py, diagnostico_gbm.py, sensibilidade_econ.py
```

## Decisões pendentes e ordem prática

### Bloqueio imediato: dados Bloomberg
**Nada de calibração/gate/backtest oficial pode rodar sem popular
`data/bloomberg/noticias.csv` via Bloomberg Terminal na FGV.**

**Janelas necessárias:**
- ago-dez/2025 (para gate de custo + calibração real do ECON).
- 2024-01-01 a 2025-12-31 (para backtest oficial).

### Roadmap ajustado (4 semanas até apresentação)

**Semana 1 (concluída):**
- ✅ Etapa 1 do PROGRAM (engine).
- ✅ Etapa 3 do PROGRAM (métricas).
- ✅ Etapa 2 do PROGRAM (Monte Carlo).
- ✅ Gate de custo (script pronto, execução bloqueada).
- 🔲 Visita à FGV para exportar Bloomberg (pendente).

**Semana 2:**
- 🔲 **Etapa 4 do PROGRAM** (visualizações + notebook estrutural) —
  autônoma, não depende de Bloomberg.
- 🔲 Popular Bloomberg + rodar gate de custo (destravado).
- 🔲 Escolher modelo LLM (Haiku ou Sonnet).

**Semana 3:**
- 🔲 Calibração real do ECON com prompt engineering (≤10 iterações).
- 🔲 Backtest oficial 2024-2025 com ECON real.
- 🔲 Integração do backtest oficial no notebook (Etapa 4 já pronta).

**Semana 4:**
- 🔲 Notebook de apresentação final.
- 🔲 Relatório final.
- 🔲 Revisão + polimento.

### Plano B se calibração real estourar
Se calibração real der ruim (custo, iterações, IC baixo), manter ECON
mock no relatório final, declarar abertamente como limitação, apresentar
sistema com sensibilidade completa em 4 modos (que já existe).

### TODOs abertos (não bloqueiam)
- **Cache negativo no JOURNAL** para tickers com falha permanente no
  yfinance (JBSS3, ELET3). Reduz cold run do backtest oficial.
- **Teste extra `test_capital_final_menor_com_multiplas_posicoes_no_fim`**
  para fortalecer suíte do PROGRAM no caso N>1.
- **Mini-gate Sonnet 4.6 vs Sonnet 5** — condicional a Sonnet ganhar
  o gate 1.
- **DFP anual vs ITRs no 4º trimestre** (resultado anual sai na DFP).
- **Cobertura completa do IBOV 2025** (TODOs em UNIVERSO_HISTORICO).
- **Sensibilidade take-profit** (5%, 8%, 12%, 15%) como análise
  complementar se sobrar tempo na semana 4.
- **Expansão do universo** de 24 para 60-80 tickers Ibov, condicional
  a Bloomberg cobrir os novos tickers na janela do backtest.

## Limitações conhecidas (documentadas, não bloqueantes)
- NewsAPI gratuito: 30 dias + retorna 426 em janelas históricas.
  **Não é mais fonte primária.**
- Bloomberg sem API; integração via CSV manual (Terminal FGV).
  **Promovida a fonte primária em 17/07/2026.**
- `publishedAt` ≠ momento do evento (proxy).
- Ações em circulação na CVM têm gaps em reorganização.
- B3 sem API limpa para composição histórica do IBOV.
- **GDELT com IP-flakiness em janelas específicas** — mesmo com backoff,
  pode ficar inacessível por dias. Bloomberg cobre.
- Paginação GDELT/NewsAPI não implementada.
- Lookahead do LLM: MITIGADO, não eliminado.
- **PEAD clássico não testado** — usamos `crescimento_lucro_yoy` como
  proxy de growth.
- **yfinance frágil em runtime:** ELET3 pós-privatização, JBSS3
  pós-delistagem. `_prefetch` tolera.
- **BCB SGS rate-limita por IP.** Pré-fetch do MATH&ML elimina o gatilho.
- MATH&ML empata com B1 no mock (GAP=+0.0037). Esperado por construção.
- **Take-profit de +15% dispara raramente** com alvo 5du. Confirmado
  no smoke test do PROGRAM: 0 takes em 9 trades.
- **Calendário B3 ausente no `pd.bdate_range`.** Fix no PROGRAM via
  `pandas_market_calendars`.
- **Cobertura de fontes brasileiras via GDELT é desigual.** Bloomberg
  cobre.
- **Cold-run do PROGRAM ~48 min** por yfinance throttling + falta de
  cache negativo. Warm ~2 min. Determinismo cold==warm validado.
- **Contrato de fronteira tz assimétrico:** ORQUESTRADOR/JOURNAL exigem
  tz-aware SP, PROGRAM opera naive internamente. Adaptador
  `_para_sp`/`_get_precos` gerencia a fronteira.
- **CDI não disponível no `journal.get_macro`.** Metrics usa Selic
  realizada (SGS 11) como proxy de risk-free. Aceito como padrão em
  Fama-French empírico brasileiro.
- **MDD reamostrado (Monte Carlo) é aproximação.** Equity sintética
  sem gaps temporais; distinto do MDD real do backtest. Reportado
  separadamente.
- **Universo do smoke test = 24 tickers.** Backtest oficial pode
  expandir para 60-80 (Ibov ativo) se Bloomberg cobrir.