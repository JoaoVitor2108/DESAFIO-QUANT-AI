# MATH&ML — Manual do Desenvolvedor para a Banca

Documento de estudo intensivo do terceiro agente do JEMPO, o **MATH&ML**. Depois
de ler isso, você consegue explicar qualquer linha de `agents/math_ml.py` e
qualquer decisão metodológica para a banca.

Use junto com o código aberto na tela. Cada seção aponta para um pedaço do
código, explica em camadas (o que faz, por que faz, o que rejeitamos, como
defender) e termina com a frase pronta para responder à banca.

Pré-requisitos de leitura: `MANUAL_JOURNAL.md` e `MANUAL_ECON.md`. O MATH&ML é
cliente dos dois — consome os dados crus do JOURNAL (`get_precos`,
`get_fundamentals`, `get_macro`) e o `ScoreEcon` do ECON. Toda a disciplina
anti-lookahead dos *dados* mora no JOURNAL; o MATH&ML herda isso e adiciona a sua
própria camada (anti-lookahead do *aprendizado supervisionado*, §4).

---

## 0. Para que serve o MATH&ML no JEMPO

O JEMPO tem cinco agentes. O JOURNAL é o **research** (dados puros), o ECON é o
**analista fundamentalista** (julgamento qualitativo da notícia). O MATH&ML é o
**quant**: pega o `score_total` do ECON e o combina com **features quantitativas
cruas** do JOURNAL num modelo de aprendizado supervisionado que **prevê o retorno
idiossincrático de 5 dias úteis** de cada ação e **ranqueia o universo**.

Concretamente, `MathMLAgent.prever_universo(tickers, data_limite)` devolve um
DataFrame ordenado por `y_pred` (previsão) decrescente, com um contrato de saída
rico para o **ORQUESTRADOR** (§6): `rank`, `score_econ`, `tem_evento`,
`volume_relativo` (cru), `data_noticia_mais_recente` (timing D+1 vs D+2) e `setor`
por ticker. É o insumo que o ORQUESTRADOR consome para decidir posição (top-N,
limites setoriais, filtros de liquidez, regras de saída). O MATH&ML **não decide
trade** — ele **prevê e ordena**.

Pense no MATH&ML como o quant da mesa: recebe o dossiê do research (JOURNAL) e a
leitura do analista (ECON), e produz um ranking com base histórica. Ele responde
a UMA pergunta: *dado tudo que sei em `t`, quais ações têm o maior retorno
idiossincrático esperado nos próximos 5 pregões?*

**A regra que governa o MATH&ML é: cada feature é uma HIPÓTESE, não um padrão
pescado.** Toda coluna do modelo mapeia 1-para-1 numa hipótese da literatura de
finanças, com uma direção de sinal esperada. Se o modelo aprende o sinal
invertido, isso é um **red flag de overfit** que a auditoria expõe (§8) — não um
"insight". Tudo no código serve a essa disciplina e ao anti-lookahead.

---

## 1. Arquitetura geral

O MATH&ML está inteiro em `agents/math_ml.py` (um módulo só). Estrutura do
arquivo, de cima para baixo:

```
agents/math_ml.py
├── docstring          # NOTA DE DESIGN (consumo do ECON) + princípios + anti-lookahead
├── LookaheadError     # exceção própria (defesa em profundidade na predição)
├── FEATURES           # as 16 features consumidas pelo modelo (§3)
├── FLAGS              # metadados da linha (NÃO são features): beta_fallback, ...
├── SINAL_ESPERADO     # direção teórica de cada feature (auditoria de sinal, §8)
├── @dataclass MathMLConfig     # hiperparâmetros, janelas e períodos oficiais
├── @dataclass _DatasetCache    # cache em memória do pré-fetch (anti n+1, §5)
├── PurgedTimeSeriesSplit       # split temporal com purge + embargo (§6)
├── make_econ_mock + _MockHandle  # mock estruturado do ECON (§9)
└── class MathMLAgent
    ├── _alvo                   # alvo idiossincrático beta-ajustado (§2)
    ├── _montar_features        # as 16 features de uma linha (§3)
    ├── _assert_no_lookahead    # guarda final anti-lookahead (§4)
    ├── construir_dataset       # painel (data, ticker) com pré-fetch (§5)
    ├── _imputar_cross_sectional  # imputação sem futuro (§4)
    ├── treinar / _escolher_n_estimators / _escolher_n_estimators_platau  # modelo (§6)
    ├── avaliar_ic / _calcular_baselines   # IC + IC95 + baselines/GAP (§7)
    ├── prever / prever_universo           # predição e ranking (§0, §6)
    ├── walk_forward                       # backtest com retreino mensal (§7)
    └── importancia_features / _sinal_observado  # auditoria de sinal (§8)
```

Reuso deliberado: o MATH&ML **não reimplementa dados** — importa `tickers_ativos`
e os períodos oficiais de `config.py`, e consome o JOURNAL/ECON por injeção de
dependência (`MathMLAgent(journal=..., econ=...)` ou `econ_mock=...`). Isso torna
a suíte inteira (`tests/test_math_ml.py`, 33 testes) **determinística e offline**:
um `FakeJournal` fixturizado devolve preços/fundamentos/macro sintéticos e o ECON
entra por `make_econ_mock`. Nenhum teste toca rede, API ou a chave Anthropic.

### O modelo escolhido — GradientBoosting raso

O estimador é um `GradientBoostingRegressor` do scikit-learn com `max_depth=3`,
`learning_rate=0.05`, `subsample=0.8` (`MathMLConfig.gb_params`). É uma escolha
**deliberadamente conservadora**: árvores rasas (profundidade 3) capturam
interações não-lineares de baixa ordem sem decorar o ruído; learning rate baixo +
subsample<1 são regularização (stochastic gradient boosting). O número de árvores
NÃO é fixo — é escolhido por validação cruzada com uma regra de platô (§6).

### Como defender na banca

> "O MATH&ML é o agente quant: combina o score de notícia do ECON com features
> quantitativas cruas do JOURNAL num GradientBoosting raso e regularizado, que
> prevê o retorno idiossincrático de 5 dias e ranqueia o universo para o
> ORQUESTRADOR. Cada feature é uma hipótese da literatura com sinal esperado, e
> auditamos se o modelo aprendeu o sinal certo. É testável de ponta a ponta
> offline por injeção de dependência do JOURNAL e do ECON."

---

## 2. O alvo — retorno idiossincrático beta-ajustado de 5 dias (`_alvo`)

### O que faz

Para uma ação em `t`, o alvo `y` é o **retorno em excesso ao Ibovespa ajustado
por beta** nos próximos 5 dias úteis:

```
y = r_ação(t→t+5)  −  β(t) · r_ibov(t→t+5)
```

onde `β(t)` é estimado por OLS (`np.polyfit`, grau 1) dos retornos diários da ação
contra os do Ibov, na janela de **252 dias úteis que termina em `t`** (defasado,
sem lookahead). Se há menos de `min_obs_beta=200` observações ou variância nula do
mercado, cai no fallback `β = 1.0` (a flag `beta_fallback` marca a linha).

### Por que idiossincrático (e não retorno bruto)

Retorno bruto de 5 dias é dominado pelo movimento do **mercado** — se o Ibov sobe
3%, quase toda ação sobe junto, e o modelo aprenderia a prever o mercado, não a
**seleção de ações**. O que a estratégia quer é o **alpha específico da ação**: o
quanto ela anda além do que o beta dela explicaria. Subtrair `β·r_ibov` remove a
exposição sistemática e deixa o retorno idiossincrático — exatamente o que um
long-short de seleção busca capturar.

### Por que OLS com `np.polyfit` (e a lição do ddof)

O beta é a inclinação de uma reta `r_ação = α + β·r_ibov`. Usamos `np.polyfit`
(mínimos quadrados) em vez de `cov/var` manual porque `cov` do numpy usa `ddof=1`
e `var` usa `ddof=0` por padrão — misturá-los introduz um viés de fator
`n/(n−1)` no beta. O `polyfit` resolve o sistema de uma vez, sem essa
inconsistência. (Foi um bug real que a auditoria pegou; ver §11.)

### Anti-lookahead do alvo

O beta usa só dados `≤ t` (janela termina em `t`). O retorno forward usa
`t+1..t+5` — é **legítimo** porque é o que estamos tentando prever (o label). Se
não existir `t+5` (fim da série), `y = NaN` e a linha é **dropada** do dataset
(§4) — nunca treinamos com label incompleto.

### Como defender na banca

> "O alvo é o retorno idiossincrático de 5 dias: retorno da ação menos beta vezes
> retorno do Ibov, com o beta estimado por OLS numa janela de 252 dias úteis que
> termina na data de decisão. Isso isola o alpha específico da ação do movimento
> do mercado — é o que uma estratégia de seleção busca. Usamos polyfit para evitar
> a inconsistência de ddof entre cov e var, e labels sem janela forward completa
> são descartados, nunca imputados."

---

## 3. As 16 features — hipótese antes de padrão (`_montar_features`)

### O princípio

Cada feature mapeia numa **hipótese da literatura de finanças**, com uma direção
de sinal esperada declarada em `SINAL_ESPERADO`. Não jogamos 200 colunas num
modelo e "deixamos ele achar" — cada coluna tem uma tese, e a §8 audita se o
modelo aprendeu a direção certa. Isso é o oposto de data mining.

### As features (todas anti-lookahead, janela terminando em `t`)

| Feature | Hipótese / origem | Sinal esperado |
|---|---|---|
| `mom_12_1` | Momentum 12-1 (retorno de 252du pulando os últimos 21) — Jegadeesh-Titman | + |
| `rev_1m` | Reversão de curto prazo (retorno de 21du) | − |
| `dias_desde_resultado` | Dias desde o `DT_RECEB` do último balanço (regime PEAD) | — (gate) |
| `crescimento_lucro_yoy` | Crescimento anual do lucro líquido TTM (growth) | + |
| `pl` | Preço/Lucro (value) | − |
| `pvp` | Preço/Valor Patrimonial (value) | − |
| `roe` | Return on Equity (qualidade) | + |
| `margem` | Margem líquida (qualidade) | + |
| `divida_ebitda` | Dívida líquida / EBITDA (alavancagem/risco) | − |
| `volume_relativo` | Volume do dia / média de 20du (atenção/liquidez) | + |
| `selic_nivel` | Nível da Selic meta | — (gate) |
| `selic_var_21d` | Variação da Selic em 21du | — (gate) |
| `cambio_var_21d` | Variação do câmbio USD/BRL em 21du | — (gate) |
| `score_econ` | `ScoreEcon.score_total` do ECON (a feature-tese) | + |
| `econ_confianca` | `ScoreEcon.confianca` (detecta os "dois zeros", §9) | — (gate) |
| `econ_n_noticias` | `ScoreEcon.n_noticias` (intensidade do evento) | — (gate) |

`SINAL_ESPERADO` só fixa direção para as features com tese direcional clara; as de
**gate/regime** (macro, dias desde resultado, confiança) ficam `None` — não têm
sinal fixo porque agem condicionando outras, não empurrando o retorno num sentido.

### Como o ECON entra — só o `score_total` (decisão Opção A)

Ponto de design central, documentado na **NOTA DE DESIGN** no topo do módulo:
saúde financeira, momento setorial e macro entram como **features cruas do
JOURNAL** (`get_fundamentals`/`get_macro`) — **nunca** os componentes `comp_*` do
ECON. O único campo do ECON usado como feature de sinal é o `score_total` (impacto
da notícia). `econ_confianca` e `econ_n_noticias` entram como **gates** (o modelo
aprende a descontar um score de baixa confiança), não como sinal duplicado.

Por quê: se alimentássemos `comp_saude_financeira` **e** `pl/roe/margem` da CVM,
estaríamos dando ao GradientBoosting o mesmo sinal por dois caminhos (colinearidade
explícita), inflando artificialmente a importância daquele bloco. Consumindo só o
`score_total` do ECON + os fundamentos crus do JOURNAL, cada fonte entra uma vez.

### `crescimento_lucro_yoy` não é PEAD clássico (honestidade)

A feature de "surpresa de lucro" é o **crescimento anual do lucro líquido TTM**,
não o SUE (Standardized Unexpected Earnings) do PEAD clássico — que exigiria
consenso de analistas, indisponível no nosso pipeline. É uma feature de growth
(literatura value/growth), com sinal teórico +. Chamamos pelo nome certo para não
vender PEAD que não temos.

### Como defender na banca

> "São 16 features, cada uma uma hipótese da literatura com direção de sinal
> esperada: momentum e reversão de preço, growth e value e qualidade de
> fundamentos vindos da CVM, regime macro, e o score de notícia do ECON. O ECON
> entra só pelo score_total — os fundamentos e macro vêm crus do JOURNAL, para não
> alimentar sinal duplicado ao modelo. E somos honestos na nomenclatura: nossa
> feature de lucro é crescimento YoY de TTM, growth, não PEAD com surpresa vs
> consenso, que não temos como calcular."

---

## 4. Anti-lookahead — defesa em profundidade no aprendizado

Lookahead num modelo supervisionado é ainda mais traiçoeiro que na coleta de
dados, porque o vazamento pode entrar pela feature, pelo label OU pelo
procedimento de validação. São **cinco defesas**:

**1. Feature em `t` usa só dados `≤ t`.** Toda janela (momentum, volume, macro)
termina em `t`. Os fundamentos herdam o corte por `DT_RECEB` do JOURNAL.

**2. Label usa `t+1..t+5` (forward, legítimo) e linhas sem `t+5` são DROPADAS.**
O alvo é o futuro que queremos prever; incompletos não viram zero, são removidos.

**3. Split com PURGE + EMBARGO (López de Prado — `PurgedTimeSeriesSplit`).** Como
o label de `t` olha até `t+5`, uma amostra de treino cujo intervalo `[t, t+5]`
invade o bloco de teste **vaza o futuro do teste para o treino**. O purge remove
essas amostras; o embargo adiciona um gap extra de `embargo=5` dias úteis entre
treino e teste (o `gap = horizonte + embargo = 10du`). Sem isso, a validação
cruzada superestimaria o poder do modelo.

**4. Imputação só com estatística cross-sectional do próprio dia.** Um valor
faltante (ex.: `pl` sem balanço) é preenchido com a **mediana daquele dia entre os
outros tickers** — nunca com estatística futura ou global. O fallback (quando o
dia inteiro é NaN naquela coluna) usa a **mediana computada só no subset de
treino** (`data ≤ treino_fim`), congelada antes de tocar o OOS. Nunca a mediana
global (que veria o futuro).

**5. `_assert_no_lookahead` na saída de `prever` (defesa em profundidade).** Antes
de emitir uma previsão em produção, revalida que preços, fundamentos (`DT_RECEB`),
séries macro e o `ScoreEcon` (`data_referencia`) não contêm nada `> t`. Não
confiamos só nos contratos do JOURNAL/ECON — checamos de novo e levantamos
`LookaheadError` se algo passar.

### Por que cinco camadas

Porque um único vazamento **invalida todo o backtest** — não "tira pontos", zera o
resultado. O purge+embargo em especial é o que separa um IC honesto de um IC
inflado por sobreposição de labels de 5 dias. É padrão da indústria (López de
Prado, *Advances in Financial Machine Learning*).

### Como defender na banca

> "Temos anti-lookahead em profundidade no aprendizado: features só com dados até
> t, labels forward com descarte de incompletos, e — o mais importante — split
> temporal com purge e embargo de López de Prado, que remove do treino as amostras
> cujo label de 5 dias invade o teste, mais um gap de 5 dias úteis. A imputação usa
> só a mediana cross-sectional do dia, com fallback na mediana do treino congelada
> antes do OOS. E revalidamos a ausência de lookahead na saída de cada previsão."

---

## 5. `construir_dataset` — pré-fetch anti-n+1 + survivorship por dia

### O que faz

Monta o painel `(data, ticker)` com as 16 features, o alvo e as flags, para um
range de datas. Duas fases:

- **FASE 1 — pré-fetch (`_prefetch`):** uma chamada por fonte por ticker,
  guardada num `_DatasetCache` em memória (preços, Ibov, fundamentos indexados por
  âncoras mensais, macro).
- **FASE 2 — montagem em memória (zero I/O):** itera o calendário de pregões e,
  para cada `(t, ticker)`, calcula alvo e features **lendo do cache**.

### Por que o pré-fetch (o n+1 que ele mata)

A versão ingênua chamaria `get_precos`/`get_fundamentals`/`get_macro` uma vez por
`(ticker, dia)` — para ~24 tickers × ~1500 dias, são **dezenas de milhares** de
chamadas ao JOURNAL, cada uma podendo tocar disco/rede. O pré-fetch reduz isso a
**uma chamada por fonte por ticker** (mais âncoras mensais de fundamentos), e a
montagem vira aritmética em memória. É a diferença entre um run de minutos e um de
horas. Um teste guardião (`test_prefetch_chamadas_unicas`) trava a regressão a n+1
contando chamadas.

### Survivorship por dia — a peça que a banca cobra

Se `tickers=None` (modo produção), o dataset usa **`tickers_ativos(t)` por dia** —
o universo de cada data reflete quem estava no IBOV **naquela data**, com entradas
e saídas do `UNIVERSO_HISTORICO` (ver `MANUAL_JOURNAL.md` §9). Uma ação que saiu do
índice (ou quebrou, como AMER3) aparece até a data de saída e some depois. Passar
uma lista fixa de tickers aplica o mesmo universo a todo o range — **incorreto
para survivorship** e usado só em testes unitários. O modo oficial é `None`.

O pré-fetch, nesse modo, busca a **união** dos tickers ativos no range e depois a
FASE 2 filtra por dia. Tickers sem dados (ex.: ELET3/JBSS3 delisted, 404 no
yfinance) são **pulados** no pré-fetch sem abortar o run (`test_prefetch_tolera_
ticker_sem_dados`) — a FASE 2 os exclui do painel sem tempestade de retries.

### Como defender na banca

> "O dataset é montado em duas fases: um pré-fetch que faz uma chamada por fonte
> por ticker e um assembly em memória, o que elimina o problema n+1 e leva o run de
> horas para minutos. E o universo é survivorship-correto por construção: no modo
> de produção usamos os tickers ativos em cada data específica, então empresas que
> saíram do índice ou quebraram entram no painel só até a data de saída — não há
> viés de sobrevivência."

---

## 6. O modelo — GradientBoosting, seleção de árvores e peso de eventos

### `treinar` — o fluxo

Treina o `GradientBoostingRegressor` no subset `data ≤ data_treino_fim`.
Determinístico (`random_state=42`). O número de árvores é escolhido por CV (abaixo),
salvo se `n_estimators_override` for passado (experimentos).

### Seleção de `n_estimators` — regra de platô COM fallback para argmax

O número de árvores é o principal knob de regularização do boosting: poucas
sub-ajustam, muitas sobre-ajustam. Escolhemos por **validação cruzada temporal**
(`PurgedTimeSeriesSplit`): para cada fold, treinamos o máximo de árvores e olhamos
o IC de validação em cada estágio (`staged_predict`). Daí:

- **argmax** = número de árvores com o maior IC médio de CV.
- **platô** = o **menor** n cujo IC médio está dentro de `0.5σ` do pico (o começo
  do platô — mais regularizado que o argmax, evita pescar o pico do ruído).
- **FALLBACK (`_escolher_n_estimators_platau`):** se `n_platau < 0.3 × n_argmax`,
  o platô foi **ganancioso demais** (com IC de CV quase-ruído, o σ no pico fica
  grande e o threshold cai abaixo do estágio 1 → o platô "pesca" n≈1, que dá
  **predição constante e IC=NaN**). Nesse caso, cai no `n_argmax`. A fonte da
  escolha (`platau` vs `argmax_fallback`) fica registrada em `cv_report` para
  auditoria.

Essa guarda foi adicionada depois que um diagnóstico dirigido flagrou o platô
colapsando para 1 árvore no run oficial (predição constante, IC zerado). Com o
fallback, os 4 modos de sensibilidade passaram a escolher um n saudável (4 nos
modos de sinal fraco, 17 no forte), e o IC de evento do modo `meta` recuperou de
**+0.00 → +0.15** (§9, §10).

### `sample_weight` de eventos — `MathMLConfig.sample_weight_eventos = 5.0`

Por padrão, `treinar` pondera as linhas de **evento** (`tem_evento=True`) com peso
5× no fit (`np.where(tem_evento, 5.0, 1.0)`). Motivação: os eventos (dias com
notícia) são a minoria das linhas, mas são exatamente onde o `score_econ` carrega
sinal; sem o peso, o sinal do evento fica **diluído** no mar de dias sem notícia, e
o modelo aprende a ignorar a feature-tese. Com o peso 5×, o ganho de `score_econ`
na importância subiu de ~0.02 para ~0.26 (virou a feature #1) sem sacrificar o IC
líquido. Passar `sample_weight_eventos=1.0` desativa (peso uniforme). Um callable
explícito em `treinar(sample_weight=...)` tem precedência (usado em experimentos).

### O contrato de saída (`prever_universo`)

Em produção, `prever_universo(tickers, data_limite)` monta as features de cada
ticker no dia, prevê `y_pred`, ordena por `y_pred` desc e entrega ao ORQUESTRADOR
um DataFrame com **8 colunas**: `ticker`, `y_pred`, `score_econ`, `tem_evento`,
`rank` (1 = melhor), `volume_relativo`, `data_noticia_mais_recente` e `setor`.

Dois detalhes de contrato que importam:

- **`volume_relativo` sai CRU (pré-imputação).** A previsão `y_pred` usa o valor
  imputado (mediana cross-sectional do dia), mas a coluna projetada expõe o valor
  **medido** — `NaN` quando há <20du de histórico. Assim o filtro de liquidez do
  ORQUESTRADOR (`volume_relativo > 1.5`) falha corretamente para o `NaN`
  (`NaN > 1.5 == False`); imputar aqui vazaria sinal transversal entre tickers.
- **`setor` e `data_noticia_mais_recente` são metadados de decisão**, não features.
  `setor` vem de `get_setor` (lookup puro no `UNIVERSO_HISTORICO`; propaga
  `DadoIndisponivel` se faltar — sinaliza bug de universo em vez de silenciar).
  `data_noticia_mais_recente` é normalizado para tz-aware `America/Sao_Paulo` (ou
  `NaT`) e serve ao ORQUESTRADOR para decidir D+1 vs D+2 (notícia pós-17h da B3).

### Por que boosting raso e não uma rede / um modelo linear

Um modelo linear (ex.: a regressão implícita de "só score_econ") captura o sinal
linear mas nenhuma **interação** (ex.: "momentum só funciona quando o volume é
alto"). Uma rede neural teria capacidade de sobra para decorar ~30 mil linhas
ruidosas. O GradientBoosting raso é o meio-termo: captura interações de baixa
ordem, é robusto a features em escalas diferentes (não precisa normalizar), e a
profundidade 3 + subsample + learning rate baixo o mantêm regularizado. Para o
tamanho e a razão sinal/ruído deste problema, é a escolha padrão de quant.

### Como defender na banca

> "O número de árvores é escolhido por validação cruzada temporal com purge, por
> uma regra de platô — o começo do platô de IC, mais regularizado que o pico. E há
> uma guarda: se o platô colapsa para pouquíssimas árvores porque o IC de CV é
> quase ruído, caímos no argmax, evitando um modelo de predição constante.
> Ponderamos as linhas de evento em 5× no treino, porque é onde mora o sinal de
> notícia — isso elevou o score do ECON de importância marginal para a feature
> principal do modelo, sem perder IC."

---

## 7. Avaliação — IC, IC95 por block bootstrap, baselines e walk-forward

### `avaliar_ic` — as métricas

- **IC_total** = Spearman(previsão, alvo) sobre todas as linhas. Rank correlation:
  mede se o modelo **ordena** bem, não se acerta a magnitude (o que importa para
  ranking). 0 = aleatório; >0.05 já é útil em painel; >0.10 respeitável.
- **IC_evento** = o mesmo, mas só no subset `tem_evento=True` (≥3 linhas). É onde
  o ECON contribui — a métrica que mais importa para a tese do sistema.
- **IC95** = intervalo de confiança de 95% por **block bootstrap** (abaixo).
- **baselines + GAP** = o valor agregado do modelo sobre alternativas triviais.

### IC95 por block bootstrap (`_block_bootstrap_ic`)

Um bootstrap i.i.d. de pares `(previsão, alvo)` **subestima a incerteza** aqui,
porque (a) tickers do mesmo dia são correlacionados (choque comum de mercado) e
(b) labels de 5 dias se sobrepõem no tempo (dependência serial). O block bootstrap
reamostra **datas inteiras com reposição** — cada data sorteada entra com todos os
seus tickers —, preservando a correlação cross-sectional e quebrando a
sobreposição serial. O intervalo resultante é apropriadamente mais largo
(`test_block_bootstrap_intervalo_mais_largo` prova que é ≥1.3× o i.i.d.). Com
menos de 10 datas, o bootstrap é declarado inviável (não inventamos precisão).

### Baselines competitivos e o GAP (`_calcular_baselines`)

O modelo só se justifica se **bater alternativas triviais**. Três baselines:

- **B1 — só `score_econ`** (sem ML): usa o score do ECON cru como previsão.
- **B2 — só `mom_12_1`** (o competidor óbvio em equities: momentum).
- **B3 — intercepto puro** (constante → IC=0; o piso absoluto).

`GAP = IC_modelo − max(baselines)`. Um GAP positivo é o que prova que o
GradientBoosting agrega sobre "só usar o score" ou "só usar momentum". Reportamos o
GAP **honestamente**, mesmo quando é ≈0 (empate) — ver §9/§10.

**Detalhe de honestidade (`_max_baseline`):** quando há poucos eventos (3 ≤ n ≤ 30),
o `IC_evento` existe mas os baselines de evento não são robustos. Nesse caso o
`GAP_evento` é reportado como **NaN** (comparação inviável), não como
`IC_evento − 0` — um piso 0 inflaria artificialmente o edge no subset pequeno.

### `walk_forward` — o backtest sem vazamento

Em vez de treinar uma vez e prever todo o OOS, o `walk_forward` **retreina na
fronteira de cada mês** (`freq="MS"`), usando só dados até `teste − horizonte −
embargo` (purge+embargo entre treino e cada mês de teste). Isso simula produção
(o modelo é atualizado conforme dados chegam) e mede **non-stationarity**: se o IC
walk-forward divergir muito do estático, o sinal não é estável no tempo. O
retreino herda o `sample_weight` do config (consistente com o modelo principal).

### Como defender na banca

> "Medimos o IC por Spearman, total e no subset de evento, com intervalo de
> confiança por block bootstrap que reamostra datas inteiras — preservando a
> correlação intradia e a sobreposição dos labels de 5 dias, o que um bootstrap
> ingênuo ignoraria. E não reportamos o IC no vácuo: comparamos contra baselines
> triviais — só o score do ECON, só momentum, e o intercepto — e reportamos o GAP,
> inclusive quando é um empate. O backtest é walk-forward com retreino mensal e
> purge+embargo, que também mede estabilidade temporal do sinal."

---

## 8. Importância de features + checagem de sinal (`importancia_features`)

### O que faz

Depois de treinado, o modelo reporta, por feature: o **ganho** (importância do
GradientBoosting) e o **sinal observado** — a direção do efeito da feature na
previsão, medida variando a feature em `±1` a partir das medianas de treino e
olhando o sinal da mudança na previsão (proxy de dependência parcial). Esse sinal
observado é cruzado com `SINAL_ESPERADO`: se diferirem, a linha é marcada
`sinal_invertido = True`.

### Por que — o detector de overfit

Uma feature com **sinal invertido** vs a hipótese teórica é um **red flag**: ou o
modelo decorou ruído da amostra, ou há um regime específico do período (ex.:
momentum invertendo numa fase de reversão). Não escondemos: reportamos e
investigamos. No run oficial, com o `sample_weight` ativo, **nenhuma** feature saiu
invertida e o `score_econ` virou a de maior ganho — o modelo aprendeu a tese certa
(§10). Sem o peso, o `mom_12_1` aparecia invertido; a auditoria pegou isso.

### Como defender na banca

> "Não tratamos o modelo como caixa-preta: para cada feature reportamos a
> importância e a direção do efeito na previsão, e cruzamos com a direção teórica
> esperada. Uma feature que o modelo aprendeu com sinal invertido vs a hipótese é
> um alarme de overfit ou de regime — reportamos e investigamos. No resultado
> final, nenhuma feature ficou invertida e o score do ECON é a de maior ganho: o
> modelo aprendeu exatamente a hipótese econômica que o sistema propõe."

---

## 9. O mock estruturado do ECON (`make_econ_mock`)

### Por que existe

O ECON real depende de `ANTHROPIC_API_KEY`, ainda ausente nesta fase. Para
desenvolver, testar e **estressar** o MATH&ML sem a chave, `make_econ_mock` injeta
um `score_econ` **controlado**, calibrado a um IC-alvo conhecido. Ele responde a
duas perguntas: *(a)* o pipeline reconhece sinal quando há sinal? e *(b)* qual a
sensibilidade da estratégia à qualidade do ECON (se o ECON entregar IC 0.10 em vez
de 0.15, o que acontece)?

### Como o sinal é construído (unimodal, transferível)

Em um dia com evento, o mock injeta:

```
score = clamp( α·z(y) + √(1−α²)·ruído , −1, +1 )
```

onde `z(y)` é a **padronização cross-sectional do alvo realizado** daquele dia
(quão idiossincraticamente forte aquela ação foi vs as outras do universo naquele
dia), e `ruído` é gaussiano reprodutível (seed estável entre processos via SHA-256
— `_stable_seed`, porque o `hash()` do Python é salgado por `PYTHONHASHSEED`). O
`α` controla quanto do score é sinal vs ruído. A distribuição é **unimodal** (sem
reescala bimodal), para que a passagem mock → ECON real não exija retreino.

**O `y` futuro entra SÓ aqui, na geração do score (fixture) — nunca como feature
do modelo.** É a única concessão de lookahead, e é deliberada e isolada: o mock
*é* o sinal sintético; no deploy real o ECON entra de verdade e o mock some.

### `α` auto-calibrado por busca binária

`α` não é chutado: é encontrado por **busca binária** (até 40 iterações) até que
`Spearman(score, y) ≈ ic_alvo` numa `amostra_calibracao`. A amostra é usada **só
para calibrar** — em runtime o `y` é recomputado via JOURNAL, nunca lido da
amostra. Há um **fast-path** (quando a amostra já traz a coluna `y`) que computa o
`z(y)` in-memory com `ddof=0` (populacional) — casando exatamente com o `.std()` do
numpy usado no runtime (um bug de ddof inconsistente entre calibração e runtime foi
corrigido e travado por `test_mock_fastpath_calibracao_consistente`).

### Os "dois zeros" — degradação que o MATH&ML detecta

Igual ao ECON real (ver `MANUAL_ECON.md` §5), o mock distingue dois tipos de zero:
- **neutro genuíno:** sem evento → `score=0, confianca>0, tem_evento=False`.
- **degradado:** evento que não pôde ser avaliado (`universo_insuficiente` com
  |U|<3, ou `y_indisponivel` sem janela `t+5`) → `score=0, confianca=0,
  tem_evento=True`. O agente marca `econ_degradado=True`.

Por isso `econ_confianca` é feature: o modelo aprende a **não confiar** num
`score_econ=0` que veio com confiança zero. Distinguir "avaliado como neutro" de
"não consegui avaliar" é um contrato do sistema inteiro.

### Como defender na banca

> "Sem a chave da Anthropic, usamos um mock estruturado do ECON que injeta um sinal
> controlado — z-score cross-sectional do retorno realizado misturado com ruído —,
> com o alpha calibrado por busca binária a um IC-alvo. Isso não é trapaça: o
> retorno futuro entra só na geração do score sintético, nunca como feature, e o
> propósito é medir a sensibilidade da estratégia à qualidade do ECON. O mock
> respeita os mesmos contratos do ECON real, incluindo os dois tipos de zero, então
> a transição para o ECON de verdade não exige retreinar."

---

## 10. Calibração de sensibilidade — os 4 modos (o resultado oficial)

O run oficial (`scripts/sensibilidade_econ.py`, relatório em
`calibration/results/RELATORIO_CALIBRACAO_MATHML.md`) roda o pipeline completo nos
períodos oficiais (warmup 2019, treino 2020-2023, OOS 2024-2025, universo real com
survivorship por dia) em **quatro modos** de qualidade do mock do ECON:

| Modo | IC-alvo do ECON | IC_evento OOS do MATH&ML |
|---|---|---|
| ruído | 0.00 | +0.029 (≈0 — pipeline não inventa sinal) |
| fraco | 0.10 | +0.095 |
| meta | 0.15 | **+0.147** |
| forte | 0.20 | +0.189 |

### O que o resultado prova

- **Piso limpo:** com ECON puro ruído (IC-alvo 0), o MATH&ML entrega IC_evento ≈ 0
  — o pipeline **não fabrica sinal** onde não há. É o controle negativo.
- **Monotonia:** o IC de evento cresce monotonicamente com a qualidade do ECON. A
  estratégia **responde proporcionalmente** ao sinal injetado — exatamente o
  comportamento esperado de um stress test.
- **Empate honesto com B1 no modo meta:** o modelo (IC_evento +0.147) empata
  estatisticamente com o baseline B1 (só `score_econ`, +0.143; GAP +0.004, dentro
  do IC95). Isso é o resultado **esperado por construção**: o mock injeta sinal
  essencialmente **linear** no `score_econ`, e B1 (o ótimo linear) é difícil de
  bater sem interações não-lineares — que o mock não contém. O empate **valida o
  mock, não desqualifica o ML**: com o ECON real (sinal contextual, possíveis
  não-linearidades), a expectativa é o GBM abrir vantagem via interações.
- **`score_econ` vira a feature #1** (ganho ~0.26) e **nenhuma feature sai
  invertida** — o `sample_weight` de eventos alinhou o modelo com a hipótese
  econômica central.

### Como defender na banca

> "Estressamos o MATH&ML em quatro cenários de qualidade do ECON. Com ECON puro
> ruído, o IC de evento é zero — o pipeline não inventa sinal. E o IC cresce
> monotonicamente com a qualidade do ECON, de 0.03 a 0.19. No cenário-meta, o
> modelo empata com o baseline de usar só o score — que é o esperado, porque o mock
> é linear e um baseline linear é o ótimo nesse caso; a vantagem do boosting
> aparece quando o sinal real tiver não-linearidades. O importante: o score do ECON
> é a feature de maior ganho e nenhuma feature ficou com sinal invertido."

---

## 11. Limitações honestas (e como apresentar)

**1. O mock não é o ECON real.** "Este run mede a **sensibilidade** do MATH&ML à
qualidade do ECON, não a performance final do sistema — que depende do ECON real,
pendente de `ANTHROPIC_API_KEY`. O mock injeta sinal linear por construção, então o
empate do modelo com o baseline linear é esperado e valida a metodologia."

**2. `crescimento_lucro_yoy` é growth, não PEAD.** "É o crescimento YoY do lucro
TTM, não o SUE clássico com surpresa vs consenso de analistas — que não temos como
calcular sem dados de consenso. Chamamos pelo nome certo."

**3. Beta contra Ibov, não contra setor.** "O alvo usa beta vs Ibovespa;
`beta_contra_setor` está plumado mas levanta `NotImplementedError` — `get_retornos_
setor` entrega um agregado, não a série diária necessária para estimar o beta
setorial. É um refinamento futuro."

**4. Universo restrito ao `UNIVERSO_HISTORICO`.** "Cobrimos os ~24 tickers
emblemáticos com survivorship por data; pode não cobrir 100% do IBOV em cada
rebalanceamento. Casos incertos estão marcados com confiança no config."

**5. Tickers sem dados no yfinance.** "ELET3 e JBSS3 não são baixáveis neste
ambiente (404 no yfinance) e são **excluídos** do painel sem abortar o run.
Cobertura reduzida nos setores afetados, reportada."

**6. Custos de transação não entram aqui.** "O MATH&ML prevê e ranqueia; custos,
slippage e a P&L final são responsabilidade do PROGRAM (backtest financeiro). O IC
mede poder preditivo, não retorno líquido."

**7. Cache do JOURNAL com TTL de 24h.** "Reruns dentro de 24h aproveitam o cache em
disco; após isso, o primeiro run é cold (fetch ao vivo do yfinance/BCB/CVM), da
ordem de ~90 min para os 4 modos."

---

## 12. Glossário rápido

- **Retorno idiossincrático:** retorno da ação além do explicado pelo beta vezes o
  mercado (`r_ação − β·r_ibov`). Isola o alpha específico da seleção.
- **Beta:** sensibilidade do retorno da ação ao do mercado, aqui via OLS numa
  janela de 252du terminando em `t`.
- **IC (Information Coefficient):** correlação de Spearman entre previsão e retorno
  realizado. Métrica padrão de poder preditivo em quant; mede ordenação, não
  magnitude.
- **GradientBoosting:** ensemble de árvores rasas somadas em sequência, cada uma
  corrigindo o erro da anterior. Raso + regularizado = interações sem decorar ruído.
- **`n_estimators` / regra de platô:** número de árvores, escolhido por CV como o
  começo do platô de IC (mais regularizado que o pico), com fallback ao argmax se o
  platô colapsar.
- **PurgedTimeSeriesSplit (purge + embargo):** split temporal que remove do treino
  amostras cujo label futuro invade o teste, mais um gap extra. Evita vazamento por
  sobreposição de labels (López de Prado).
- **Block bootstrap:** reamostragem de **datas inteiras** com reposição para o IC95,
  preservando correlação cross-sectional e sobreposição serial.
- **Baseline / GAP:** alternativa trivial (só score, só momentum, intercepto) e a
  diferença de IC do modelo sobre a melhor delas. Prova (ou não) o valor do ML.
- **`sample_weight` de eventos:** peso maior (5×) nas linhas com notícia no treino,
  para o sinal do evento não diluir no mar de dias sem evento.
- **Walk-forward:** backtest com retreino periódico (mensal), simulando produção e
  medindo estabilidade temporal do sinal.
- **`score_econ`:** o `ScoreEcon.score_total` do ECON — a feature-tese do sistema.
- **Dois zeros:** um `score=0` pode ser "avaliado como neutro" (confiança>0) ou
  "não consegui avaliar" (confiança=0, `econ_degradado`). O modelo olha a confiança.
- **Mock estruturado:** ECON sintético que injeta sinal calibrado a um IC-alvo para
  testar e estressar o MATH&ML sem a API real.
- **`sinal_invertido`:** feature cujo efeito aprendido contradiz a hipótese teórica
  — red flag de overfit/regime.

---

## 13. Perguntas que a banca pode fazer (com respostas)

**P: Por que prever retorno idiossincrático e não o retorno bruto?**
R: Retorno bruto é dominado pelo mercado — o modelo aprenderia a prever o Ibov, não
a selecionar ações. Subtraindo beta vezes o retorno do índice, isolamos o alpha
específico da ação, que é o que uma estratégia de seleção captura. O beta é
estimado por OLS numa janela de 252 dias úteis terminando na data de decisão.

**P: Como garantem que não há lookahead no modelo?**
R: Cinco camadas: features só com dados até t; labels forward com descarte de
incompletos; split com purge e embargo de López de Prado (remove amostras cujo
label de 5 dias invade o teste); imputação só com mediana cross-sectional do dia,
com fallback na mediana de treino congelada; e um assert final anti-lookahead na
saída de cada previsão. O purge+embargo é o que separa um IC honesto de um inflado.

**P: Por que GradientBoosting e não deep learning ou um modelo linear?**
R: Um linear captura sinal mas nenhuma interação; uma rede decoraria ~30 mil linhas
ruidosas. O boosting raso (profundidade 3) com learning rate baixo e subsample
captura interações de baixa ordem regularizado, é robusto a escalas diferentes de
feature e é o padrão de quant para este tamanho e razão sinal/ruído.

**P: O modelo bate um baseline trivial?**
R: Reportamos o GAP contra três baselines — só o score do ECON, só momentum, e o
intercepto. No mock (linear por construção), o modelo empata com o baseline de só o
score, porque um baseline linear é o ótimo para sinal linear; a vantagem do
boosting aparece com não-linearidades, esperadas no ECON real. Reportamos o empate
honestamente em vez de esconder.

**P: Como escolhem o número de árvores sem overfitar na escolha?**
R: Por validação cruzada temporal com purge, pegando o começo do platô de IC (mais
regularizado que o pico). E há uma guarda: se o platô colapsa para pouquíssimas
árvores porque o IC de CV é quase ruído, caímos no argmax — senão o modelo viraria
predição constante. A fonte da escolha fica registrada para auditoria.

**P: O que é esse mock do ECON — não é trapaça injetar o futuro?**
R: O mock injeta um score sintético calibrado a um IC-alvo para podermos
desenvolver e **estressar** o MATH&ML sem a chave da Anthropic. O retorno futuro
entra só na geração do score sintético, nunca como feature do modelo. O propósito é
medir a sensibilidade da estratégia à qualidade do ECON — e provamos que com ECON
puro ruído o IC é zero, e que ele cresce monotonicamente com a qualidade do sinal.

**P: Como sabem que o modelo aprendeu o sinal certo e não ruído?**
R: Cada feature tem uma direção de sinal esperada da literatura, e auditamos o sinal
que o modelo de fato aprendeu (variando a feature e olhando a previsão). Uma feature
com sinal invertido é red flag. No resultado final, nenhuma ficou invertida e o
score do ECON é a feature de maior ganho — o modelo aprendeu a hipótese econômica
central do sistema.

**P: Como o MATH&ML conversa com os outros agentes?**
R: Consome os dados crus do JOURNAL e o ScoreEcon do ECON como features, e entrega
ao ORQUESTRADOR um ranking do universo por retorno idiossincrático previsto
(`prever_universo`). O MATH&ML prevê e ordena; o ORQUESTRADOR decide posição
(top-N, limites setoriais, saídas) e o PROGRAM roda o backtest financeiro com
custos.

**P: Qual a métrica de sucesso e por que Spearman?**
R: O IC (Information Coefficient) por Spearman entre previsão e retorno realizado,
total e no subset de evento, com intervalo de confiança por block bootstrap.
Spearman porque queremos ordenar bem as ações (rank), não acertar a magnitude
linear do retorno — o ORQUESTRADOR usa o ranking, não o número absoluto.
