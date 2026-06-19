# ECON — Manual do Desenvolvedor para a Banca

Documento de estudo intensivo do segundo agente do JEMPO, o **ECON**. Depois de
ler isso, você consegue explicar qualquer linha de `agents/econ.py` e qualquer
decisão metodológica para a banca.

Use junto com o código aberto na tela. Cada seção aponta para um pedaço do
código, explica em camadas (o que faz, por que faz, o que rejeitamos, como
defender) e termina com a frase pronta para responder à banca.

Pré-requisito de leitura: `MANUAL_JOURNAL.md`. O ECON é cliente do JOURNAL —
consome `get_noticias`, `get_fundamentals`, `get_macro`, `get_retornos_setor` e
`get_setor`. Toda a disciplina anti-lookahead dos *dados* já mora no JOURNAL; o
ECON herda isso e adiciona a sua própria camada (a do raciocínio do LLM, §6).

---

## 0. Para que serve o ECON no JEMPO

O JEMPO tem cinco agentes. O JOURNAL é o **research** (provedor de dados puro). O
ECON é o **analista fundamentalista**: pega o dossiê limpo do JOURNAL e emite um
**julgamento qualitativo** sobre o impacto econômico das notícias de uma ação.

Concretamente, `EconAgent.avaliar(ticker, data_limite)` devolve um `ScoreEcon`:
um **`score_total` em [-1, +1]** que mede o impacto esperado da(s) notícia(s) no
**retorno em excesso ao Ibovespa nos próximos 5 dias úteis** (decisão Opção A,
§5), mais **componentes de contexto** (saúde financeira, momento setorial, macro)
que calibram essa leitura sem serem somados ao total, uma `confianca` em [0, 1], a
flag `tem_evento` e uma `justificativa` textual.

Pense no ECON como o analista sênior da mesa: ele não vai à fonte primária (isso
é o research/JOURNAL), recebe o dossiê e dá a leitura. Essa leitura — número +
componentes — é o que o MATH&ML combina com features quantitativas para decidir
posição. O ECON **não decide trade** (isso é ORQUESTRADOR); ele **pontua**.

**A regra que governa o ECON é UMA SÓ: pontuar o mecanismo econômico, não o tom.**
Uma manchete alarmista sobre "investigação política genérica" sem efeito em caixa,
margem ou múltiplo vale ~0. Um fato seco sobre revisão de capex que muda a
geração de caixa futura vale muito. Tudo no código serve a essa regra.

---

## 1. Arquitetura geral

O ECON está inteiro em `agents/econ.py` (um módulo só — ele é fino; o peso está
no JOURNAL). Estrutura do arquivo, de cima para baixo:

```
agents/econ.py
├── imports + constantes        # MODELO_PADRAO, _MAX_TOKENS, _MAX_NOTICIAS, _MAX_CONTEUDO_CHARS
├── @dataclass ScoreEcon        # contrato de saída tipado (consumido pelo MATH&ML)
├── _TOOL                       # schema rígido da ferramenta registrar_avaliacao
├── _SYSTEM_PROMPT              # persona + instruções (incl. anti-lookahead do LLM)
├── helpers (_clamp, _hash_noticias, _neutro)
└── class EconAgent
    ├── __init__                # injeta journal + client (DI p/ teste), model, cache_dir
    ├── _get_client             # cliente Anthropic lazy (None se sem chave/lib)
    ├── avaliar                 # API pública (orquestra o fluxo)
    ├── _montar_contexto        # monta o payload textual a partir do JOURNAL
    └── _parsear                # extrai o tool_use → ScoreEcon (ou None se inválido)
```

Reuso deliberado do JOURNAL (não reimplementamos): `from agents.journal import
JournalAgent, _DiskCache, _validate_aware`. O mesmo cache em disco (pickle, TTL
24h) e a mesma validação de timezone do JOURNAL valem aqui — consistência e menos
superfície de bug.

### Injeção de dependência (por que `EconAgent(journal=..., client=...)`)

O construtor aceita `journal` e `client` opcionais. Em produção, `journal=None`
cria um `JournalAgent` real e `client=None` cria o cliente Anthropic *lazy*. Nos
testes (`tests/test_econ.py`), injetamos um `journal` falso (dados fixos) e um
`client` falso (resposta fixa) — assim a suíte inteira é **determinística e
offline**, sem rede, sem chave, sem custo. Esse é o motivo de o ECON ser testável
de ponta a ponta sem nunca tocar a API real.

### Como defender na banca

> "O ECON é o agente analista: consome o dossiê limpo do JOURNAL e usa o Claude
> como analista fundamentalista para emitir um score de impacto econômico em
> [-1, 1] com componentes desagregados. Ele é fino de propósito — a disciplina de
> dados pesada está no JOURNAL — e é testável de ponta a ponta offline por
> injeção de dependência do provedor de dados e do cliente de LLM."

---

## 2. Event-driven: sem notícia, sem chamada ao LLM

### O que faz

A primeira coisa que `avaliar` faz depois de validar a data é
`self.journal.get_noticias(ticker, data_limite, lookback_days)`. **Se a lista vier
vazia, retorna um `ScoreEcon` neutro imediatamente — sem chamar o Claude.** Todos
os scores 0.0, `tem_evento=False`, `confianca=0.0`, justificativa "sem notícia
relevante".

### Por que

Duas razões. **Custo:** um backtest de 2 anos × ~18 tickers × passos diários são
milhares de avaliações; a esmagadora maioria dos dias **não tem notícia relevante**
para um dado ticker. Chamar o LLM nesses dias seria queimar dinheiro para receber
"neutro". **Semântica:** sem evento novo, a tese do ECON é que não há catalisador
qualitativo — o sinal correto é exatamente 0 (neutro), que é o que o MATH&ML
espera para "nada a dizer aqui".

É o análogo do design event-driven do JOURNAL: trabalho só quando há informação.
O teste `test_sem_noticia_nao_chama_claude` trava isso com
`client.messages.create.assert_not_called()`.

### Como defender na banca

> "O ECON é event-driven: sem notícia relevante na janela, ele devolve score
> neutro sem sequer chamar o LLM. Isso reflete a tese (sem catalisador novo, o
> sinal qualitativo correto é zero) e controla custo — a maioria dos dias-ticker
> não tem evento, e não pagamos inferência para ouvir 'neutro'."

---

## 3. Tool use com schema rígido — por que não parsear texto livre

### O que faz

Quando há notícia, o ECON chama `client.messages.create(...)` com três peças que
forçam saída estruturada:

- `tools=[_TOOL]` — uma única ferramenta `registrar_avaliacao` com `input_schema`
  JSON Schema declarando cada campo, tipo e faixa (`minimum`/`maximum`).
- `tool_choice={"type": "tool", "name": "registrar_avaliacao"}` — **força** o
  modelo a chamar essa ferramenta (não é opcional; ele não pode responder em prosa).
- `temperature=0` — estabilidade: a mesma entrada tende à mesma saída. É reforço;
  a reprodutibilidade do backtest vem do cache versionado (§8), não da temperatura.

O parsing (`_parsear`) varre `resposta.content`, acha o bloco com
`type == "tool_use"` e `name == "registrar_avaliacao"`, e lê o dict `.input`.

### Por que function-calling em vez de "peça um JSON no texto"

Pedir "responda em JSON" e dar `json.loads` na string é frágil: o modelo às vezes
embrulha em ```` ```json ````, adiciona um parágrafo antes, ou erra uma vírgula —
e o parser quebra no meio do backtest. Com **tool use forçado**, a própria API
garante que a saída é um objeto que adere ao `input_schema`. Trocamos parsing
de texto livre (frágil) por um contrato (robusto). É a prática recomendada para
extração estruturada com Claude.

### Os campos do schema

`score_total`, `componente_noticia`, `componente_saude_financeira`,
`componente_setorial`, `componente_macro` em [-1, 1]; `confianca` em [0, 1];
`justificativa` string (1-3 frases). Todos `required`.

### Defesa em profundidade no parse

Mesmo com schema, **não confiamos cegamente** no modelo. `_clamp` re-satura cada
número na sua faixa (um modelo pode, raramente, devolver 1.2). E se *nenhum* bloco
`tool_use` válido vier (resposta degenerada), `_parsear` devolve `None` e o
chamador degrada para neutro **sem cachear** (ver §7). O teste `test_score_clamp`
cobre a saturação; `test_parse_tool_use`, o caminho feliz.

### Como defender na banca

> "Usamos function-calling com tool_choice forçado e temperatura zero: o Claude é
> obrigado a responder via uma ferramenta cujo schema declara cada campo e sua
> faixa. Isso troca parsing de texto livre — frágil — por um contrato estruturado.
> Ainda assim aplicamos defesa em profundidade: re-saturamos os números nas faixas
> e degradamos com segurança se a resposta vier malformada."

---

## 4. O system prompt — a persona e as instruções

### O que faz

`_SYSTEM_PROMPT` define o ECON como **analista fundamentalista sênior de ações
brasileiras (buy-side)** e fixa as regras de pontuação:

1. **Mecanismo, não tom.** Avaliar o efeito esperado em caixa, margem, posição
   competitiva ou múltiplo — não o sentimento superficial da manchete.
2. **`score_total` = impacto da NOTÍCIA (decisão Opção A).** A nota principal mede
   **só o impacto da(s) notícia(s)** no **retorno em excesso ao Ibovespa nos
   próximos 5 dias úteis** (-1 muito negativo, 0 neutro, +1 muito positivo) — é o
   que o ECON tem autoridade para julgar. Ela deve refletir essencialmente o
   `componente_noticia`. Ancorar num horizonte e num benchmark torna a nota
   comparável entre ações e calibrável (§9).
3. **Contexto calibra, não soma.** Saúde financeira (fundamentos TTM), momento
   setorial e macro são o **contexto** que o ECON usa para calibrar a leitura da
   notícia (a mesma notícia pesa mais numa empresa frágil). São reportados nos
   campos próprios para interpretabilidade, mas **NÃO são parcelas somadas ao
   `score_total`** — essa separação evita alimentar sinal duplicado/colinear ao
   MATH&ML (ver nota de design em `agents/math_ml.py`).
4. **Agregar múltiplas notícias (P5).** Havendo várias notícias, ponderar pelo
   impacto fundamental e pela confiabilidade da fonte; notícias **contraditórias
   entre si devem REDUZIR a `confianca`**.
5. **Descontar ruído.** Eventos sem efeito fundamental (ex.: política genérica)
   tendem a 0.
6. **Anti-lookahead do LLM** (§6).
7. **Responder só pela ferramenta.**

### Por que "buy-side" e "mecanismo econômico"

Porque é exatamente a leitura que dá sinal preditivo. "Sentiment de manchete" é
ruidoso e já está embutido no preço quando a notícia sai; o que move retorno em
excesso é a **reavaliação do valor fundamental** — e isso exige raciocinar o
mecanismo (essa notícia muda o fluxo de caixa/risco/múltiplo da empresa?). A
persona buy-side orienta o modelo para esse tipo de julgamento, não para resumir.

### O payload do usuário (montado em `_montar_contexto`)

A partir do JOURNAL, montamos um JSON com: `data_limite`, `ticker`, `empresa`,
`setor`; lista de `noticias` (título, fonte, peso da fonte, data, conteúdo
truncado em `_MAX_CONTEUDO_CHARS`); `fundamentos` (P/L, P/VP, ROE, margem,
dívida/EBITDA, receita, lucro, e o mapa de `periodicidade` TTM/point-in-time);
`macro` (últimos valores ≤ data_limite de Selic, IPCA 12m, câmbio); e
`retornos_setor` (médio/mediano da janela). Limitamos a `_MAX_NOTICIAS` notícias
(as de maior peso, pois o JOURNAL já ordena) para controlar contexto e custo.

Cada coletor está em `try/except` que só adiciona um aviso — **uma fonte
indisponível não derruba a avaliação** (mesma filosofia do `_coletar` do JOURNAL).

### Como defender na banca

> "O prompt instala uma persona de analista buy-side e fixa o alvo: impacto no
> retorno em excesso ao Ibovespa em 5 dias úteis, avaliando o mecanismo econômico
> e não o tom. O contexto entregue é o dossiê do JOURNAL — notícia, fundamentos em
> TTM, macro e momento setorial — e cada peça é tolerante a falha, então a leitura
> sai mesmo com uma fonte fora do ar."

---

## 5. O contrato de saída `ScoreEcon` — o que cada campo significa

| Campo | Faixa | Significado |
|-------|-------|-------------|
| `score_total` | [-1, 1] | **Impacto da NOTÍCIA** (Opção A) — o sinal principal para o MATH&ML. |
| `comp_noticia` | [-1, 1] | Impacto isolado da(s) notícia(s) — base do `score_total`. |
| `comp_saude_financeira` | [-1, 1] | CONTEXTO: qualidade dos fundamentos (TTM). Não somado ao total. |
| `comp_setorial` | [-1, 1] | CONTEXTO: momento do setor. Não somado ao total. |
| `comp_macro` | [-1, 1] | CONTEXTO: vento macro (Selic, IPCA, câmbio). Não somado ao total. |
| `confianca` | [0, 1] | Confiança do modelo; notícias contraditórias a reduzem. |
| `tem_evento` | bool | Houve notícia relevante? |
| `n_noticias` | int | Quantas notícias entraram. |
| `justificativa` | str | Raciocínio econômico curto (rastreabilidade + auditoria, §6). |
| `modelo` | str | Qual modelo gerou (rastreabilidade). |
| `avisos` | list | Degradações/fontes indisponíveis / divergência score×notícia (§7). |

### Por que `score_total` é só a notícia (e os componentes são contexto)

Decisão **Opção A**: o `score_total` reflete **só o impacto da notícia** — o que o
ECON tem autoridade para julgar. Saúde financeira/setor/macro são produzidos como
**contexto** (interpretabilidade), não como parcelas somadas. Motivo prático: se o
total fosse combinação linear dos componentes, alimentaríamos **sinal duplicado e
colinear** ao GradientBoosting do MATH&ML — que já recebe saúde financeira, setor e
macro como **features cruas independentes do JOURNAL**. Assim, a contribuição
central do ECON (`score_total` = efeito da notícia) não tem redundância com as
features cruas, e os componentes `comp_*` ficam como interpretabilidade + features
opcionais (testadas por CV). A regra está fixada na nota de design em
`agents/math_ml.py`. A desagregação ainda dá **interpretabilidade** para a banca:
abrir um trade e ver "entrou por notícia positiva apesar de macro contrário".

### Coerência `score_total` × `comp_noticia` (P7)

Como `score_total` deve refletir `comp_noticia`, o `_parsear` faz uma **checagem de
sanidade leve**: se `abs(score_total − comp_noticia) > 0.5` (`_DIVERGENCIA_MAX`),
loga e acrescenta um **aviso** ao `ScoreEcon` — sem levantar exceção (degradação
graciosa). É um detector barato de resposta incoerente do modelo.

### O par `confianca` + `tem_evento` é como o consumidor detecta degradação

Detalhe importante de design: numa degradação (sem chave, erro de API, resposta
malformada), o ECON devolve `score_total = 0.0` **mas** `confianca = 0.0` e um
`aviso`. Um `score_total = 0` "de verdade" (o modelo avaliou e achou neutro) vem
com `confianca > 0`. **Portanto o MATH&ML deve olhar `confianca`/`avisos`, não só
o score**, para distinguir "avaliado como neutro" de "não consegui avaliar". Isso
está documentado aqui de propósito para a banca não confundir os dois zeros.

### Como defender na banca

> "A saída é um dataclass tipado com o score combinado mais quatro componentes
> desagregados, confiança e justificativa. A desagregação dá interpretabilidade e
> deixa o MATH&ML pesar fontes de sinal. E há um contrato sutil mas importante:
> confiança zero marca avaliação degradada, distinguindo-a de um neutro genuíno —
> os consumidores olham confiança e avisos, não só o número."

---

## 6. Anti-lookahead em duas frentes (a frente do LLM é a delicada)

O ECON tem **duas** camadas anti-lookahead, e a segunda é a mais sutil do sistema
inteiro.

### Frente 1 — os dados (herdada do JOURNAL)

Todos os dados que entram no payload vêm do JOURNAL, que já corta tudo em
`data_limite` (notícias por data de publicação, fundamentos por `DT_RECEB`, IPCA
por data de divulgação, etc.). O ECON ainda valida a própria entrada com
`_validate_aware(data_limite)` (timezone-aware obrigatório; `test_data_limite_
naive_levanta`). Essa frente é sólida — é a defesa em profundidade do JOURNAL.

### Frente 2 — o conhecimento do próprio LLM (a limitação honesta)

Aqui está a sutileza que a banca vai querer ouvir com clareza. **O Claude foi
treinado com dados históricos.** Para uma notícia de 2020-2021 — justamente o
período de calibração do ECON (§9) —, o modelo **pode "lembrar" como a ação de
fato se moveu nos dias seguintes** e usar essa memória para acertar o score. Isso
é um lookahead *pela memória do modelo*, e ele **inflaria artificialmente o IC** da
calibração (o modelo pareceria mais preditivo do que seria em produção).

Note a direção, porque é contraintuitiva: **não é que o modelo tenha mais
dificuldade com notícias antigas — é que ele tem facilidade demais, com cola.**
Para notícias **posteriores ao corte de conhecimento** do modelo, não há
vazamento: ele raciocina genuinamente do zero a partir do texto fornecido.

**As duas datas que governam isso (doc oficial Anthropic, `claude-haiku-4-5-20251001`):**
- **Reliable knowledge cutoff: FEV/2025** — até aqui o modelo conhece os fatos de
  forma confiável.
- **Training data cutoff: JUL/2025** — o treino inclui dados (menos confiáveis)
  entre fev e jul/2025.
- **Fronteira da validação LIMPA = o TRAINING cutoff (jul/2025)**, não o reliable:
  só notícia publicada **após jul/2025** está genuinamente fora do conhecimento do
  modelo. Como o backtest OOS é 2024-2025, **apenas jul-dez/2025** é a janela do
  backtest sem risco de lookahead de memória do LLM.

**Mitigações — duas defesas ativas + uma auditoria (no `calibration/econ_calibration.py`):**
- **DEFESA 1 — IC segmentado:** a calibração reporta o IC **separado por exposição
  ao treino**: 2020-2021 rotulado "TETO OTIMISTA (dentro do treino)"; jul-dez/2025
  "IC LIMPO (pós-training cutoff)". Helper `segmentar_por_exposicao`.
- **DEFESA 2 — Teste de placebo:** rerodamos o ECON nas mesmas notícias com a
  empresa **anonimizada** (ou trocada por par do setor) e reportamos
  **ΔIC = IC_real − IC_placebo**. Se o IC desaba ao anonimizar, parte do sinal era
  memória do modelo sobre aquela empresa — red flag. (Os hooks `noticias_override`
  e `nome_override` do `avaliar` existem para isso.)
- **AUDITORIA (P8):** `auditar_justificativa` varre cada justificativa por
  linguagem **ex-post** ("caiu X% depois", "posteriormente", "veio a se confirmar",
  "em retrospecto"…). Justificativa flagueada = inspeção manual. Monitoramos o
  lookahead do LLM **ativamente**, não só por instrução de prompt.
- Reforços passivos: o `_SYSTEM_PROMPT` manda raciocinar só com os dados da
  `data_limite`, e só entregamos o **texto ex-ante** da notícia (nunca o desfecho).

A instrução de prompt sozinha é fraca (o modelo não introspecta o que sabe do
treino); por isso as defesas 1 e 2 e a auditoria — é a forma honesta e ativa de
cercar a limitação, e é argumento direto para o critério de banca "Backtest: rigor
e mitigação de vieses" (15% da nota).

### Como defender na banca

> "Temos duas camadas anti-lookahead. A dos dados é herdada do JOURNAL — tudo
> cortado em data_limite. A segunda é específica de LLM e tratada com rigor: o
> Haiku 4.5 tem reliable cutoff em fev/2025 e training cutoff em jul/2025, então só
> notícia pós-jul/2025 está fora do conhecimento dele. Reportamos o IC segmentado
> por exposição ao treino (2020-2021 como teto otimista, jul-dez/2025 como IC
> limpo), rodamos um teste de placebo anonimizando a empresa para medir quanto do
> sinal vinha de memória, e auditamos as justificativas por linguagem ex-post.
> Instrução de prompt sozinha não basta — mitigamos ativamente."

---

## 7. Degradação graciosa — e por que não cachear o degradado

### O que faz

O ECON **nunca levanta exceção** para o chamador por causa de uma falha de
infraestrutura — devolve um `ScoreEcon` neutro com `aviso`. Três caminhos de
degradação:

1. **Sem chave / lib ausente** (`_get_client()` devolve `None`): retorna neutro +
   aviso "ANTHROPIC_API_KEY ausente…". É o estado atual do projeto (o `.env` ainda
   não tem a chave) — o sistema roda e degrada, não quebra. (`test_degrada_sem_chave`)
2. **Erro na chamada da API** (rede, rate limit, 5xx): o `try/except` em volta de
   `messages.create` captura, loga e devolve neutro + aviso. (`test_erro_api_degrada`)
3. **Resposta malformada** (sem bloco `tool_use`): `_parsear` devolve `None`, e
   `avaliar` retorna neutro + aviso.

Por que essa filosofia: igual às fontes do JOURNAL — **um backtest de milhares de
passos não pode abortar porque uma chamada falhou**. Falha vira sinal neutro com
rastro (aviso), e a vida segue.

### O bug que a revisão pegou: cache-poisoning (BUG-1)

Na primeira implementação, `avaliar` cacheava **incondicionalmente** o resultado
de `_parsear`. Isso tinha um furo: se a resposta viesse malformada, o neutro
degradado era **gravado no cache por 24h** — e toda chamada seguinte para aquele
(ticker, data, conjunto de notícias) recebia o neutro envenenado por cache hit,
mesmo que o modelo já estivesse respondendo bem de novo.

É exatamente o padrão que o JOURNAL já tinha corrigido em `get_noticias` ("só
cacheia resultado não-vazio"). A correção: `_parsear` devolve `None` em falha;
`avaliar` só chama `cache.set` no caminho de **sucesso**; degradações retornam sem
cachear (são baratas de refazer, e a próxima tentativa pode dar certo). O teste
`test_resposta_malformada_nao_cacheia` trava isso exigindo que a 2ª chamada
**rechame a API** (`call_count == 2`), provando que não houve cache hit envenenado.

### Como defender na banca

> "O ECON degrada com segurança: sem chave, erro de API ou resposta malformada, ele
> devolve neutro com aviso em vez de levantar — um backtest de milhares de passos
> não pode abortar por uma falha pontual. E só cacheamos avaliações bem-sucedidas:
> resultados degradados nunca entram no cache, senão uma falha transitória
> contaminaria 24 horas de chamadas. É a mesma lição que aplicamos no JOURNAL."

---

## 8. Cache — reusando a infraestrutura do JOURNAL

### O que faz

Quando há notícia e o LLM é chamado, o resultado é cacheado com `_DiskCache` (o
mesmo do JOURNAL: pickle em `data/cache/`, TTL 24h). A chave é
`(ticker, data_limite.date(), modelo, _PROMPT_VERSION, nome_override, hash das notícias)`.

O `hash do conjunto de notícias` vem de `_hash_noticias`: ordena as notícias por
`(fonte, data, título)` e tira um SHA-256 — **determinístico** e independente da
ordem em que vieram. Incluí-lo na chave garante que, se o conjunto de notícias
mudar (saiu uma nova entre 10h e 17h), a avaliação é refeita; se for idêntico, é
reaproveitada.

### `_PROMPT_VERSION` na chave (P4) — a peça que faltava

A chave inclui **`_PROMPT_VERSION`**, uma string que bumpamos manualmente sempre
que `_SYSTEM_PROMPT` ou o schema da tool (`_TOOL`) mudam. Sem ela, havia um bug
sério **na calibração**: nas ≤10 iterações de ajuste de prompt, o cache não
invalidaria ao trocar o prompt → compararíamos o IC de um prompt novo usando
**avaliações cacheadas do prompt antigo**, corrompendo a calibração inteira. Com a
versão na chave, cada iteração de prompt tem seu próprio espaço de cache. (O
`nome_override` também entra na chave, para o placebo da §6 não colidir com a
avaliação real.) `test_cache_invalida_ao_mudar_prompt_version` trava isso.

### Reprodutibilidade vem do cache versionado (não do `temperature=0`)

Ponto a alinhar para a banca: `temperature=0` deixa a saída do LLM **altamente
estável**, mas LLMs não garantem reprodutibilidade bit-a-bit. **A
reprodutibilidade do backtest vem do cache persistente versionado**: a primeira
execução fixa a avaliação de cada `(ticker, dia, notícias, prompt)`, e as
seguintes leem do disco — idênticas por construção. Temperatura zero é um reforço,
não a garantia.

### Por que

No walk-forward, o mesmo (ticker, dia) pode ser avaliado mais de uma vez entre
execuções. O LLM é a parte cara e lenta do pipeline; cachear evita repagar pela
mesma inferência. `test_cache_evita_segunda_chamada` prova que a 2ª `avaliar`
idêntica **não** rechama o cliente.

### Como defender na banca

> "Cacheamos cada avaliação por ticker, data, modelo, VERSÃO DO PROMPT e um hash
> determinístico das notícias, reusando o cache em disco do JOURNAL. A versão do
> prompt na chave é o que torna a calibração correta — ao iterar o prompt, não
> reaproveitamos avaliações antigas. E é desse cache versionado que vem a
> reprodutibilidade do backtest, com a temperatura zero como reforço. Só sucessos
> são cacheados (ver §7)."

---

## 9. Calibração — IC segmentado + placebo (`calibration/econ_calibration.py`)

O módulo já está escrito (helpers puros testados em `tests/test_econ_calibration.py`);
a execução ao vivo depende de `ANTHROPIC_API_KEY` + uma amostra de eventos.

### O que mede

1. Eventos com notícia `[(ticker, data_limite), …]`. Para cada um, `score_total`
   do ECON vs. **retorno em excesso de 5 pregões** (preço da ação − Ibovespa,
   ambos do JOURNAL).
2. **IC = Spearman(score, excesso)** (`calcular_ic`, via pandas — sem nova
   dependência). **Meta > 0.15.**
3. Loop de ajuste de prompt ≤ 10 iterações; depois **congela** (bump de
   `_PROMPT_VERSION` a cada iteração, ver §8).

### Por que Spearman e não Pearson

Porque queremos saber se o ECON **ordena** bem as ações (rank), não se acerta a
magnitude linear do retorno. IC (Information Coefficient) por Spearman é o padrão
de quant research: 0 = aleatório, >0.05 já é útil em painel grande, >0.10–0.15 é um
sinal respeitável para texto.

### DEFESA 1 — IC segmentado por exposição ao treino

`calibrar()` não reporta um IC único: usa `segmentar_por_exposicao(data)` para
quebrar o resultado em **TETO OTIMISTA (≤ fev/2025, dentro do treino)**,
**INTERMEDIÁRIO (fev–jul/2025)** e **IC LIMPO (> jul/2025, fora do training
cutoff)**. Só o último é evidência limpa de generalização (ver §6).

### DEFESA 2 — Teste de placebo

`teste_placebo()` reroda o ECON nas mesmas notícias com a empresa **anonimizada**
(`anonimizar_noticias` + os hooks `noticias_override`/`nome_override` do `avaliar`)
e reporta **ΔIC = IC_real − IC_placebo**. ΔIC pequeno = o sinal vem do mecanismo
(bom); ΔIC grande = parte do sinal era memória do modelo sobre a empresa (red
flag). Suporta também o modo "swap" (trocar por um par do setor).

### Auditoria das justificativas (P8)

`calibrar()` agrega, por evento, os flags de `auditar_justificativa` — linguagem
ex-post que delataria vazamento de memória. O relatório conta quantas
justificativas foram flagueadas para inspeção manual.

### Como defender na banca

> "A calibração mede o IC (Spearman, meta >0.15) mas com mitigação ativa de viés:
> reportamos o IC segmentado por exposição ao treino do Haiku (teto otimista até
> fev/2025, IC limpo após jul/2025), rodamos um teste de placebo anonimizando a
> empresa e medindo o ΔIC, e auditamos as justificativas por linguagem ex-post.
> Ajustamos o prompt em ≤10 iterações com versão no cache, e congelamos."

---

## 10. Limitações honestas (e como apresentar)

**1. Conhecimento de treino do LLM (a principal).** "O Haiku 4.5 tem reliable
cutoff em fev/2025 e training cutoff em jul/2025, então pode ter memória do
desfecho de notícias até jul/2025 — o que inflaria o IC de calibração de 2020-2021.
Não só documentamos: mitigamos ativamente — IC segmentado por exposição ao treino,
teste de placebo (ΔIC ao anonimizar a empresa) e auditoria das justificativas por
linguagem ex-post. A validação limpa usa a fronteira do training cutoff (notícia
após jul/2025)." (Detalhe na §6 e §9.)

**2. Determinismo aproximado.** "Usamos temperature=0, o que torna a saída
altamente estável, mas LLMs não garantem bit-a-bit reprodutibilidade. A
reprodutibilidade do backtest vem do **cache persistente versionado** (chave inclui
`_PROMPT_VERSION`), que fixa a avaliação de cada (ticker, dia, notícias, prompt) na
primeira execução; a temperatura zero é reforço, não a garantia." (Detalhe na §8.)

**3. Custo de cobertura total.** "Avaliar todo dia-ticker com LLM seria caro; por
isso o design event-driven — só avaliamos quando há notícia relevante — e o cache.
A maioria dos dias-ticker não tem evento e custa zero."

**4. Qualidade depende do JOURNAL.** "O ECON é tão bom quanto o dossiê que recebe.
Se o GDELT/NewsAPI não cobriram uma notícia, o ECON não a vê. Isso é mitigado pela
arquitetura em camadas de fontes do JOURNAL, mas é uma dependência real."

**5. GDELT entrega pouco corpo de artigo.** "Para o histórico, o GDELT muitas vezes
dá só título e URL. O ECON pondera o mecanismo a partir do título + fundamentos +
macro; com o Bloomberg CSV (corpo curado) a leitura fica mais rica nos eventos
críticos."

**6. Os 'dois zeros'.** "Um score_total = 0 pode significar 'avaliado como neutro'
ou 'falhei em avaliar'. Resolvemos isso com confianca = 0 + avisos na degradação;
os consumidores devem checar esses campos, não só o número." (Detalhe na §5.)

---

## 11. Glossário rápido

- **Tool use / function calling:** recurso da API do Claude em que o modelo
  responde chamando uma ferramenta com input que adere a um JSON Schema, em vez de
  texto livre. Saída estruturada e validável.
- **`tool_choice` forçado:** configuração que obriga o modelo a usar uma ferramenta
  específica (não pode responder em prosa).
- **`temperature=0`:** parâmetro de amostragem em zero — saída quase determinística,
  necessário para backtest reproduzível.
- **IC (Information Coefficient):** correlação (aqui Spearman) entre o sinal previsto
  e o retorno realizado. Métrica padrão de poder preditivo em quant.
- **Retorno em excesso:** retorno da ação menos o do benchmark (Ibovespa). Isola o
  desempenho específico da ação do movimento geral do mercado.
- **Score desagregado:** além do número combinado, componentes separados (notícia,
  saúde financeira, setor, macro) para interpretabilidade e ponderação a jusante.
- **Degradação graciosa:** em falha de infraestrutura, devolver resultado neutro com
  aviso em vez de levantar exceção. Mantém o backtest rodando.
- **Cache-poisoning:** gravar no cache um resultado de falha transitória, que passa
  a contaminar chamadas seguintes pela validade do cache. Evitado cacheando só
  sucessos.
- **TTM / point-in-time:** ver `MANUAL_JOURNAL.md` §6 — fluxo em 12 meses contínuos
  vs. foto de balanço. O ECON recebe o mapa de periodicidade pronto do JOURNAL.
- **Lookahead do LLM:** o modelo "saber" o desfecho de um evento por tê-lo visto no
  treino. Específico de usar LLM em dados históricos.

---

## 12. Perguntas que a banca pode fazer (com respostas)

**P: Por que usar um LLM como analista em vez de um modelo de sentiment clássico?**
R: Sentiment de manchete é ruidoso e já está no preço quando a notícia sai. O que
move retorno em excesso é a reavaliação do valor fundamental, e raciocinar o
mecanismo econômico (efeito em caixa, margem, múltiplo) é exatamente onde um LLM
com a persona certa supera um classificador de polaridade. Pedimos julgamento
estruturado, não rótulo positivo/negativo.

**P: Como garantem que a saída do LLM é confiável e não quebra o pipeline?**
R: Tool use com schema rígido e tool_choice forçado — a API garante aderência ao
schema. Re-saturamos os números nas faixas, e qualquer resposta malformada degrada
para neutro com aviso, sem levantar. O cache versionado dá a reprodutibilidade
(temperatura zero é reforço).

**P: O LLM não está vendo o futuro nas notícias antigas?**
R: O `claude-haiku-4-5-20251001` tem reliable knowledge cutoff em **fev/2025** e
training data cutoff em **jul/2025** (doc Anthropic). Logo, para eventos até
jul/2025 — incluindo a calibração 2020-2021 — há risco de memória do desfecho.
Somos transparentes e mitigamos ativamente: IC **segmentado** por exposição ao
treino (2020-2021 = teto otimista; pós-jul/2025 = IC limpo), **teste de placebo**
(ΔIC ao anonimizar a empresa) e **auditoria** das justificativas por linguagem
ex-post. A fronteira da validação limpa é o training cutoff (jul/2025), não o
reliable.

**P: Qual modelo e por quê?**
R: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), temperatura zero, cutoffs
reliable fev/2025 e training jul/2025. Haiku porque o volume de avaliações no
backtest é alto e o custo/latência importam; a tarefa é julgamento estruturado bem
especificado, em que o Haiku entrega bom custo-benefício. O modelo é parametrizável
— trocar é uma linha. **TODO de calibração:** rodar um subconjunto também com
**Claude Sonnet 4.6** e comparar IC — modelos menores às vezes captam o TOM em vez
do MECANISMO (o erro que a regra fundadora proíbe). Se Haiku empata com Sonnet,
defendemos o custo; se Sonnet for muito superior, reavaliamos o trade-off.

**P: Como controlam o custo de chamar um LLM milhares de vezes?**
R: Design event-driven (sem notícia, sem chamada), cache em disco versionado por
(ticker, dia, modelo, versão do prompt, hash das notícias), teto de notícias
(`_MAX_NOTICIAS=8`) e truncamento do corpo por notícia (`_MAX_CONTEUDO_CHARS=800`).
A maioria dos dias-ticker não tem evento e custa zero.

**P: O que acontece se a API da Anthropic estiver fora no meio do backtest?**
R: O ECON captura o erro, loga, e devolve score neutro com aviso para aquele
ponto — o backtest continua. E não cacheamos esse neutro degradado, então quando a
API volta a avaliação é refeita.

**P: Como o ECON conversa com os outros agentes?**
R: Consome o JOURNAL (dados) e entrega o `ScoreEcon` ao MATH&ML, que combina o
score e seus componentes com features quantitativas. O ORQUESTRADOR usa o
resultado dessa combinação para decidir posição. O ECON pontua; não decide trade.

**P: Por que componentes desagregados além do score total?**
R: Interpretabilidade (abrir um trade e ver por que entrou) e flexibilidade (o
MATH&ML pode pesar notícia, fundamentos, setor e macro separadamente). Entregar só
um número jogaria informação fora.
