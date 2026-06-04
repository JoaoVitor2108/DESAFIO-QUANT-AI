# JOURNAL — Manual do Desenvolvedor para a Banca (v2 — pós-migração CVM)

Documento de estudo intensivo. Cobre o JOURNAL após a migração de fundamentos
do yfinance para a CVM. Depois de ler isso, você consegue explicar qualquer
linha do código e qualquer decisão metodológica para a banca.

Use junto com o código aberto na tela. Cada seção aponta para um pedaço do
código, explica em camadas (o que faz, por que faz, o que rejeitamos, como
defender) e termina com a frase pronta para responder à banca.

---

## 0. Para que serve o JOURNAL no JEMPO

O JEMPO tem cinco agentes. O JOURNAL é o primeiro e mais fundamental: é o
**provedor de dados puro** do sistema. Não avalia, não decide, não pondera.
Sua única função é coletar dados de fontes externas (notícias, preços,
fundamentos, macro), limpá-los, organizá-los e entregá-los aos outros agentes
(ECON, MATH&ML, ORQUESTRADOR, PROGRAM) num formato confiável, tipado e com
rastreabilidade.

Pense no JOURNAL como o departamento de research de uma gestora. Os analistas
(ECON) e os quants (MATH&ML) não vão à fonte primária — eles recebem dossiês
limpos do research. Se o research entregar lixo, tudo o que vem depois é lixo.
Por isso o JOURNAL precisa ser paranoicamente correto.

**A regra que governa o JOURNAL inteiro é UMA SÓ: anti-lookahead.** O JOURNAL
nunca, em hipótese alguma, entrega informação que não existiria publicamente
no momento `data_limite`. Tudo o que está no código serve a essa regra direta
ou indiretamente.

---

## 1. Arquitetura geral

O JOURNAL está em `agents/journal.py`. Mas ele não está sozinho — depois da
migração para CVM, criamos um módulo de fontes especializadas em
`agents/sources/`, que hoje contém:

```
agents/
├── journal.py              # API pública - métodos consumidos pelos outros agentes
└── sources/
    ├── __init__.py
    └── cvm.py              # Leitor especializado dos dados abertos da CVM
```

**Por que separar?** O parsing da CVM tem detalhes técnicos pesados (encoding
latin-1, estrutura diferente para bancos, versões reapresentadas) e ocupa ~400
linhas de código. Misturar tudo em `journal.py` polui o módulo principal.
Separar permite testar isolado, evoluir o parser sem mexer no JOURNAL, e
deixar a API pública (`get_fundamentals`) estável mesmo quando a fonte muda
internamente.

A estrutura de pastas de dados:

```
data/
├── cache/                  # cache geral (notícias, preços) - pickle, TTL 24h
├── bloomberg/              # CSVs manuais de notícias do Bloomberg
└── cvm/
    ├── raw/                # ZIPs originais baixados da CVM (10-50MB cada)
    └── processed/          # DataFrames consolidados em parquet (rápido)
```

`data/cvm/raw/` e `data/cvm/processed/` estão no `.gitignore` — são arquivos
grandes, e qualquer um pode regenerá-los baixando da CVM.

**Topo do journal.py:** imports, constantes, exceções customizadas
(`LookaheadError`, `DadoIndisponivel`), dataclasses tipadas (`Noticia`,
`Fundamentals`).

**Meio:** funções utilitárias privadas (com prefixo `_`).

**Núcleo:** classe `JournalAgent` com os métodos públicos.

**Final:** bloco `if __name__ == "__main__":` para smoke test rápido contra
APIs reais.

---

## 2. Timezone-aware: a decisão estruturante do sistema

### O que é

Todo timestamp no sistema é "timezone-aware" no fuso `America/Sao_Paulo`. É
diferente de um timestamp "naive" (sem fuso), e essa diferença separa
backtests confiáveis de backtests com lookahead silencioso.

Em Python:
```python
naive = pd.Timestamp("2024-05-15 14:00")
# → não sabe em que fuso está; comparações ficam ambíguas

aware = pd.Timestamp("2024-05-15 14:00", tz="America/Sao_Paulo")
# → sabe que está em SP; comparações são corretas
```

### Por que importa (cenário concreto)

São 14h de 15/maio/2024. Sai uma notícia sobre PETR4. O ORQUESTRADOR vai
decidir se compra. Pergunta: ele pode usar o preço de fechamento do dia 15?

**Resposta:** NÃO, porque o fechamento do dia 15 só existe depois das 17h05
(fim do call de fechamento da B3). Às 14h, esse candle ainda não fechou. Usar
esse preço seria ver 3 horas no futuro.

Se o sistema só pensa em "data" (15/maio sem hora), ele não distingue 14h de
18h30 — trata ambos como "o mesmo dia 15" e usa o fechamento. Lookahead.

Com timezone-aware, o sistema sabe que horas são, e a função
`_ultimo_fechamento_disponivel(agora_aware)` decide certo:
- Se `agora < 17h05`, o último fechamento disponível é D-1.
- Se `agora >= 17h05`, é D.

### As quatro decisões parametrizadas

1. **Fuso:** `America/Sao_Paulo`. Notícias vêm de várias fontes (NewsAPI em
   UTC, CVM em horário de Brasília, GDELT em UTC) — converter todas para SP
   elimina ambiguidade.

2. **Corte da B3: 17h05.** Após 17h00 ocorre o call de fechamento (leilão
   final de ~5 minutos). Às 17h05 o candle está definitivamente formado.
   17h30 (fim do after-market) seria mais conservador, mas excluiria
   operações pós-fechamento que afetam o preço final.

3. **Dados sem hora explícita:** assumimos disponibilidade só no fim do dia
   (23h59) da divulgação. Conservador. Vale para IPCA, dados macro que não
   trazem hora exata. **OBS:** para fundamentos vindos da CVM, isso NÃO se
   aplica — usamos a data exata de recebimento (`DT_RECEB`), mais preciso.

4. **Momento de decisão no backtest: 10h.** Como 10h < 17h05, toda decisão
   usa por construção apenas dados até o fechamento de D-1.

### Como defender na banca

> "Adotamos timestamps timezone-aware em America/Sao_Paulo em todo o sistema,
> com corte de 17h05 (fim do call de fechamento da B3) para definir
> disponibilidade do candle diário. Isso elimina o lookahead sutil de horas
> que ocorre quando uma decisão intradia 'enxerga' o fechamento do próprio
> dia antes dele de fato existir. A decisão de operar na abertura (10h)
> garante, por construção, que toda decisão usa apenas dados até o
> fechamento de D-1."

---

## 3. Anti-lookahead — defesa em profundidade

O sistema tem TRÊS camadas de defesa contra lookahead:

**Camada 1 — Parâmetro `data_limite` obrigatório.** Toda função pública
recebe `data_limite`. Não existe método que devolva "o dado mais recente
disponível". Sempre é "o dado até essa data".

**Camada 2 — Corte correto na fonte.** No yfinance, `end` é EXCLUSIVO; somamos
1 dia e cortamos. No BCB, passamos `dataFinal`. **Na CVM, usamos `DT_RECEB <= data_limite`** — a data exata em que o documento se tornou público.

**Camada 3 — Validação defensiva.** Antes de retornar qualquer DataFrame ou
Series, `_assert_no_lookahead` verifica que `timestamp.max() <= data_limite`.
Se passar, levanta `LookaheadError`. Protege contra bugs intermitentes das
fontes externas e contra erros futuros nossos.

### Por que três camadas

Porque lookahead é o erro mais grave possível em backtest. Errar significa
**invalidar todos os resultados** — não "perder pontos", mas "o backtest não
vale nada". Defesa em profundidade é padrão da indústria financeira
justamente porque o custo do erro é catastrófico.

### Como defender na banca

> "Implementamos defesa em profundidade contra lookahead bias. Toda função
> exige data_limite explícito; requisições às fontes externas usam cortes
> apropriados (incluindo DT_RECEB da CVM para fundamentos, que é a data
> exata de divulgação pública); e antes de qualquer retorno, validamos por
> assert que o timestamp máximo não ultrapassa a data_limite, levantando
> LookaheadError em caso de violação."

---

## 4. `get_noticias` — arquitetura em camadas

### O que faz

Recebe `query` (geralmente nome ou ticker da empresa) e `data_limite`.
Retorna lista de objetos `Noticia` publicados entre `data_limite - lookback`
e `data_limite`, ordenados por peso da fonte e recência.

### Por que múltiplas fontes em camadas

Nenhuma fonte sozinha cobre bem todo o período do backtest:

| Fonte | Cobertura | Qualidade | Limitação |
|-------|-----------|-----------|-----------|
| Bloomberg CSV | manual, eventos chave | Máxima (peso 1.0) | Trabalho manual; ToS proíbe scraping |
| GDELT | 2015+, automatizado | Boa (título+URL) | Pouco corpo do artigo |
| Wayback Machine | snapshots datados | Variável | Latência de acesso |
| NewsAPI | últimos 30 dias (free) | Boa, com corpo | Não serve para histórico |

Combinar reduz lacunas. A whitelist de fontes (`bloomberg.com`, `reuters.com`,
`valor.globo.com`, etc.) elimina ruído — só imprensa econômica reconhecida
passa.

### Pesos por fonte

- Bloomberg 1.00 — padrão-ouro do jornalismo financeiro.
- Reuters 0.95, Valor 0.95 — referências sérias e independentes.
- Broadcast 0.90 — agência reconhecida no Brasil.
- Estadão 0.85, InfoMoney 0.75 — boas, com mais opinião.

O peso entra no ranking: ao deduplicar, mantemos a versão da fonte de maior
peso.

### Anti-lookahead nas notícias

`publishedAt` vem das APIs em UTC. Convertemos para `America/Sao_Paulo` antes
de comparar. Qualquer notícia com `publicado_em > data_limite` é descartada.
Para CSVs do Bloomberg, a coluna `data_publicacao` é tratada da mesma forma.

### Limitação documentada

`publishedAt` é o momento em que o **artigo** foi publicado, não em que o
**evento** aconteceu. Uma notícia das 14h pode analisar algo das 10h.
Assumimos `publishedAt` como proxy do momento em que a informação ficou
pública. Documentamos como limitação.

### Como defender na banca

> "Implementamos coleta de notícias em camadas com whitelist rígida de fontes
> reconhecidas, pesos refletindo confiabilidade editorial, e deduplicação
> entre fontes. Bloomberg entra como camada de máxima qualidade via CSV
> manual (a API pública não existe e ToS proíbe scraping); GDELT cobre o
> volume histórico desde 2015. Assumimos publishedAt como proxy do momento
> em que a informação estava pública e documentamos isso como limitação."

---

## 5. `get_precos` — limpeza de dados e duas versões

### O que faz

Recebe ticker (formato `PETR4.SA`), `data_inicio` e `data_limite`. Retorna
DataFrame com OHLCV diário, flags de qualidade e duas versões de preço.

### Por que duas versões de preço

O yfinance, com `auto_adjust=True`, ajusta históricos por **splits** E
**dividendos**. Para estratégias de longo prazo, está correto — você quer o
retorno total. Para holding de **5 dias** (como o JEMPO), ajustar por
dividendo distorce: o dividendo é distribuição de caixa, não movimento de
preço que você captura no intraday.

Por isso coletamos as duas versões: `Close` (ajustado) e `Close_raw` (bruto).
O MATH&ML decide qual usar no retorno de operação. O JOURNAL não impõe —
entrega ambas.

### Tratamento de qualidade

Toda linha recebe uma `flag_qualidade`:

- `volume_zero`: Volume == 0. Dia sem negócio efetivo (suspensão, ticker
  novo). Não apagamos — marcamos.
- `outlier_retorno`: |retorno diário| > 30%. Provável split não ajustado ou
  tick errado.
- `nan`: Close ausente. Erro de coleta. Pode ser preenchido por ffill
  limitado a 3 dias.
- `ok`: tudo certo.

**Por que marcar e não apagar?** Apagar dado silenciosamente é o pior tipo
de "limpeza" — esconde problemas sistemáticos. Marcando, o problema fica
visível e quem consome decide.

### Forward-fill com critério

`ffill` só no Close (não no Volume — inventar volume seria fraude) e só para
gaps curtos (≤3 dias). Gaps maiores merecem atenção, não preenchimento.

### Como defender na banca

> "Os preços são coletados em duas versões — ajustada por split+dividendo e
> bruta — porque o ajuste por dividendo distorce retornos de curto prazo
> característicos da nossa estratégia de 5 dias. Aplicamos uma camada de
> qualidade que marca (não apaga) anomalias. Preenchimento de gaps é
> limitado a 3 dias e só no preço de fechamento — inventar dado é pior que
> ausência declarada."

---

## 6. `get_fundamentals` — CVM como fonte primária (a grande migração)

### Por que migramos do yfinance para a CVM

O yfinance entregava apenas os ~4 trimestres mais recentes — para o backtest
2023-2024 e ainda mais para o treino 2020-2022, os dados frequentemente não
existiam. A solução é integrar com a **CVM, fonte regulatória oficial**, que
publica todos os ITRs (Informe Trimestral) e DFPs (anual) em formato aberto
desde 2011 em `dados.cvm.gov.br/dados/CIA_ABERTA/DOC/`.

Resultado: cobertura histórica completa, fonte primária regulatória,
metodologia idêntica à usada por gestoras profissionais. Vira argumento
forte na banca: *"nossos fundamentos vêm da CVM, mesma fonte que gestores
profissionais usam para análise"*.

### O upgrade do DT_RECEB

Cada documento da CVM tem um campo chamado `DT_RECEB`, que é a **data exata
em que a CVM recebeu o documento**. Isso é a data efetiva em que o balanço
ficou público.

Antes, usávamos um **lag heurístico de 45 dias** (prazo regulatório máximo
da CVM Res. 80). Funcionava como aproximação conservadora, mas era
imprecisa: empresas frequentemente divulgam em 30 dias, não 45.

Com `DT_RECEB`, sabemos exatamente quando cada documento ficou disponível.
O método só retorna fundamentos cujo `DT_RECEB <= data_limite`. Mais
preciso, mais defensável.

### Arquitetura modular

A classe `CVMSource` em `agents/sources/cvm.py` encapsula o trabalho pesado:

```python
class CVMSource:
    def baixar_ano(ano, tipo="ITR") -> Path
    def carregar_demonstrativos(ano) -> dict[str, DataFrame]
    def get_cnpj_por_cd_cvm() -> dict[int, str]
    def get_fundamentals(cd_cvm, data_limite) -> dict | None
    def get_acoes_em_circulacao(cd_cvm, data_limite) -> int | None
```

O `journal.get_fundamentals(ticker, data_limite)` delega a essa classe.
Vantagens da separação: o parser fica isolado (testável, substituível), e a
API pública do JOURNAL não muda quando a fonte interna evolui.

### Detalhes técnicos do parsing CVM (o que pega projetos)

Os arquivos da CVM têm peculiaridades que confundem implementações
desatentas:

- **Encoding `latin-1`** (ISO-8859-1), não UTF-8. Se você ler como UTF-8, os
  acentos viram bagunça (`PETRÓLEO` virou `PETR\xc3\x93LEO`). Erro silencioso.
- **Separador `;`** (ponto e vírgula), não vírgula.
- **Decimal `,`** (vírgula brasileira). Pandas precisa de `decimal=","`.
- **Valores em milhares de reais.** Multiplicar `VL_CONTA` por 1000 para
  obter reais.
- **Versões reapresentadas.** Empresas podem reapresentar balanços (correções,
  republicações). Pegar SEMPRE a versão máxima por `(CNPJ_CIA, DT_REFER)`.
- **Bancos têm estrutura diferente.** Ao invés de BPA (Balanço Ativo) + BPP
  (Balanço Passivo), bancos usam BPB (Balanço Patrimonial de Bancos) com
  contas próprias. O código ramifica por setor.
- **Consolidado vs Individual.** Sempre usar consolidado (`_con_` no nome do
  arquivo) — reflete o grupo econômico, não só a holding.

### Fluxo do `get_fundamentals`

1. Receber `ticker` e `data_limite` (timezone-aware).
2. Buscar `cd_cvm` do ticker em `UNIVERSO_HISTORICO`.
3. `CVMSource.get_fundamentals(cd_cvm, data_limite)`:
   a. Identificar ano(s) necessário(s).
   b. Baixar ZIP da CVM se não estiver em `data/cvm/raw/` (cache disco).
   c. Parsear e cachear em parquet em `data/cvm/processed/`.
   d. Filtrar versão máxima por documento.
   e. Selecionar último com `DT_RECEB <= data_limite`.
   f. Extrair lucro, receita, EBIT, patrimônio, dívida, caixa, ativo total.
4. yfinance buscado APENAS para o campo `setor` (classificação setorial).
5. P/L e P/VP best-effort: combinar (lucro CVM) + (preço yfinance histórico)
   + (ações via CVM capital_social). Qualquer peça que falhe → `None` com
   aviso explícito.
6. Validação `_assert_no_lookahead` e retorno como dataclass `Fundamentals`.

### Periodicidade: TTM para fluxo, point-in-time para estoque

A CVM tem uma peculiaridade que confunde implementações desatentas: ela não
publica trimestres standalone, e sim **acumulados**. O ITR Q1 traz Janeiro a
Março; o ITR Q2 traz Janeiro a Junho (6 meses); o Q3 traz 9 meses acumulados;
a DFP traz o ano inteiro. Isso significa que se você pegar "o último documento
disponível" e usar diretamente, está comparando períodos de tamanhos
diferentes — um Q1 (3 meses) com uma DFP (12 meses). Quebra comparabilidade
no ECON e no MATH&ML.

A solução é o **TTM (Trailing Twelve Months)**, padrão de mercado para P/L,
EV/EBITDA, P/Receita. TTM sempre olha 12 meses contínuos, eliminando
sazonalidade (importante para varejistas, por exemplo, onde o Q4 é muito
mais forte que os outros). É a métrica que Bloomberg, Refinitiv, Itaú BBA
usam — falar "TTM" para a banca é falar a língua do mercado.

A fórmula é:
- **Se o documento mais recente é DFP** (4T/anual): TTM = valor anual direto.
- **Se é um ITR** (Q1/Q2/Q3): TTM = ULTIMO_YTD + (DFP_ano_anterior − PENULTIMO_YTD).

A coluna `PENULTIMO` (mesmo período do ano anterior, sempre incluída pela
CVM em todo ITR) torna esse cálculo barato: só precisamos do ITR atual mais
o DFP do ano anterior, sem rebaixar todos os trimestres.

**Mas isso só vale para itens de fluxo** (receita, EBIT, lucro líquido,
D&A, EBITDA). Itens de balanço (ativo total, patrimônio líquido, caixa,
dívida) são **fotos de um momento** (point-in-time), não fluxos. Patrimônio
em 31/março/2023 é o valor naquele dia; não faz sentido "acumular" patrimônio
de 12 meses. Esses ficam como vêm na CVM.

O dicionário retornado inclui um campo `periodicidade` explícito mapeando
cada métrica para `"TTM"` ou `"point_in_time"`. O ECON e o MATH&ML
consultam esse mapa antes de fazer cálculos derivados (ex: crescimento
de receita YoY usa receita_TTM atual ÷ receita_TTM de 1 ano atrás).

### Best-effort blindado em P/L e P/VP

P/L = Preço × Ações / Lucro. Para calcular historicamente, precisamos:
- Preço na data: vem do yfinance (estável).
- Ações em circulação na data: vem da CVM (`capital_social_cia_aberta_YYYY`),
  mas pode ter gaps.
- Lucro do trimestre: vem da CVM (estável).

Se qualquer peça falhar, retornamos `None` com aviso. **Nunca número
silenciosamente errado.** Princípio geral: dado faltante explícito > dado
errado silencioso.

### Como defender na banca

> "Os fundamentos vêm da CVM — fonte regulatória oficial — através dos
> arquivos abertos de ITR e DFP que cobrem todas as companhias listadas
> desde 2011. Implementamos o módulo agents/sources/cvm.py para baixar,
> parsear e cachear esses dados. O anti-lookahead usa o campo DT_RECEB, que
> é a data exata em que cada documento foi recebido pela CVM, garantindo
> precisão por documento em vez de um lag heurístico. Itens de fluxo
> (receita, lucro, EBITDA) são entregues em TTM — padrão de mercado para
> múltiplos — enquanto itens de balanço (patrimônio, dívida, caixa) ficam
> point-in-time. O dicionário retornado explicita a periodicidade de cada
> métrica para que ECON e MATH&ML não confundam. Métricas que dependem
> de séries instáveis como ações em circulação são tratadas como
> best-effort: se qualquer peça falhar, retornamos None com aviso, nunca
> número incorreto. Para bancos, o código ramifica para tratar a estrutura
> específica do BPB."

---

## 7. `get_macro` — BCB primário, FRED como fallback

### O que faz

Retorna dicionário com Selic diária, Selic meta, câmbio PTAX USD/BRL, IPCA
mensal e IPCA acumulado 12 meses. Tudo cortado em `data_limite`.

### Por que BCB e não FRED

O FRED (Fed de St. Louis) republica dados de bancos centrais do mundo,
incluindo BCB. Mas:

- FRED tem latência de republicação (dias a semanas).
- A série de Selic no FRED é mensal, não diária.
- BCB é a fonte primária — qualquer republicação é mais lenta e menos
  granular.

Por isso usamos **BCB SGS** (Sistema Gerenciador de Séries Temporais) como
primária. API pública, gratuita, sem chave, granularidade diária. Séries:

- Série 11: Selic diária anualizada base 252.
- Série 432: Selic meta (Copom).
- Série 1: PTAX USD/BRL (venda).
- Série 433: IPCA mensal.

### FRED como fallback (e por que existe)

Se a API do BCB cair durante execução, queremos que o sistema não pare.
Implementamos FRED como fallback automático via `try/except`. Redundância
em coleta é barata e o benefício é alto.

### IPCA 12m composto

O IPCA mensal vem em variação percentual. Para acumulado 12m, compomos
multiplicativamente: `(1 + ipca/100).prod() - 1`. Não é soma — é composição,
correto para inflação.

### Como defender na banca

> "Os dados macroeconômicos vêm primariamente do BCB SGS, fonte oficial com
> granularidade diária e sem latência de republicação. FRED foi
> implementado como fallback automático para resiliência operacional."

---

## 8. Helpers auxiliares

### Conversão de timestamps

Função utilitária que converte qualquer input (string, naive, aware) para
timestamp aware em `America/Sao_Paulo`. Centralizar essa conversão garante
que nenhum método "esqueça" do fuso.

### `_assert_no_lookahead`

A guarda final. Recebe DataFrame ou Series, verifica `index.max() <= data_limite`.
Se falhar, levanta `LookaheadError` com contexto (qual método, qual ticker).

### Cache em dois níveis

**Cache geral** (`data/cache/`): pickle, TTL 24h, chave = hash de
`(método, parâmetros)`. Usado para notícias, preços, macro.

**Cache CVM** (`data/cvm/`): mais persistente porque dados regulatórios
raramente são revisados. `data/cvm/raw/` guarda ZIPs (~50MB cada),
`data/cvm/processed/` guarda DataFrames em parquet (rápido de ler). TTL de
90 dias.

**Por que cache?** Backtest de 2 anos × 18 tickers × dados diários =
milhares de chamadas. Sem cache, queima rate limit na primeira execução.
Com cache, a segunda execução é quase instantânea.

### Dataclasses tipadas

`Noticia` e `Fundamentals` são `@dataclass` com tipos. Dois ganhos:
**erro cedo** (campo errado quebra na hora) e **autocomplete/legibilidade**
(`f.roe` é claro; `f["roe"]` exige saber a chave).

A dataclass `Fundamentals` ganhou após a migração CVM o campo
`data_recebimento_cvm`, registrando a `DT_RECEB` para rastreabilidade.

---

## 9. `tickers_ativos` e o tratamento de survivorship

### O problema

Se montar o universo com tickers do IBOV de **hoje** (2026) e rodar backtest
em 2023, está excluindo empresas que faliram ou saíram do índice. Isso infla
artificialmente os resultados. É a primeira coisa que uma banca de gestora
pergunta.

### A solução implementada

`UNIVERSO_HISTORICO` em `config.py` é um dicionário onde cada ticker tem:
- `setor`: classificação setorial padronizada.
- `entrada`: data em que entrou no IBOV (ou None se já estava em 01/01/2019).
- `saida`: data em que saiu (ou None se ainda está em 31/12/2024).
- `confianca`: alta / média / baixa, sobre a precisão das datas.
- `fonte`: de onde tiramos a informação.
- `cd_cvm`: código CVM da companhia (necessário para o lookup de fundamentos).
- `cnpj`: CNPJ formatado (redundante mas útil para verificação).

A função `tickers_ativos(data_aware)` retorna apenas tickers cuja janela
`[entrada, saida]` contém a data. Toda iteração sobre o universo (treino,
backtest, `get_retornos_setor`) usa essa função — nunca lista estática.

### O caso AMER3 (Americanas)

Americanas estava no IBOV até começo de 2023. Em 11/janeiro/2023, anunciou
"inconsistências contábeis" de R$ 20 bilhões. Em 12/janeiro pediu recuperação
extrajudicial. A ação despencou 80%+ em dias.

Com `saida = 2023-01-12`, AMER3 está no universo até o último dia útil antes
da quebra e some logo depois. **Argumento ouro na banca:** implementamos
tratamento de survivorship inclusive com casos emblemáticos como Americanas,
para demonstrar que o sistema processa empresas que quebraram, não apenas
sobreviventes.

### Período coberto: 2019-2024

Por que 2019 e não 2020? Porque features de momentum usam **252 dias úteis
(~1 ano)** de retorno passado. Para o treino começar efetivamente em 2020,
precisamos de dados desde 2019. O universo precisa refletir composição do
IBOV em todo esse período, não só do backtest.

### Como defender na banca

> "Implementamos universo histórico com membership por data — cada ticker
> tem datas de entrada e saída do IBOV, e a qualquer momento da simulação
> só consideramos tickers ativos naquela data. Cobrimos 2019-2024 (warmup +
> treino + backtest). Incluímos casos emblemáticos como AMER3 (saída
> 12/jan/2023, após pedido de RJ) para que o sistema processe empresas que
> quebraram, não só as sobreviventes. Cada entrada tem também o cd_cvm para
> o lookup de fundamentos na CVM."

---

## 10. Limitações honestas (e como apresentar)

A banca pergunta as limitações. Negar é pior que admitir. Veja como
apresentar cada uma:

**1. NewsAPI gratuito cobre só 30 dias.** "Por isso a arquitetura em camadas:
GDELT para o histórico (2015+), Wayback Machine para snapshots datados, e
Bloomberg via CSV manual para os eventos mais importantes. NewsAPI
complementa o período recente."

**2. Bloomberg sem API e ToS proibindo scraping.** "Respeitamos os termos de
uso. A integração é manual via CSV — incompatível com automação total, mas
viável para os ~50 a 100 eventos mais relevantes do período, onde a
Bloomberg agrega mais valor."

**3. `publishedAt` ≠ momento do evento.** "Assumimos publishedAt como proxy
do momento em que a informação ficou pública. Em alguns casos uma notícia
analisa um fato anterior; documentamos como limitação. Em produção, seria
mitigado por NER (extração de entidades temporais)."

**4. Universo histórico do IBOV com gaps.** "A B3 publica composições por
rebalanceamento sem API direta. Cobrimos com confiança alta os casos
emblemáticos e marcamos com TODO os pontos que exigem verificação manual
adicional."

**5. Ações em circulação com gaps na CVM.** "O dataset capital_social da
CVM tem cobertura boa mas não perfeita — em particular para empresas em
processos de reorganização (cisões, fusões). P/L e P/VP que dependem desse
dado são best-effort: se a peça falhar, retornamos None com aviso. Nunca
inventamos número."

**6. Bancos têm estrutura BPB diferente.** "Implementamos ramificação por
setor no parser da CVM — bancos do universo (ITUB4, BBDC4) usam o esquema
BPB ao invés de BPA+BPP. As métricas finais entregues são padronizadas."

---

## 11. Glossário rápido

- **Lookahead bias:** usar no presente informação que só estaria disponível
  no futuro. Erro fatal em backtest.
- **Survivorship bias:** usar apenas empresas que sobreviveram, ignorando
  as que quebraram. Infla retornos.
- **Walk-forward validation:** retreinar o modelo periodicamente,
  incorporando dados novos sem voltar ao passado.
- **Timezone-aware:** timestamp que carrega informação de fuso horário.
- **OHLCV:** Open, High, Low, Close, Volume — colunas básicas de preço.
- **PTAX:** taxa de câmbio oficial divulgada pelo BCB, calculada como média
  ponderada dos negócios do dia.
- **ITR:** Informe Trimestral, demonstrativos trimestrais que empresas
  listadas no Brasil são obrigadas a divulgar à CVM.
- **DFP:** Demonstrações Financeiras Padronizadas, equivalente anual do ITR.
- **DT_RECEB:** campo da CVM com a data exata em que ela recebeu o
  documento — data efetiva de divulgação pública.
- **CD_CVM:** código numérico que a CVM atribui a cada companhia listada.
  Chave primária para lookup de dados.
- **BPA / BPP / BPB:** Balanço Patrimonial Ativo / Passivo / de Bancos.
  Bancos usam BPB ao invés de BPA+BPP.
- **Consolidado vs Individual:** consolidado agrega o grupo econômico;
  individual é só a holding. Sempre usar consolidado.
- **Forward-fill (ffill):** preencher valores ausentes com o último conhecido.
- **Dataclass:** estrutura Python tipada que substitui dicionários soltos.
- **Best-effort:** tentativa que pode falhar; em falha, retorna resultado
  degradado com indicação clara, não erro silencioso.
- **TTM (Trailing Twelve Months):** janela móvel de 12 meses contínuos
  usada para suavizar sazonalidade em métricas de fluxo (receita, lucro).
  Padrão de mercado para P/L e múltiplos de avaliação.
- **Point-in-time:** valor de um instante específico (como patrimônio em
  31/março), oposto a fluxo acumulado. Aplica-se a itens de balanço.
- **PENÚLTIMO (coluna CVM):** coluna que todo ITR traz, com o acumulado do
  mesmo período do ano anterior. Permite cálculo de TTM sem baixar dados
  adicionais.
- **Parquet:** formato binário comprimido para DataFrames, mais rápido que
  CSV para leitura.

---

## 12. Perguntas que a banca pode fazer (com respostas)

**P: Por que escolheram Python e essas bibliotecas?**
R: Python pelo ecossistema científico maduro. pandas para séries temporais,
yfinance para preços (open source), scikit-learn para ML. Pyarrow para
cache binário eficiente em parquet. Decisões pragmáticas alinhadas com o
ecossistema padrão da indústria de quant research.

**P: Qual a maior limitação metodológica do JOURNAL?**
R: A cobertura completa do universo histórico do IBOV — a B3 não tem API
limpa para isso, então cobrimos com confiança alta os casos emblemáticos e
marcamos TODOs para verificação manual onde resta incerteza.

**P: Como evitam lookahead em granularidade fina?**
R: Timezone-aware completo, corte de 17h05 da B3, decisão sempre na
abertura (10h) — toda decisão usa por construção apenas o fechamento de
D-1. Validação defensiva `_assert_no_lookahead` em toda saída. Para
fundamentos, usamos `DT_RECEB` da CVM, que é a data exata de recebimento
do documento — mais preciso que lag heurístico.

**P: Por que migraram para a CVM?**
R: O yfinance entregava apenas os ~4 trimestres mais recentes, o que era
insuficiente para um backtest histórico em 2023-2024 com treino em
2020-2022. A CVM é a fonte regulatória oficial, com cobertura completa
desde 2011, e ainda agrega rigor metodológico pelo campo DT_RECEB. É a
mesma fonte usada por gestoras profissionais.

**P: Como justificam o uso de DT_RECEB em vez do prazo regulatório?**
R: A Resolução CVM 80 estabelece um prazo MÁXIMO de 45 dias para
divulgação de ITR. Mas empresas frequentemente divulgam antes — usar o
prazo máximo como lag fixo descarta sinal válido. O DT_RECEB é a data
exata por documento, mais precisa e tão (ou mais) defensável.

**P: O que acontece se uma das fontes de notícia retornar lixo?**
R: A whitelist rígida descarta fontes desconhecidas. Dedup por título
elimina réplica. Pesos por fonte priorizam qualidade. Em casos extremos,
o erro propaga como `DadoIndisponivel`, não como dado silencioso.

**P: Por que esses pesos para as fontes (1.0, 0.95, etc.)?**
R: Refletem credibilidade editorial reconhecida. Bloomberg e Reuters
como padrões internacionais; Valor e Broadcast como referências
brasileiras de mercado. Pesos são parâmetros revisáveis; o ranking
relativo é conservador.

**P: Como o sistema reagiria a um evento como Americanas em produção?**
R: O ECON receberia notícias de alta relevância de Bloomberg/Reuters/Valor
sobre o caso. Geraria score fortemente negativo. O ORQUESTRADOR não abre
posição em ações com score abaixo de -0.3; se houvesse posição aberta, o
critério "score invertido" dispararia saída. Validamos esse fluxo
treinando com AMER3 incluída no universo até a data da quebra.

**P: Por que entregam TTM em vez de trimestre standalone?**
R: A CVM publica acumulados (Q2 = 6 meses, Q3 = 9 meses, DFP = ano), não
trimestres standalone. Usar valores brutos quebraria comparabilidade entre
documentos de períodos diferentes. TTM (Trailing Twelve Months) é o padrão
de mercado para P/L e múltiplos porque sempre cobre 12 meses contínuos,
eliminando sazonalidade — importante especialmente para varejo. Bloomberg,
Refinitiv e o mercado profissional brasileiro usam TTM. Mantemos itens de
balanço como point-in-time (patrimônio em 31/março é o valor naquele dia,
não acumulado) e usamos a coluna PENÚLTIMO da CVM para calcular TTM sem
download adicional.

**P: Como tratam empresas que mudaram de denominação ou cd_cvm?**
R: O UNIVERSO_HISTORICO documenta cada caso com fonte e nível de confiança.
Para reorganizações societárias (cisões, fusões), tratamos cada entidade
do ponto de vista do investidor — ações da Americanas após a RJ continuam
sendo AMER3 na B3 e CD_CVM correspondente na CVM, então o lookup
funciona normalmente. Casos mais complexos (mudança de ticker) estão
documentados.

**P: O que aconteceria se a CVM mudasse o formato dos arquivos?**
R: O parser está isolado em `agents/sources/cvm.py`. Mudança de formato
exigiria atualização de uma classe específica sem mexer no resto do
sistema. Mantemos cache local em parquet, então a refatoração pode ser
feita sem urgência operacional.

**P: Como garantem que o parser da CVM lê os números corretamente?**
R: Encoding latin-1 (não UTF-8), separador `;`, decimal `,`, valores em
milhares de reais multiplicados por 1000. Versões reapresentadas são
filtradas para a máxima por (CNPJ, data referência). Bancos usam
estrutura BPB com ramificação no código. Tudo isso é coberto por testes
unitários em tests/test_cvm.py contra arquivos reais da CVM.
