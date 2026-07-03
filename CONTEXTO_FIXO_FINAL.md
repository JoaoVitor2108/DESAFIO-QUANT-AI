# CONTEXTO FIXO — JEMPO (Desafio Quant AI 2026, Itaú Asset)

> Registro de decisões consolidadas e débitos honestos do sistema. Este arquivo
> foi criado no Round 5-close (não existia antes); começa pela seção do MATH&ML.
> Manuais detalhados por agente: `MANUAL_JOURNAL.md`, `MANUAL_ECON.md`,
> `RELATORIO_MATHML.md`. Runbook operacional do GDELT: `docs/RUNBOOK_GDELT.md`.

---

## DECISÕES CONSOLIDADAS DO MATH&ML

### Arquitetura de pré-fetch (contra n+1)
- `construir_dataset` opera em DUAS FASES: (1) `_prefetch` chama cada fonte do
  JOURNAL uma vez cobrindo o range inteiro e enche o `_DatasetCache` (precos,
  ibov, macro, fundamentos); (2) `_montar_features` e `_alvo` lêem SÓ do cache —
  zero I/O na montagem.
- Motivo: o padrão anterior (chamar o journal por (ticker × dia × fonte)) gerava
  ~10.000 chamadas por run com 8 tickers × 1 ano; o BCB SGS / yfinance
  rate-limitavam após centenas e o run travava.
- Teste guardião `test_prefetch_chamadas_unicas` conta chamadas ao journal via
  `CountingFakeJournal` — se alguém reintroduzir n+1, o teste pega. Limites:
  `get_precos ≤ n+1`, `get_macro ≤ 1`, `get_retornos_setor == 0`,
  `get_fundamentals ≤ n × (n_meses + folga)`.

### Mock do ECON calcula z(y) in-memory
- Bug latente descoberto no Round 5: `make_econ_mock(amostra_calibracao=ds)`
  recomputava o z(y) cross-sectional chamando o journal (n+1 dentro do mock,
  ANTES de `set_cache` rodar). Fazia o smoke travar minutos só na calibração.
- Solução: a `amostra_calibracao` já traz a coluna `y` de todo o cross-section,
  então o z(y) é calculado via `groupby("data")["y"]` em memória. Zero I/O na
  calibração do mock; mesmos valores (o y da amostra veio do mesmo `_alvo`).
- Consequência: `set_cache` só é necessário para os caminhos DEPOIS da
  calibração (montagem de features com score_econ). A calibração é
  auto-suficiente quando a amostra traz `y`; fixtures sem `y` seguem o caminho
  legacy via journal.

### Fundamentos em dois modos de coleta
- indexed (produção, dados com `data_recebimento_cvm` populado):
  `_prefetch_fundamentos` chama `get_fundamentals` em âncoras mensais e deduplica
  por `DT_RECEB`; depois `_fund_em_t` seleciona o último publicado ≤ t. Chamadas:
  O(n_tickers × n_meses).
- passthrough (fixtures sem `DT_RECEB`): cai no caminho legacy de chamada por-t.
  Usado apenas em unit tests com FakeJournal simplificado — nunca em produção.
- O teste guardião cobre o modo indexed (comportamento de produção).

### Cache como fast-path puro
- `_DatasetCache` é fast-path: se algum acessor pede ticker/data fora do cache
  (ex.: mock com `universo=None`), cai automaticamente no journal real. Não
  levanta `KeyError`. Robustez em troca de performance nos casos-limite.
- Consequência: o cold-cache (primeira execução) leva ~70s para 8 tickers × 1 ano
  por causa da latência do yfinance; reruns são ~5s via disk-cache do JOURNAL
  (TTL 24h). No backtest oficial (18 tickers × 6 anos), cold-cache pode chegar a
  ~10 min. É latência real de rede, não bug.

### Contratos consumidos (nomes reais, divergentes do spec)
- `get_macro(t)` já devolve a SÉRIE inteira warmup→t → uma chamada basta (não
  precisa reconstruir por âncoras). Reconstruir por âncoras semanais + ffill
  MUDARIA `selic_var_21d`/`cambio_var_21d` — proibido (quebra invariância).
- `get_retornos_setor` devolve dict agregado e não é consumido por nenhuma
  feature → fora do cache.
- Nomes reais: `LookaheadError`, `self._econ_mock`, `FUSO` (constante).

### Correções pós-revisão (bugs achados em auditoria)
- **z(y) com ddof consistente:** o fast-path de calibração do mock (`amostra`
  com coluna `y`) usava `pandas.Series.std()` (ddof=1) enquanto o runtime
  (`_z_info`) usa `numpy.std()` (ddof=0). O `alpha` calibrado não reproduzia o
  IC injetado no dataset (erro ∝ 1/|U|: ~3% p/ 18 tickers, ~22% p/ o smoke de
  3). Corrigido para `std(ddof=0)`; teste `test_mock_fastpath_calibracao_consistente`
  fecha o buraco (o fast-path antes não tinha teste — os testes usam amostra
  sem `y`, que cai no slow-path já consistente).
- **GAP_evento honesto:** `ic_evento` sai com ≥3 eventos, mas os baselines de
  evento só com >30. No meio-termo, `_max_baseline` devolvia 0.0 e o GAP
  superestimava o edge. Agora devolve NaN quando não há baseline do sufixo →
  `GAP_evento=NaN` (comparação inviável); `GAP_total` é imune (sempre tem
  B3_intercepto=0.0).

### Débito conhecido (latente, hoje inócuo)
- **Re-corte de IPCA no cache path:** `_macro_em_t` corta as séries do cache por
  `index <= t`. Para IPCA/ipca_12m isso usa o mês de referência, não a data de
  disponibilidade (~11d de lag) que o JOURNAL aplica — potencial lookahead. É
  inócuo HOJE porque IPCA não é feature (macro = selic_nivel, selic_var_21d,
  cambio_var_21d) e `_assert_no_lookahead` só roda no path sem cache. Se IPCA
  virar feature, cortar por disponibilidade no cache path ANTES de usá-la.

---

## DÉBITOS E LIÇÕES DO GDELT

### Rate limit por IP (backoff)
- O GDELT impõe rate limit por IP não-documentado, disparado por rajadas (rodar
  pytest várias vezes já ativa). O `GDELTSource` trata 429/503 com backoff
  exponencial (60s → 600s, 5 tentativas, ~15min no pior caso) e levanta
  `GDELTRateLimitedError` em vez de degradar silenciosamente (antes devolvia `[]`,
  escondendo "não consegui consultar" como "não houve notícia").
- `GDELTUnavailableError` para timeout/5xx/JSON inválido. O `JournalAgent` captura
  as duas, loga warning e incrementa `gdelt_degradado_count` (exposto em
  `health_check`). Throttle base configurável por `GDELT_THROTTLE_SECONDS`
  (default 5s; use 12–15s em backtest). Cache pickle nunca invalidado por erro.
- Runbook operacional: `docs/RUNBOOK_GDELT.md`.
