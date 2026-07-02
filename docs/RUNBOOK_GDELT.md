# Runbook GDELT — operação e troubleshooting

O GDELT impõe **rate limit por IP** (não documentado), disparado por rajadas de
chamadas. Rodar `pytest tests/` várias vezes seguidas já é suficiente para ativá-lo.
O `GDELTSource` trata 429/503 com **backoff exponencial** (60s → 120 → 240 → 480 →
~600s, 5 tentativas, ~15min no pior caso) e **levanta exceção tipada** em vez de
degradar em silêncio:

- `GDELTRateLimitedError` — 429/503 persistente após o backoff.
- `GDELTUnavailableError` — timeout/5xx/DNS/JSON inválido.

O `JournalAgent` captura ambas, loga warning e incrementa `gdelt_degradado_count`
(exposto em `health_check()`), seguindo com as outras camadas (Bloomberg CSV / NewsAPI).

---

## Sintoma: muitos `GDELT 429 (tentativa N/5)` no log

Você está consumindo a cota por IP. Causas comuns:
1. Rodadas frequentes de `pytest` no mesmo dia.
2. Backtest sem throttling conservador.
3. IP compartilhado (VPN, CGNAT) com outros usuários do GDELT.

### Ação imediata
1. Parar runs do GDELT por algumas horas (idealmente 24h).
2. Próximo run com throttle conservador: `GDELT_THROTTLE_SECONDS=15 python ...`
3. Conferir a magnitude: `journal.health_check()["gdelt_degradado_count"]`.

---

## Sintoma: `GDELTRateLimitedError` levantada em runtime

Esgotaram as 5 tentativas (~15min de espera total). É degradação **real** — o
JOURNAL seguiu com Bloomberg CSV / NewsAPI, mas houve queda na cobertura de
notícias. Em backtest: revisar o período afetado para decidir se replica a janela
com cache pré-populado.

---

## Pre-populating do cache para backtest histórico

O cache pickle (TTL 24h por `query + data_limite`) é a chave para evitar 429 em
runs grandes. Estratégia:

1. Fatie o backtest por ano e use throttle conservador:
   ```
   GDELT_THROTTLE_SECONDS=12 python scripts/preencher_cache_gdelt.py \
       --inicio 2020-01-01 --fim 2020-12-31
   ```
2. Se tomar 429 persistente no meio, espere e retome — o cache preserva o que já
   foi baixado (o backoff nunca invalida pickle existente).
3. Após popular: rode o backtest normal; a maioria das chamadas vem do cache.

---

## Variáveis de ambiente

| Variável | Default | Uso |
|---|---|---|
| `GDELT_THROTTLE_SECONDS` | `5` | espaçamento mínimo entre requisições; subir p/ 12–15 em backtest |

---

## health_check após o run

Sempre conferir:
- `gdelt_degradado_count == 0` → run limpo.
- `gdelt_degradado_count > 0` → houve degradação; investigar magnitude vs.
  universo/período afetado antes de confiar nos resultados.
