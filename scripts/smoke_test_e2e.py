"""Smoke test end-to-end do JEMPO — 4 agentes reais em cadeia, dados reais.

Objetivo
--------
Rodar o sistema inteiro (JOURNAL + ECON-mock + MATH&ML + ORQUESTRADOR) com
dados reais numa janela histórica, verificando que os agentes conversam entre
si sem exceção e que o pipeline de dados (yfinance, CVM, BCB, cache local)
funciona ponta a ponta. Serve de demo executável para a banca.

Escopo
------
É SMOKE TEST, não backtest. Verifica integração e plausibilidade das decisões.
NÃO calcula P&L, retorno, Sharpe, drawdown real, custos, Monte Carlo ou
qualquer métrica de performance — isso é do PROGRAM (próximo agente). Aqui o
PROGRAM é trivial: equity constante e execução simulada pelo fechamento do
JOURNAL no dia de execução.

Escolhas travadas
-----------------
- Janela: 1º mar → 30 abr 2024 (~40 dias úteis), dentro do range do OOS
  2024-2025. Valida os dados históricos antes da rodada oficial.
- ECON: mock estruturado `make_econ_mock` (agents/math_ml.py) no modo "meta"
  (IC alvo 0.15). Grátis, determinístico, sem API. O mock é um callable
  `f(ticker, data_limite)`; o ORQUESTRADOR espera `.avaliar(...)` → usamos um
  adapter minimal LOCAL (não modificamos o módulo).
- PROGRAM mock: equity constante em R$ 100.000,00 durante todo o loop.
  Execução simulada usa o fechamento do JOURNAL no `data_execucao` da Ordem.
  Zero P&L, zero custos.
- Universo: `config.tickers_ativos(data)` real, cardinalidade natural (22-24).

Rode com: `python scripts/smoke_test_e2e.py`
"""

import sys
from pathlib import Path

# adiciona a raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config

# ── Constantes ────────────────────────────────────────────────────────────────

FUSO = "America/Sao_Paulo"

DATA_INICIO = pd.Timestamp("2024-03-01", tz=FUSO)
DATA_FIM = pd.Timestamp("2024-04-30 23:59:59", tz=FUSO)
EQUITY_INICIAL = 100_000.0

# Modo do mock do ECON: "meta" = IC alvo 0.15 (§3 das escolhas travadas).
MODO_MOCK = "meta"
IC_ALVO = 0.15


def main() -> None:
    print("smoke test iniciando")

    # sanidade do ambiente: universo do primeiro dia útil da janela
    u = config.tickers_ativos(DATA_INICIO)
    print(f"sanidade: tickers_ativos({DATA_INICIO.date()}) = {len(u)} tickers")

    # 1. Instanciar JOURNAL real
    # TODO(bloco2): journal = JournalAgent(...)  # config default do projeto

    # 2. Instanciar mock do ECON com adapter para .avaliar(ticker, data_limite)
    # TODO(bloco2): handle = make_econ_mock(journal, ic_alvo=IC_ALVO,
    #                                       amostra_calibracao=<dataset treino>)
    # TODO(bloco2): econ = _EconMockAdapter(handle)

    # 3. Instanciar MATH&ML real e treinar (não há modelo em disco; sem persist.)
    # TODO(bloco2): math_ml = MathMLAgent(journal=journal, econ_mock=handle)
    # TODO(bloco2): ds = math_ml.construir_dataset(tickers, treino_ini, treino_fim)
    # TODO(bloco2): math_ml.treinar(ds)   # <-- PARAR e perguntar antes de treinar

    # 4. Instanciar ORQUESTRADOR real com esses agentes
    # TODO(bloco2): orq = OrchestratorAgent(journal, econ, math_ml, config,
    #                                       tickers_ativos=config.tickers_ativos)

    # 5. Loop diário sobre DATA_INICIO..DATA_FIM (pd.bdate_range)
    # TODO(bloco3): for dia in dias_uteis:
    #   a. dec = orq.decidir(dia, equity_hoje=EQUITY_INICIAL)
    #   b. print do resumo do dia (§5)
    # TODO(bloco4):
    #   c. para cada Ordem: preço de exec do JOURNAL -> orq.notificar_execucao
    #   d. para cada FechamentoOrdem: orq.notificar_fechamento
    # TODO(bloco5):
    #   e. coletar métricas de auditoria (contadores + anti-lookahead)

    # 6. Print do status() final
    # TODO(bloco3/4): print(orq.status())

    # 7. Print do sumário de auditoria (§6)
    # TODO(bloco5): print sumário (dias, ordens, fechamentos, setores, anti-lookahead)


if __name__ == "__main__":
    main()
