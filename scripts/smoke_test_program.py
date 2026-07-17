"""Smoke test de integração do PROGRAM (BacktestEngine) — Etapa 1, Subetapa 1.3.

Objetivo
--------
Comprovar que o `BacktestEngine` conversa em PRODUÇÃO com 3 dos 4 agentes reais
(JOURNAL + MATH&ML + ORQUESTRADOR) através da fronteira tz-aware, sem crash. O
ECON entra como mock estruturado (`make_econ_mock`, reusado do smoke_test_e2e)
para não gastar API key.

NÃO é backtest oficial. NÃO julga performance financeira (Sharpe/MDD/Monte Carlo
são Etapa 3+). Só roda o loop diário do engine sobre dados reais e imprime um
resumo. Sai com exit 0 sempre que rodar sem crash — inclusive com trades = 0 ou
capital final < inicial.

Setup (espelha `scripts/smoke_test_e2e.py`)
-------------------------------------------
- Monkeypatch do TTL do `_DiskCache` do JOURNAL (janela histórica; preço passado
  não muda, honrar cache é seguro e determinístico).
- Bootstrap de dataset (mock não-calibrado) só para extrair a amostra (data,
  ticker, y) usada na calibração do mock (IC alvo 0.15).
- MATH&ML real treinado (GBM) na janela 2022-2023 (mantém mar-jun/2024 OOS).
- ORQUESTRADOR real com `config.tickers_ativos` (survivorship por data).

Rode com: `python -m scripts.smoke_test_program`  (ou `python scripts/...`).
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

# Raiz do projeto no path (executável tanto por -m quanto direto).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from agents import journal as journal_mod
from agents.journal import JournalAgent
from agents.math_ml import MathMLAgent, make_econ_mock
from agents.orchestrator import OrchestratorAgent, OrchestratorConfig
from backtest.engine import BacktestConfig, BacktestEngine

# Reuso do adapter do ECON do smoke_test_e2e (NÃO reimplementar o mock do zero).
from scripts.smoke_test_e2e import _EconMockAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_program")

# ── Monkeypatch do TTL do cache do JOURNAL (mesma justificativa do e2e) ────────
# O relógio do ambiente está >24h à frente de todo o cache já gravado; sem este
# patch cada get viraria download ao vivo (não-determinístico). Idempotente com o
# patch aplicado no import de smoke_test_e2e.
journal_mod._DiskCache.TTL = timedelta(days=3650)

# ── Constantes do smoke ───────────────────────────────────────────────────────

FUSO = "America/Sao_Paulo"

# Janela do backtest (datas NAIVE — convenção do PROGRAM; o engine localiza em SP
# na fronteira). ~85 pregões B3.
DATA_INICIO = pd.Timestamp("2024-03-01")
DATA_FIM = pd.Timestamp("2024-06-30")

# Janela de treino do MATH&ML (tz-aware SP, como o e2e). 2022-2023 → OOS em 2024.
TREINO_INICIO = pd.Timestamp("2022-01-01", tz=FUSO)
TREINO_FIM = pd.Timestamp("2023-12-31 23:59:59", tz=FUSO)

IC_ALVO = 0.15  # modo "meta" do mock do ECON (§3 das escolhas travadas do e2e)
CAPITAL_INICIAL = 100_000.0

_SEP = "=" * 72
_SUB = "-" * 72


# ── Montagem dos agentes reais (bootstrap + treino, espelha o e2e) ────────────


def _montar_agentes() -> tuple[JournalAgent, OrchestratorAgent]:
    """Constrói JOURNAL real, MATH&ML real treinado e ORQUESTRADOR real com o
    mock estruturado do ECON calibrado (IC≈0.15). Mesma sequência do e2e."""
    journal = JournalAgent()

    # Bootstrap: dataset com mock não-calibrado (α=0) só para extrair (data,
    # ticker, y) — o y (retorno forward) independe de α.
    logger.info("bootstrap: construir_dataset %s..%s (calibração)",
                TREINO_INICIO.date(), TREINO_FIM.date())
    boot_handle = make_econ_mock(journal, ic_alvo=0.0)
    boot_agent = MathMLAgent(journal=journal, econ_mock=boot_handle)
    ds_boot = boot_agent.construir_dataset(None, TREINO_INICIO, TREINO_FIM)
    amostra = ds_boot[["data", "ticker", "y"]].copy()

    # Mock calibrado (meta, IC≈0.15) reaproveitando a amostra.
    econ_handle = make_econ_mock(journal, ic_alvo=IC_ALVO, amostra_calibracao=amostra)
    econ = _EconMockAdapter(econ_handle)
    logger.info("mock ECON: ic_alvo=%.2f ic_realizado=%+.3f alpha=%.3f",
                econ_handle.ic_alvo, econ_handle.ic_realizado, econ_handle.alpha)

    # MATH&ML real treinado (rebuild bate no cache; depois treina o GBM).
    math_ml = MathMLAgent(journal=journal, econ_mock=econ_handle)
    ds_train = math_ml.construir_dataset(None, TREINO_INICIO, TREINO_FIM)
    math_ml.treinar(ds_train, data_treino_fim=TREINO_FIM)

    # Serve-time volta a buscar preços ao vivo (disk cache cobre 2024).
    econ_handle.set_cache(None)

    orq = OrchestratorAgent(
        journal=journal,
        econ=econ,
        math_ml=math_ml,
        config=OrchestratorConfig(),
        tickers_ativos=config.tickers_ativos,
    )
    logger.info("4 agentes prontos (JOURNAL + ECON-mock + MATH&ML + ORQUESTRADOR)")
    return journal, orq


# ── Impressão do resumo (8 blocos) ────────────────────────────────────────────


def _bloco_config(cfg: BacktestConfig) -> None:
    print(_SEP)
    print("1) CONFIGURAÇÃO")
    print(_SUB)
    print(f"  Janela          : {DATA_INICIO.date()} → {DATA_FIM.date()}")
    print(f"  Capital inicial : R$ {cfg.capital_inicial:,.2f}")
    print(f"  Corretagem      : {cfg.corretagem:.3%}  |  Slippage: {cfg.slippage:.3%}")
    print(f"  Custo por perna : {cfg.custo_perna:.3%} (sobre o notional)")
    print(f"  Sizing          : {cfg.sizing_pct:.0%} do equity corrente")
    print(f"  Stop / Take     : -{cfg.stop_pct:.0%} / +{cfg.take_pct:.0%} do preço de entrada")


def _bloco_contagens(res) -> None:
    print(_SEP)
    print("2-4) CONTAGENS")
    print(_SUB)
    print(f"  n_dias_uteis (calendário BMF): {res.n_dias_uteis}")
    print(f"  n_trades total               : {res.n_trades}")
    motivos = Counter(res.trades["motivo"]) if res.n_trades else Counter()
    print("  Decomposição por motivo:")
    for motivo in ("stop", "take", "prazo", "reversao", "fim_backtest"):
        print(f"    - {motivo:<13}: {motivos.get(motivo, 0)}")


def _bloco_capital(res, elapsed: float) -> None:
    print(_SEP)
    print("5-6) CAPITAL E TEMPO")
    print(_SUB)
    ret_pct = (res.capital_final / res.config.capital_inicial - 1) * 100
    print(f"  Capital final  : R$ {res.capital_final:,.2f}")
    print(f"  Retorno bruto  : {ret_pct:+.2f}% vs capital inicial "
          "(sem Sharpe/MDD — Etapa 3)")
    # Sanidade de contabilidade: equity_diario.iloc[-1] (MTM do último pregão,
    # posições ainda abertas) vs capital_final (pós-liquidação R8, com custo de
    # saída). Diferem pelo custo de saída forçada se havia posição aberta no fim.
    if len(res.equity_diario):
        ult_equity = float(res.equity_diario.iloc[-1])
        dif = ult_equity - res.capital_final
        print(f"  equity_diario[-1] (MTM último pregão): R$ {ult_equity:,.2f}")
        print(f"  Δ (equity[-1] − capital_final)       : R$ {dif:,.2f} "
              "(= custo de saída forçada R8, se houve posição aberta no fim)")
    print(f"  Tempo total    : {elapsed:.1f}s")


def _bloco_avisos(res) -> None:
    print(_SEP)
    print("7) TOP-5 AVISOS POR TIPO")
    print(_SUB)
    if not res.avisos:
        print("  (nenhum aviso)")
        return
    por_tipo: dict[str, list[dict]] = defaultdict(list)
    for a in res.avisos:
        por_tipo[a["tipo"]].append(a)
    ordenados = sorted(por_tipo.items(), key=lambda kv: len(kv[1]), reverse=True)
    for tipo, itens in ordenados[:5]:
        ex = itens[0]
        data = ex["data"].date() if hasattr(ex["data"], "date") else ex["data"]
        print(f"  {tipo:<26} x{len(itens):<4} ex.: [{data} {ex['ticker']}] "
              f"{ex['detalhe']}")


def _bloco_amostra_trades(res) -> None:
    print(_SEP)
    print("8) AMOSTRA DE TRADES (1º / meio / último por data de saída)")
    print(_SUB)
    if not res.n_trades:
        print("  (nenhum trade)")
        return
    cols = ["ticker", "motivo", "data_entrada", "data_saida",
            "preco_entrada", "preco_saida", "pnl_liquido"]
    df = res.trades.sort_values("data_saida").reset_index(drop=True)
    idxs = sorted({0, len(df) // 2, len(df) - 1})
    for i in idxs:
        t = df.loc[i, cols]
        print(f"  [{i}] {t['ticker']:<10} {t['motivo']:<12} "
              f"{pd.Timestamp(t['data_entrada']).date()} → "
              f"{pd.Timestamp(t['data_saida']).date()}  "
              f"entrada={t['preco_entrada']:.2f} saida={t['preco_saida']:.2f}  "
              f"pnl_liq={t['pnl_liquido']:+.2f}")


def _imprimir_resumo(res, elapsed: float) -> None:
    print()
    print(_SEP)
    print("SMOKE TEST PROGRAM — RESUMO (integração BacktestEngine + agentes reais)")
    _bloco_config(res.config)
    _bloco_contagens(res)
    _bloco_capital(res, elapsed)
    _bloco_avisos(res)
    _bloco_amostra_trades(res)
    print(_SEP)
    print("smoke test PROGRAM concluído (sem crash)")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    print("smoke test PROGRAM iniciando (integração ponta-a-ponta)")
    t0 = time.perf_counter()

    journal, orq = _montar_agentes()
    engine = BacktestEngine(journal, orq, BacktestConfig())

    # Exceções INESPERADAS (ValueError de tz, RuntimeError da invariante,
    # LookaheadError) NÃO são capturadas: devem propagar e derrubar o smoke —
    # são bugs de engine/integração, não bordas de dados.
    res = engine.rodar_backtest(DATA_INICIO, DATA_FIM)

    elapsed = time.perf_counter() - t0
    _imprimir_resumo(res, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
