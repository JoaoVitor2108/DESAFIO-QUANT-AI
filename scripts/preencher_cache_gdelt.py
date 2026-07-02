"""Pre-populating do cache do GDELT para backtest histórico.

Itera (ticker, dia útil) no range pedido chamando JournalAgent.get_noticias, o
que força o cache pickle do GDELT (TTL 24h por query+data_limite). Linhas já
cacheadas são instantâneas. Em GDELTRateLimitedError, aborta com instrução clara
— o que já foi baixado fica preservado no cache; basta esperar e retomar.

Uso:
    GDELT_THROTTLE_SECONDS=12 python scripts/preencher_cache_gdelt.py \\
        --inicio 2020-01-01 --fim 2020-12-31 [--tickers PETR4.SA VALE3.SA]

Ver docs/RUNBOOK_GDELT.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.journal import JournalAgent  # noqa: E402
from agents.sources.gdelt import GDELTRateLimitedError, GDELTUnavailableError  # noqa: E402
from config import FUSO, tickers_ativos  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preencher_cache_gdelt")


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-populating do cache do GDELT.")
    p.add_argument("--inicio", required=True, help="data inicial YYYY-MM-DD")
    p.add_argument("--fim", required=True, help="data final YYYY-MM-DD")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="tickers (default: tickers_ativos por dia)")
    args = p.parse_args()

    inicio = pd.Timestamp(args.inicio, tz=FUSO)
    fim = pd.Timestamp(args.fim, tz=FUSO)
    journal = JournalAgent()
    dias = pd.bdate_range(inicio, fim, tz=FUSO)
    logger.info("Pré-carregando GDELT: %s..%s (%d dias úteis)",
                inicio.date(), fim.date(), len(dias))

    n = 0
    for dia in dias:
        universo = args.tickers if args.tickers is not None else tickers_ativos(dia)
        for ticker in universo:
            try:
                journal.get_noticias(ticker, dia)
            except (GDELTRateLimitedError, GDELTUnavailableError) as e:
                logger.error("GDELT degradou em %s/%s: %s", ticker, dia.date(), e)
                logger.error("Cache preserva o que já foi baixado. Espere ~horas "
                             "e retome o mesmo comando. (chamadas feitas: %d)", n)
                return 1
            n += 1
            if n % 100 == 0:
                logger.info("progresso: %d chamadas (último: %s %s)",
                            n, ticker, dia.date())

    hc = journal.health_check()
    logger.info("Concluído: %d chamadas. gdelt_degradado_count=%s",
                n, hc.get("gdelt_degradado_count"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
