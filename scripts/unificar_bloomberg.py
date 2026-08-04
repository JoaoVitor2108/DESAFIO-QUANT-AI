"""
Unifica os Excel brutos do Bloomberg num único CSV para o backtest 2024-2025.

Roda o parser da Etapa 1 sobre os arquivos das duas visitas à FGV, concatena,
deduplica por (ticker, data, titulo) e sobrescreve
`data/bloomberg/parsed/noticias.csv`, guardando o CSV anterior em `.bak`.

Uso: python3 scripts/unificar_bloomberg.py
"""
from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.sources.bloomberg_parser import (  # noqa: E402
    NoticiaBloomberg,
    escrever_csv,
    parsear_excel_bloomberg,
)

# Excel brutos, na ordem de coleta. O CSV do gate é reconstruído a partir do
# Excel original em vez de reaproveitado, para que todo o dataset passe pela
# mesma versão do parser.
ENTRADAS = (
    Path("data/raw/Pasta2(Recuperado Automaticamente).xlsx"),
    Path("data/bloomberg/raw/trab.xlsx"),
    Path("data/bloomberg/raw/trab-2.xlsx"),
)
SAIDA = Path("data/bloomberg/parsed/noticias.csv")
BACKUP = Path("data/bloomberg/parsed/noticias.csv.bak")

# Tickers que o backtest OOS 2024-2025 espera cobrir (config.tickers_ativos).
TICKERS_ESPERADOS = {
    "PETR4", "PRIO3", "VALE3", "CMIN3", "WEGE3", "GGBR4", "ITUB4", "BBDC4",
    "BBAS3", "BPAC11", "ABEV3", "JBSS3", "ELET3", "EGIE3", "SUZB3", "KLBN11",
    "VIVT3", "RDOR3", "LREN3", "MGLU3", "ASAI3", "BBSE3", "CYRE3", "TOTS3",
}


def _chave(n: NoticiaBloomberg) -> tuple[str, str, str]:
    return (n.ticker, n.data.isoformat(), n.titulo)


def main() -> int:
    faltando = [p for p in ENTRADAS if not p.exists()]
    if faltando:
        print("ERRO: arquivos ausentes:", *faltando, sep="\n  ")
        return 1

    todas: list[NoticiaBloomberg] = []
    print("── Parsing por arquivo " + "─" * 46)
    for caminho in ENTRADAS:
        noticias, rel = parsear_excel_bloomberg(caminho)
        todas.extend(noticias)
        print(f"\n{caminho.name}: {rel.n_noticias_extraidas} notícias")
        print(f"  abas vazias ({len(rel.abas_vazias)}): {rel.abas_vazias}")
        print(f"  códigos de fonte desconhecidos: {rel.codigos_fonte_desconhecidos}")
        print(f"  puladas: título={rel.n_puladas_titulo_vazio} "
              f"data={rel.n_puladas_data_invalida} | "
              f"duplicatas intra-arquivo={rel.n_duplicatas_removidas}")

    # Dedup cross-arquivo, preservando a primeira ocorrência.
    vistas: set[tuple[str, str, str]] = set()
    unicas: list[NoticiaBloomberg] = []
    for n in todas:
        if _chave(n) in vistas:
            continue
        vistas.add(_chave(n))
        unicas.append(n)
    n_dup = len(todas) - len(unicas)

    # Backup só na primeira execução: reexecutar não pode sobrescrever o
    # estado pré-unificação com uma cópia da própria saída.
    if SAIDA.exists() and not BACKUP.exists():
        shutil.copy2(SAIDA, BACKUP)
        print(f"\nbackup: {SAIDA} → {BACKUP}")
    elif BACKUP.exists():
        print(f"\nbackup preservado (já existe): {BACKUP}")

    escrever_csv(unicas, SAIDA)

    print("\n── Validações (R5) " + "─" * 50)
    print(f"1. total antes do dedup ....... {len(todas)}")
    print(f"   total no CSV unificado ..... {len(unicas)}")
    print(f"4. duplicatas removidas ....... {n_dup}")

    por_ticker = Counter(n.ticker for n in unicas)
    print(f"2. tickers presentes .......... {len(por_ticker)}")
    for t, c in sorted(por_ticker.items()):
        print(f"     {t:<12} {c:>4}")

    por_ano = Counter(n.data.year for n in unicas)
    print("3. por ano:")
    for ano, c in sorted(por_ano.items()):
        print(f"     {ano} {c:>4}")

    presentes = {t.removesuffix(".SA") for t in por_ticker}
    ausentes = sorted(TICKERS_ESPERADOS - presentes)
    extras = sorted(presentes - TICKERS_ESPERADOS)
    print(f"6. tickers esperados ausentes .. {ausentes or 'nenhum'}")
    print(f"   tickers fora da lista ....... {extras or 'nenhum'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
