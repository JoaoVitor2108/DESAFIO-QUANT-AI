"""Etapa 2 da calibração do ECON — comparação entre versões de prompt.

A rodada paga em si é do `baseline_econ.py` (amostragem, avaliação, checkpoint,
hard cap). Este módulo é a camada de DECISÃO em cima dela: resume cada versão na
mesma métrica, compara contra o baseline e a iteração anterior, aplica os
critérios de parada e mantém o relatório multi-versão.

Separação de arquivos por versão de prompt: `baseline_econ.caminho_completo` /
`caminho_checkpoint`. É o que impede a iteração N de retomar o checkpoint da N-1
(cuja chave é só (modelo, evento_id)) e reportar ΔIC = 0 falso.

Fluxo de uma iteração:
  1. edita `_SYSTEM_PROMPT` em agents/econ.py e bumpa `_PROMPT_VERSION`
  2. `python calibration/baseline_econ.py --sem-relatorio`   (rodada paga)
  3. `python calibration/etapa2_prompt.py`                   (comparação, grátis)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from calibration.baseline_econ import (
    RELATORIO_MD,
    VERSAO_BASELINE,
    calcular_metricas,
    caminho_completo,
    carregar_de_checkpoint,
    diagnosticar,
    separar_validas,
)
from calibration.econ_calibration import IC_META, registrar_iteracao_prompt

logger = logging.getLogger("etapa2_prompt")

# ── Critérios travados (R7) ───────────────────────────────────────────────────

MAX_ITERACOES = 5
CUSTO_HARD_CAP_ETAPA2 = 15.0
CUSTO_ALERTA_ULTIMA = 12.0     # acima disso, resta margem para ~1 iteração
CUSTO_PARAR = 14.0             # acima disso, para por orçamento
DELTA_CONVERGENCIA = 0.005     # 2 iterações seguidas abaixo disso = convergiu
IC_ACEITAVEL = 0.10

MARCA_SECAO = "<!-- etapa2:{versao} -->"


# ── Resumo de uma versão ──────────────────────────────────────────────────────


def resumo_versao(versao: str, df: pd.DataFrame) -> dict:
    """Métricas de uma versão de prompt, na mesma régua do baseline."""
    met = calcular_metricas(df)
    validas, degrad = separar_validas(df)
    lat = validas["latencia_llm_s"].dropna()
    return {
        "versao": versao,
        "n": met["econ_completo"]["n"],
        "n_limpo": met["econ_limpo"]["n"],
        "ic_completo": met["econ_completo"]["ic"],
        "ic_completo_low": met["econ_completo"]["ic95_low"],
        "ic_completo_high": met["econ_completo"]["ic95_high"],
        "ic_limpo": met["econ_limpo"]["ic"],
        "ic_limpo_low": met["econ_limpo"]["ic95_low"],
        "ic_limpo_high": met["econ_limpo"]["ic95_high"],
        "lexical_limpo": met["lexical_limpo"]["ic"],
        "gap_limpo": met["gap_limpo"],
        "gap_completo": met["gap_completo"],
        "custo": float(df["custo_usd"].sum()),
        "taxa_degradacao": degrad["taxa"],
        "lat_mediana": float(lat.median()) if len(lat) else float("nan"),
        "dispersao_score": float(validas["score"].std()),
        "zeros": int((validas["score"] == 0).sum()),
    }


def _fmt(x: float, casas: int = 4) -> str:
    return "n/d" if not (x == x) else f"{x:+.{casas}f}"


def tabela_comparativa(resumos: list[dict]) -> str:
    """Tabela acumulada exigida por R6 — uma linha por versão, custo acumulado."""
    linhas = ["| Versão | n | IC completo | IC limpo | GAP limpo | Custo iter | Custo acum |",
              "|---|---|---|---|---|---|---|"]
    acum = 0.0
    for r in resumos:
        acum += r["custo"]
        linhas.append(
            f"| {r['versao']} | {r['n']} | {_fmt(r['ic_completo'])} | "
            f"{_fmt(r['ic_limpo'])} | {_fmt(r['gap_limpo'])} | "
            f"US$ {r['custo']:.4f} | US$ {acum:.4f} |")
    return "\n".join(linhas)


# ── Critérios de parada (R7) ──────────────────────────────────────────────────


def decidir_parada(atual: dict, anteriores: list[dict], iteracao: int) -> dict:
    """Aplica os critérios de parada travados e diz se a decisão é minha ou humana.

    `atual` precisa de ic_limpo, gap_limpo e custo_acumulado; `anteriores` é o
    histórico de resumos (o baseline é o primeiro).
    """
    ic, gap = atual["ic_limpo"], atual["gap_limpo"]
    custo = atual["custo_acumulado"]

    # Sucesso exige IC alto E vantagem sobre o léxico: IC alto com GAP negativo
    # é o dicionário carregando o resultado, não o raciocínio do LLM.
    if ic == ic and ic > IC_META and gap > 0:
        return {"parar": True, "perguntar": False, "motivo": "sucesso",
                "detalhe": f"IC limpo {ic:+.4f} > {IC_META} com GAP {gap:+.4f} > 0"}

    if iteracao == 1 and anteriores and ic == ic and ic < anteriores[0]["ic_limpo"]:
        return {"parar": True, "perguntar": True, "motivo": "piorou_vs_baseline",
                "detalhe": (f"IC limpo caiu de {anteriores[0]['ic_limpo']:+.4f} para "
                            f"{ic:+.4f} já na primeira iteração — investigar antes de gastar mais")}

    if custo > CUSTO_PARAR:
        return {"parar": True, "perguntar": False, "motivo": "orcamento",
                "detalhe": f"custo acumulado US$ {custo:.2f} > US$ {CUSTO_PARAR:.0f}"}

    if iteracao >= MAX_ITERACOES:
        return {"parar": True, "perguntar": False, "motivo": "max_iteracoes",
                "detalhe": f"{MAX_ITERACOES} iterações atingidas"}

    ganhos = _ganhos_recentes(atual, anteriores)
    if len(ganhos) >= 2 and all(abs(g) < DELTA_CONVERGENCIA for g in ganhos[-2:]):
        return {"parar": True, "perguntar": False, "motivo": "convergencia",
                "detalhe": (f"2 iterações seguidas com ΔIC < {DELTA_CONVERGENCIA} "
                            f"({ganhos[-2]:+.4f}, {ganhos[-1]:+.4f})")}

    if iteracao >= 3 and ic == ic and IC_ACEITAVEL <= ic <= IC_META:
        return {"parar": False, "perguntar": True, "motivo": "faixa_intermediaria",
                "detalhe": (f"IC limpo {ic:+.4f} está entre {IC_ACEITAVEL} e {IC_META} "
                            "após 3 iterações — vale gastar mais 2?")}

    return {"parar": False, "perguntar": False, "motivo": "continuar",
            "detalhe": f"IC limpo {ic:+.4f}; segue para a próxima iteração"}


def _ganhos_recentes(atual: dict, anteriores: list[dict]) -> list[float]:
    """ΔIC limpo entre versões consecutivas, incluindo a atual."""
    serie = [r["ic_limpo"] for r in anteriores] + [atual["ic_limpo"]]
    return [b - a for a, b in zip(serie, serie[1:])]


# ── Relatório multi-versão ────────────────────────────────────────────────────


def secao_iteracao(resumo: dict, base: dict, anterior: dict | None,
                   dgn: dict, mudanca: str, custo_acumulado: float) -> str:
    """Seção da iteração no formato exigido por R5."""
    d_base = resumo["ic_limpo"] - base["ic_limpo"]
    d_ant = resumo["ic_limpo"] - anterior["ic_limpo"] if anterior else float("nan")
    piores = dgn["piores_casos"]
    lren = piores[piores["ticker"] == "LREN3.SA"]

    L = [MARCA_SECAO.format(versao=resumo["versao"]),
         f"\n## Iteração — prompt `{resumo['versao']}`\n",
         "### Mudança no prompt\n", mudanca, "\n### Resultados\n",
         f"- IC completo: {_fmt(resumo['ic_completo'])} "
         f"[{_fmt(resumo['ic_completo_low'])}, {_fmt(resumo['ic_completo_high'])}]",
         f"- IC limpo: {_fmt(resumo['ic_limpo'])} "
         f"[{_fmt(resumo['ic_limpo_low'])}, {_fmt(resumo['ic_limpo_high'])}] (n={resumo['n_limpo']})",
         f"- Lexical B0 limpo: {_fmt(resumo['lexical_limpo'])} (referência fixa)",
         f"- GAP limpo: {_fmt(resumo['gap_limpo'])}",
         f"- ΔIC limpo vs baseline: {_fmt(d_base)}",
         f"- ΔIC limpo vs iteração anterior: {_fmt(d_ant)}",
         f"- Custo desta iteração: US$ {resumo['custo']:.4f}",
         f"- Custo acumulado Etapa 2: US$ {custo_acumulado:.4f}",
         f"- Taxa de degradação: {resumo['taxa_degradacao']:.1%} | "
         f"latência mediana {resumo['lat_mediana']:.2f}s",
         f"- Dispersão do score: desvio {resumo['dispersao_score']:.4f}, "
         f"zeros {resumo['zeros']}",
         "\n### Diagnóstico\n",
         f"- **LREN3 tracker**: {len(lren)} de {len(piores)} piores casos "
         + ("— o eixo 1 NÃO resolveu o caso emblemático."
            if len(lren) else "— **saiu dos piores casos** (eixo 1 funcionou aqui)."),
         "\n**Piores casos desta versão:**\n",
         "| ticker | data | score | y |", "|---|---|---|---|"]
    L += [f"| {r['ticker']} | {pd.Timestamp(r['data']).date()} | "
          f"{r['score']:+.2f} | {r['y']:+.4f} |" for _, r in piores.iterrows()]

    tab = dgn["ic_por_ticker"]
    if not tab.empty:
        piores_t = ", ".join(f"{r['grupo']} ({r['ic']:+.3f})"
                             for _, r in tab.head(4).iterrows())
        melhores_t = ", ".join(f"{r['grupo']} ({r['ic']:+.3f})"
                               for _, r in tab.tail(3).iterrows())
        L += [f"\n- Piores tickers: {piores_t}", f"- Melhores tickers: {melhores_t}"]
    return "\n".join(L) + "\n"


def atualizar_relatorio(secao: str, tabela: str, versao: str,
                        caminho: Path = RELATORIO_MD,
                        conclusao: str = "") -> None:
    """Acrescenta (ou substitui) a seção da versão e refaz a tabela comparativa.

    Idempotente: rodar duas vezes a mesma versão não duplica seção — a marca
    HTML delimita o bloco.
    """
    texto = caminho.read_text(encoding="utf-8") if caminho.exists() else ""
    marca = MARCA_SECAO.format(versao=versao)
    if marca in texto:  # substitui o bloco desta versão
        ini = texto.index(marca)
        resto = texto[ini + len(marca):]
        prox = resto.find("<!-- etapa2:")
        fim = len(texto) if prox < 0 else ini + len(marca) + prox
        texto = texto[:ini] + texto[fim:]

    corpo, _, cauda = texto.partition(_MARCA_TABELA)
    # A conclusão vive DEPOIS da tabela; se não vier nova, preserva a que já existe
    # (senão uma reexecução da comparação apagaria o fechamento da etapa).
    if not conclusao:
        _, _, conclusao = cauda.partition(_MARCA_CONCLUSAO)
    novo = corpo.rstrip("\n") + "\n\n" + secao.rstrip("\n") + "\n\n"
    novo += (_MARCA_TABELA + "\n\n## Comparativo entre versões\n\n" + tabela + "\n")
    if conclusao.strip():
        novo += "\n" + _MARCA_CONCLUSAO + "\n" + conclusao.strip() + "\n"
    caminho.write_text(novo, encoding="utf-8")


_MARCA_TABELA = "<!-- etapa2:comparativo -->"
_MARCA_CONCLUSAO = "<!-- etapa2:conclusao -->"


def _dedupe_historico() -> None:
    """Uma linha por VERSÃO no histórico, mantendo a última.

    `registrar_iteracao_prompt` é append-only, então reexecutar a comparação da
    mesma versão (para corrigir o texto da mudança, por exemplo) duplicaria a
    entrada e faria a versão parecer duas iterações distintas."""
    import json

    from calibration.econ_calibration import RESULTS_DIR

    caminho = RESULTS_DIR / "prompt_iterations.jsonl"
    if not caminho.exists():
        return
    porversao: dict[str, dict] = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            reg = json.loads(linha)
            porversao[reg["versao"]] = reg
    caminho.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in porversao.values()),
        encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Comparação entre versões de prompt (Etapa 2)")
    p.add_argument("--versoes", nargs="*", default=None,
                   help="ordem das versões (default: baseline + _PROMPT_VERSION atual)")
    p.add_argument("--mudanca", default="",
                   help="descrição da mudança de prompt desta iteração (vai no relatório)")
    p.add_argument("--iteracao", type=int, default=1)
    p.add_argument("--conclusao-arquivo", default=None,
                   help="markdown com o fechamento da etapa (vai depois da tabela)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args(argv)
    from agents.econ import _PROMPT_VERSION
    from agents.journal import JournalAgent

    versoes = args.versoes or [VERSAO_BASELINE, _PROMPT_VERSION]
    journal = JournalAgent()
    journal.gdelt.buscar = lambda *a, **k: []
    journal.newsapi.buscar = lambda *a, **k: []

    resumos, dfs = [], {}
    for v in versoes:
        if not caminho_completo(v).exists():
            print(f"⛔ sem resultado para `{v}` ({caminho_completo(v).name}). "
                  "Rode `baseline_econ.py --sem-relatorio` para essa versão.")
            return 2
        df, _ = carregar_de_checkpoint(journal, versao=v)
        dfs[v] = df
        resumos.append(resumo_versao(v, df))

    tabela = tabela_comparativa(resumos)
    atual, base = resumos[-1], resumos[0]
    custo_etapa2 = sum(r["custo"] for r in resumos[1:])
    decisao = decidir_parada({**atual, "custo_acumulado": custo_etapa2},
                             anteriores=resumos[:-1], iteracao=args.iteracao)

    secao = secao_iteracao(atual, base, resumos[-2] if len(resumos) > 2 else None,
                           diagnosticar(dfs[atual["versao"]]), args.mudanca or
                           "_(não informada)_", custo_etapa2)
    conclusao = (Path(args.conclusao_arquivo).read_text(encoding="utf-8")
                 if args.conclusao_arquivo else "")
    atualizar_relatorio(secao, tabela, atual["versao"], conclusao=conclusao)
    registrar_iteracao_prompt(atual["versao"], atual["ic_completo"], atual["ic_limpo"],
                              decisao["motivo"], args.mudanca[:200])
    _dedupe_historico()

    print("\n" + tabela + "\n")
    print(f"DECISÃO: {decisao['motivo']} — {decisao['detalhe']}")
    print(f"parar={decisao['parar']} | perguntar_ao_humano={decisao['perguntar']}")
    print(f"Custo Etapa 2 acumulado: US$ {custo_etapa2:.4f} "
          f"(cap US$ {CUSTO_HARD_CAP_ETAPA2:.0f})")
    if custo_etapa2 > CUSTO_ALERTA_ULTIMA:
        print(f"⚠️ acima de US$ {CUSTO_ALERTA_ULTIMA:.0f}: resta margem para ~1 iteração")
    print(f"Relatório: {RELATORIO_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
