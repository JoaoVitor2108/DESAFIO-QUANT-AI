"""
Calibração do ECON — mede o poder preditivo do score e mitiga ativamente o
lookahead de memória do LLM.

Métrica: IC = Spearman(score_total, retorno em excesso ao Ibovespa em 5 dias úteis).
Meta: > 0.15.

Mitigação de viés (crítica para a banca — critério "rigor e mitigação de vieses"):

  DEFESA 1 — IC segmentado por exposição ao treino do modelo. O Haiku 4.5
  (`claude-haiku-4-5-20251001`) tem reliable knowledge cutoff em FEV/2025 e
  training data cutoff em JUL/2025 (doc oficial Anthropic). Logo:
    - notícia ATÉ fev/2025  → o modelo a conhece → IC é TETO OTIMISTA;
    - fev–jul/2025          → ainda no treino, "menos confiável" → INTERMEDIÁRIO;
    - APÓS jul/2025         → genuinamente fora do treino → IC LIMPO.
  Como o backtest OOS é 2024-2025, só jul-dez/2025 dá validação limpa.

  DEFESA 2 — Teste de placebo. Reroda o ECON nas MESMAS notícias com o nome da
  empresa anonimizado (ou trocado por par do setor), mantendo o conteúdo factual.
  Se o IC desaba ao anonimizar (ΔIC = IC_real − IC_placebo grande), parte do sinal
  vinha da memória do modelo sobre aquela empresa — red flag de lookahead.

  P8 — Auditoria das justificativas: varredura por linguagem ex-post (desfecho já
  conhecido), que delataria vazamento de memória.

A execução ao vivo (`calibrar`, `teste_placebo`) depende de `ANTHROPIC_API_KEY` e
de uma amostra de eventos com notícia; os helpers puros abaixo são testados
offline em `tests/test_econ_calibration.py`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from config import FUSO
from agents.sources.noticia import Noticia

logger = logging.getLogger(__name__)

# ── Cutoffs confirmados do claude-haiku-4-5-20251001 (doc oficial Anthropic) ───
RELIABLE_CUTOFF = pd.Timestamp("2025-02-28 23:59:59", tz=FUSO)  # reliable knowledge
TRAINING_CUTOFF = pd.Timestamp("2025-07-31 23:59:59", tz=FUSO)  # training data

IC_META = 0.15
RESULTS_DIR = Path(__file__).parent / "results"

# Placeholder padrão do modo "anonimizar"
_ANON_PLACEHOLDER = "a companhia"

# Padrões de linguagem EX-POST nas justificativas (P8). Cada par (label, regex).
_PADROES_EXPOST: list[tuple[str, str]] = [
    ("desfecho_passado", r"\b(caiu|subiu|despencou|disparou|recuou|saltou)\b"),
    ("posteriormente", r"\bposteriormente\b"),
    ("dias_depois", r"\b(no dia seguinte|dias? depois|semanas? depois|meses? depois)\b"),
    ("veio_a_confirmar", r"\b(veio a (se )?confirmar|se confirmou|viria a|acabou (por )?)\b"),
    ("sabemos_hoje", r"\b(como sabemos hoje|hoje sabemos|em retrospecto)\b"),
]

_ROTULO_SEGMENTO = {
    "teto_otimista": "TETO OTIMISTA (dentro do treino, ≤ fev/2025)",
    "intermediario": "INTERMEDIÁRIO (treino menos confiável, fev–jul/2025)",
    "limpo": "IC LIMPO (fora do training cutoff, > jul/2025)",
}


# ── Helpers puros (testados offline) ──────────────────────────────────────────


def segmentar_por_exposicao(data: pd.Timestamp) -> str:
    """Classifica a data quanto à exposição ao treino do modelo (DEFESA 1).

    Retorna "teto_otimista" (≤ reliable), "intermediario" (reliable–training) ou
    "limpo" (> training).
    """
    if data <= RELIABLE_CUTOFF:
        return "teto_otimista"
    if data <= TRAINING_CUTOFF:
        return "intermediario"
    return "limpo"


def anonimizar_noticias(
    noticias: list[Noticia],
    nome: str,
    modo: str = "anonimizar",
    substituto: Optional[str] = None,
) -> list[Noticia]:
    """Devolve cópias das notícias com o nome da empresa removido/trocado (DEFESA 2).

    modo="anonimizar": substitui por um placeholder neutro.
    modo="swap": substitui por `substituto` (par do mesmo setor) — exige `substituto`.
    Não muta os objetos originais.
    """
    if modo == "swap":
        if not substituto:
            raise ValueError("modo='swap' exige `substituto` (nome do par do setor).")
        alvo = substituto
    elif modo == "anonimizar":
        alvo = _ANON_PLACEHOLDER
    else:
        raise ValueError(f"modo inválido: {modo!r} (use 'anonimizar' ou 'swap').")

    padrao = re.compile(re.escape(nome), re.IGNORECASE)
    return [
        replace(
            n,
            titulo=padrao.sub(alvo, n.titulo),
            conteudo=padrao.sub(alvo, n.conteudo or ""),
        )
        for n in noticias
    ]


def auditar_justificativa(texto: str) -> list[str]:
    """Retorna os rótulos de padrões EX-POST encontrados na justificativa (P8).

    Lista vazia = justificativa limpa (prospectiva). Não vazia = inspecionar:
    linguagem de desfecho já conhecido sugere vazamento de memória do modelo.
    """
    if not texto:
        return []
    t = texto.lower()
    return [label for label, rx in _PADROES_EXPOST if re.search(rx, t)]


def calcular_ic(scores: Iterable[float], retornos: Iterable[float]) -> float:
    """IC = correlação de Spearman entre score e retorno em excesso.

    < 2 pares válidos → NaN. Usa pandas (sem nova dependência).
    """
    s = pd.Series(list(scores), dtype="float64")
    r = pd.Series(list(retornos), dtype="float64")
    if len(s) < 2 or len(r) < 2:
        return float("nan")
    return float(s.corr(r, method="spearman"))


# ── Orquestração ao vivo (gated em chave + amostra de eventos) ────────────────


def _retorno_excesso_5d(journal, ticker: str, data_limite: pd.Timestamp) -> Optional[float]:
    """Retorno em excesso ao Ibovespa nos 5 pregões após data_limite.

    NOTA: usa preços APÓS data_limite — é correto AQUI porque é a medição ex-post
    do alvo na calibração (não entra em nenhuma decisão de trade).
    """
    fim = data_limite + pd.Timedelta(days=12)  # folga p/ 5 pregões + fins de semana
    try:
        precos = journal.get_precos(ticker, data_limite, fim)["Close"].dropna()
        ibov = journal.get_precos("^BVSP", data_limite, fim)["Close"].dropna()
        if len(precos) < 6 or len(ibov) < 6:
            return None
        r_acao = precos.iloc[5] / precos.iloc[0] - 1
        r_ibov = ibov.iloc[5] / ibov.iloc[0] - 1
        return float(r_acao - r_ibov)
    except Exception as e:  # pragma: no cover - depende de rede
        logger.warning("retorno 5d indisponível para %s em %s: %s", ticker, data_limite, e)
        return None


def calibrar(agent, journal, eventos: Iterable[tuple], registrar: bool = True) -> dict:
    """Roda o ECON sobre `eventos` [(ticker, data_limite), ...] e reporta o IC.

    Reporta IC GLOBAL e SEGMENTADO por exposição ao treino (DEFESA 1), além de
    agregar flags de auditoria das justificativas (P8). Requer chave da Anthropic.
    """
    linhas = []
    for ticker, data_limite in eventos:
        score = agent.avaliar(ticker, data_limite)
        if not score.tem_evento:
            continue
        excesso = _retorno_excesso_5d(journal, ticker, data_limite)
        if excesso is None:
            continue
        linhas.append({
            "ticker": ticker,
            "data": data_limite,
            "segmento": segmentar_por_exposicao(data_limite),
            "score": score.score_total,
            "excesso_5d": excesso,
            "flags_justificativa": auditar_justificativa(score.justificativa),
        })

    df = pd.DataFrame(linhas)
    relatorio = {"n": len(df), "ic_global": float("nan"), "ic_por_segmento": {},
                 "n_justificativas_flagueadas": 0, "meta": IC_META}
    if not df.empty:
        relatorio["ic_global"] = calcular_ic(df["score"], df["excesso_5d"])
        for seg, g in df.groupby("segmento"):
            relatorio["ic_por_segmento"][seg] = {
                "ic": calcular_ic(g["score"], g["excesso_5d"]),
                "n": int(len(g)),
                "rotulo": _ROTULO_SEGMENTO[seg],
            }
        relatorio["n_justificativas_flagueadas"] = int(
            df["flags_justificativa"].apply(bool).sum()
        )

    if registrar:
        _salvar(df, relatorio, "calibracao")
    return relatorio


def teste_placebo(agent, journal, eventos: Iterable[tuple],
                  modo: str = "anonimizar") -> dict:
    """Compara IC real vs. IC com a empresa anonimizada (DEFESA 2).

    ΔIC = IC_real − IC_placebo. ΔIC grande sugere que parte do sinal vinha da
    memória do modelo sobre a empresa (lookahead), não do mecanismo econômico.
    """
    from config import TICKER_PARA_NOME

    reais, placebos, excessos = [], [], []
    for ticker, data_limite in eventos:
        noticias = journal.get_noticias(ticker, data_limite)
        if not noticias:
            continue
        excesso = _retorno_excesso_5d(journal, ticker, data_limite)
        if excesso is None:
            continue
        nome = TICKER_PARA_NOME.get(ticker, ticker)

        s_real = agent.avaliar(ticker, data_limite, noticias_override=noticias)
        anon = anonimizar_noticias(noticias, nome, modo=modo)
        s_anon = agent.avaliar(ticker, data_limite, noticias_override=anon,
                               nome_override=_ANON_PLACEHOLDER)

        reais.append(s_real.score_total)
        placebos.append(s_anon.score_total)
        excessos.append(excesso)

    ic_real = calcular_ic(reais, excessos)
    ic_placebo = calcular_ic(placebos, excessos)
    delta = ic_real - ic_placebo if (pd.notna(ic_real) and pd.notna(ic_placebo)) else float("nan")
    return {"n": len(excessos), "ic_real": ic_real, "ic_placebo": ic_placebo,
            "delta_ic": delta, "modo": modo}


def _salvar(df: pd.DataFrame, relatorio: dict, prefixo: str) -> None:  # pragma: no cover
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(RESULTS_DIR / f"{prefixo}_detalhe.csv", index=False)
    pd.Series(relatorio).to_json(RESULTS_DIR / f"{prefixo}_relatorio.json", indent=2,
                                 force_ascii=False)


# ── TODOs de calibração (executar quando houver chave + amostra) ───────────────
# TODO(P3): rodar subconjunto também com claude-sonnet-4-6 e comparar IC — checar
#           se o Haiku capta o MECANISMO (não o TOM). Se Sonnet >> Haiku, reavaliar.
# TODO(P6): medir sensibilidade do score a _MAX_CONTEUDO_CHARS (rodar com truncamento
#           maior num subconjunto; se o score muda pouco, o valor atual é defensável).
# TODO: montar a amostra curada de eventos (GDELT 2020-2021 + jul-dez/2025); congelar
#       o prompt após ≤10 iterações (bump de _PROMPT_VERSION a cada iteração).
