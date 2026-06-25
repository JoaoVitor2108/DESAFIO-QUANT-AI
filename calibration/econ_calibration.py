"""
Calibração do ECON — mede o poder preditivo do score e mitiga ativamente o
lookahead de memória do LLM.

Métrica: IC = Spearman(score_total, retorno em excesso ao Ibovespa em 5 dias úteis).
Meta: > 0.15.

Mitigação de viés (crítica para a banca — critério "rigor e mitigação de vieses"):

  DEFESA 1 — IC segmentado por exposição ao treino do modelo. Para LOOKAHEAD, a
  fronteira que importa é o TRAINING CUTOFF do Haiku 4.5 — JUL/2025 (doc oficial
  Anthropic). Tudo até jul/2025 ESTAVA no treino → mesmo risco de "cola". Logo,
  DOIS segmentos (não três):
    - "dentro_treino" (data ≤ jul/2025) → risco de cola → IC é TETO OTIMISTA;
    - "limpo"         (data >  jul/2025) → fora do treino → IC LIMPO.
  O reliable cutoff (fev/2025) mede a QUALIDADE do conhecimento, NÃO a presença no
  treino, então não entra na segmentação anti-lookahead. Como o backtest OOS é
  2024-2025, só jul-dez/2025 dá validação limpa.

  DEFESA 2 — Teste de placebo. Reroda o ECON sobre o mesmo evento ocultando a
  identidade da empresa, em dois modos:
    - "swap": troca a empresa A por um par B do MESMO setor — TODO o contexto
      (fundamentos/setor/macro) passa a ser de B (avaliando o ticker de B), só o
      texto factual da notícia é preservado com o nome trocado A→B;
    - "identidade_pura": anonimiza o nome E oculta fundamentos/setor/macro, deixando
      só o texto da notícia.
  Se o IC desaba (ΔIC = IC_real − IC_placebo grande), parte do sinal vinha da
  memória do modelo sobre aquela empresa — red flag de lookahead. (O placebo antigo,
  que trocava só o nome mantendo os fundamentos reais, vazava a identidade.)

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

# ── Cutoffs do claude-haiku-4-5-20251001 (doc oficial Anthropic) ───────────────
# ⚠️ PRECISÃO: a Anthropic publica apenas o MÊS ("Feb 2025" / "Jul 2025"), não o dia.
# Usamos o FIM DO MÊS como fronteira CONSERVADORA: inclui mais dias em "dentro_treino"
# e exige mais da janela "limpa" (não inventamos precisão de dia que a fonte não dá).
# RELIABLE_CUTOFF é documentado mas NÃO entra na segmentação anti-lookahead (só o
# TRAINING_CUTOFF importa para risco de cola — ver segmentar_por_exposicao).
RELIABLE_CUTOFF = pd.Timestamp("2025-02-28 23:59:59", tz=FUSO)  # reliable knowledge (fim de Feb/2025)
TRAINING_CUTOFF = pd.Timestamp("2025-07-31 23:59:59", tz=FUSO)  # training data (fim de Jul/2025; fronteira anti-lookahead)

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
    "dentro_treino": "DENTRO DO TREINO (teto otimista, ≤ jul/2025)",
    "limpo": "IC LIMPO (pós-jul/2025, fora do training cutoff)",
}


# ── Helpers puros (testados offline) ──────────────────────────────────────────


def segmentar_por_exposicao(data: pd.Timestamp) -> str:
    """Classifica a data quanto à exposição ao treino do modelo (DEFESA 1).

    Para LOOKAHEAD, a única fronteira que importa é o TRAINING_CUTOFF (jul/2025):
    tudo até lá ESTAVA no treino e tem o mesmo risco de "cola", independentemente
    do reliable cutoff (que mede só a QUALIDADE do conhecimento, não a presença).
    Por isso DOIS segmentos, não três:
      - "dentro_treino" (data ≤ jul/2025) → risco de cola → IC = teto otimista;
      - "limpo"         (data >  jul/2025) → fora do treino → IC confiável.
    RELIABLE_CUTOFF NÃO entra nesta classificação (mantido só como doc).
    """
    if data <= TRAINING_CUTOFF:
        return "dentro_treino"
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
    """IC = correlação de Spearman entre score e retorno em excesso (ponto).

    < 2 pares válidos → NaN. Usa pandas (sem nova dependência). Wrapper de ponto
    sobre `calcular_ic_com_ic` (mantido para compatibilidade).
    """
    s = pd.Series(list(scores), dtype="float64")
    r = pd.Series(list(retornos), dtype="float64")
    if len(s) < 2 or len(r) < 2:
        return float("nan")
    return float(s.corr(r, method="spearman"))


def calcular_ic_com_ic(scores: Iterable[float], retornos: Iterable[float],
                       n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """IC de Spearman com intervalo de confiança 95% e p-valor via bootstrap (v4-C2).

    Reamostra com reposição `n_bootstrap` vezes, recalcula Spearman, e toma os
    percentis 2.5/97.5. P-valor unilateral (H1: IC > 0) = fração de bootstraps com
    IC <= 0. Um IC de 0.15 com N=40 pode não ser distinguível de zero — daí o IC.
    """
    import numpy as np

    s = np.asarray(list(scores), dtype="float64")
    r = np.asarray(list(retornos), dtype="float64")
    n = len(s)
    ic = calcular_ic(s, r)
    if n < 2 or not (ic == ic):  # NaN
        return {"ic": float("nan"), "ic95_low": float("nan"),
                "ic95_high": float("nan"), "p_valor": float("nan"), "n": int(n)}

    rng = np.random.default_rng(seed)
    amostras = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        v = calcular_ic(s[idx], r[idx])
        if v == v:  # descarta NaN (reamostra degenerada)
            amostras.append(v)
    amostras = np.array(amostras) if amostras else np.array([ic])
    return {
        "ic": float(ic),
        "ic95_low": float(np.percentile(amostras, 2.5)),
        "ic95_high": float(np.percentile(amostras, 97.5)),
        "p_valor": float(np.mean(amostras <= 0.0)),
        "n": int(n),
    }


# ── v4-P5: classificação de degradação ─────────────────────────────────────────

# Marcadores nos `avisos` que indicam degradação REAL (não a mera divergência P7).
_MARCADORES_DEGRADACAO = [
    ("sem_chave", ("ANTHROPIC_API_KEY", "sem chave")),
    ("erro_api", ("erro na chamada ao Claude", "erro de API")),
    ("malformada", ("tool_use",)),
]


def classificar_degradacao(score) -> Optional[str]:
    """Tipo de degradação de um ScoreEcon, ou None se a avaliação ocorreu (v4-P5).

    A divergência score×notícia (P7) NÃO é degradação — a avaliação aconteceu.
    """
    texto = " ".join(score.avisos)
    for tipo, marcadores in _MARCADORES_DEGRADACAO:
        if any(m in texto for m in marcadores):
            return tipo
    return None


# ── v4-C3: baseline de sentimento lexical simples ──────────────────────────────

# Listas mínimas curadas (BR-PT). NÃO pretende ser bom — é só o ponto de comparação
# contra o qual medimos o valor agregado do raciocínio do LLM (GAP de IC, v4-C3).
_PALAVRAS_POS = {
    "lucro", "alta", "crescimento", "recorde", "expansão", "ganho", "valorização",
    "aprovação", "dividendo", "receita", "superávit", "melhora", "forte", "positivo",
    "aumento", "avanço", "otimista", "supera", "elevação", "robusto", "demanda",
    "contrato", "aquisição", "investimento", "eficiência", "margem", "recuperação",
    "sobe", "disparada", "premiada",
}
_PALAVRAS_NEG = {
    "prejuízo", "queda", "perda", "recuo", "investigação", "fraude", "multa",
    "rebaixamento", "dívida", "calote", "crise", "demissão", "fechamento", "greve",
    "processo", "despencou", "negativo", "fraco", "redução", "atraso", "default",
    "escândalo", "rombo", "suspensão", "deficit", "tombo", "alerta", "risco",
    "cai", "desvalorização",
}


def baseline_sentimento_simples(noticias) -> float:
    """Score de sentimento lexical em [-1, +1] (v4-C3 — baseline trivial).

    Conta palavras positivas/negativas no título+conteúdo. (pos-neg)/(pos+neg).
    """
    import re as _re
    pos = neg = 0
    for n in noticias:
        texto = f"{n.titulo} {n.conteudo or ''}".lower()
        tokens = _re.findall(r"[a-záàâãéêíóôõúç]+", texto)
        pos += sum(t in _PALAVRAS_POS for t in tokens)
        neg += sum(t in _PALAVRAS_NEG for t in tokens)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ── v4-C4: diagnóstico de colinearidade implícita (ligado a v4-P3) ─────────────


def diagnosticar_colinearidade(scores: list, contextos: list[dict]) -> dict:
    """corr(score_total, cada feature fundamental crua) — Pearson (v4-C4).

    A Opção A remove a colinearidade EXPLÍCITA (não soma componentes), mas o
    score_total é condicionado ao contexto fundamental, então resta colinearidade
    IMPLÍCITA. Aqui medimos essa correlação residual para reportar à banca.
    `contextos[i]` = payload['fundamentos'] do evento i (dict de feature→valor).
    Heurística: |corr| > 0.5 = alerta; > 0.7 = forte.
    """
    y = pd.Series([s.score_total for s in scores], dtype="float64")
    feats: dict[str, list] = {}
    for ctx in contextos:
        fund = ctx.get("fundamentos", ctx) or {}
        for k, v in fund.items():
            if isinstance(v, (int, float)):
                feats.setdefault(k, []).append(float(v))
    out = {}
    for k, vals in feats.items():
        if len(vals) == len(y) and len(y) >= 2:
            x = pd.Series(vals, dtype="float64")
            # feature constante (std 0) → correlação indefinida; evita warning numpy
            out[k] = float("nan") if x.std(ddof=1) == 0 else float(y.corr(x))
    return out


# ── Orquestração ao vivo (gated em chave + amostra de eventos) ────────────────


def _beta_setorial(journal, ticker: str, data_limite: pd.Timestamp,
                   janela_dias: int = 60) -> Optional[float]:
    """Beta do ticker vs. Ibovespa estimado na janela ANTES de data_limite (v4-P4).

    ⚠️ SEM LOOKAHEAD: a janela de estimação é [data_limite - janela_dias, data_limite]
    — só preços ATÉ data_limite. Não toca preços pós-data_limite (esses são o ALVO,
    medidos em `_retorno_excesso_5d`). beta = cov(r_acao, r_ibov) / var(r_ibov).
    """
    ini = data_limite - pd.Timedelta(days=janela_dias + 10)  # folga p/ pregões
    try:
        precos = journal.get_precos(ticker, ini, data_limite)["Close"].dropna()
        r_acao = precos.pct_change().dropna()
        r_ibov = journal.get_retorno_ibovespa(ini, data_limite).dropna()
        df = pd.concat([r_acao, r_ibov], axis=1, join="inner").dropna()
        if len(df) < 20:
            return None
        cov = df.iloc[:, 0].cov(df.iloc[:, 1])
        var = df.iloc[:, 1].var()
        if var == 0 or pd.isna(var):
            return None
        return float(cov / var)
    except Exception as e:  # pragma: no cover - depende de rede
        logger.warning("beta setorial indisponível para %s em %s: %s", ticker, data_limite, e)
        return None


def _retorno_excesso_5d(journal, ticker: str, data_limite: pd.Timestamp,
                        ajuste_beta: bool = False) -> Optional[float]:
    """Retorno em excesso ao Ibovespa nos 5 pregões após data_limite.

    ⚠️ MEDIÇÃO DE ALVO EX-POST — usa preços APÓS data_limite DE PROPÓSITO (é o
    retorno realizado que queremos prever). NUNCA reutilizar este caminho nem estes
    preços no payload de `avaliar()`: isso seria lookahead ESTRUTURAL (real, não de
    memória do LLM). Garantia de isolamento: o caminho de decisão
    (`avaliar`/`_montar_contexto`) NUNCA chama `get_precos` — só usa
    get_noticias/get_fundamentals/get_macro/get_retornos_setor/get_setor, todos
    cortados em data_limite. Logo não há objeto/cache de preço compartilhado.
    (Coberto por `tests/test_econ.py::test_avaliar_nunca_busca_precos`.)

    `ajuste_beta` (v4-P4): excesso SIMPLES (r_acao - r_ibov) é o PADRÃO — valida o
    ECON como sinal puro, desacoplado do modelo de beta setorial (que é
    responsabilidade do MATH&ML). Com `ajuste_beta=True`, desconta beta × r_ibov,
    onde o beta é estimado ANTES de data_limite (`_beta_setorial`, sem lookahead),
    alinhando à métrica que o MATH&ML otimiza (excesso ajustado por beta).
    """
    fim = data_limite + pd.Timedelta(days=12)  # folga p/ 5 pregões + fins de semana
    try:
        precos = journal.get_precos(ticker, data_limite, fim)["Close"].dropna()
        ibov = journal.get_precos("^BVSP", data_limite, fim)["Close"].dropna()
        if len(precos) < 6 or len(ibov) < 6:
            return None
        r_acao = precos.iloc[5] / precos.iloc[0] - 1
        r_ibov = ibov.iloc[5] / ibov.iloc[0] - 1
        if ajuste_beta:
            beta = _beta_setorial(journal, ticker, data_limite)
            if beta is None:
                return None
            return float(r_acao - beta * r_ibov)
        return float(r_acao - r_ibov)
    except Exception as e:  # pragma: no cover - depende de rede
        logger.warning("retorno 5d indisponível para %s em %s: %s", ticker, data_limite, e)
        return None


# Limiar mínimo de eventos na janela LIMPA para o IC limpo ser significativo (v4-C1).
# REVISAR: 30 é regra de bolso; ajustar conforme a amostra real e o poder do teste.
_N_MIN_LIMPO = 30


def contar_eventos_por_segmento(eventos: Iterable[tuple], registrar: bool = False) -> dict:
    """Conta eventos por segmento ANTES de rodar o IC (v4-C1).

    A janela limpa (pós-jul/2025) é curta (~jul-dez/2025); se N_limpo for pequeno, o
    IC limpo vira ruído. Reporta a contagem e alerta (sem bloquear) se < _N_MIN_LIMPO.
    """
    cont = {"dentro_treino": 0, "limpo": 0}
    for _ticker, data in eventos:
        cont[segmentar_por_exposicao(data)] += 1
    cont["total"] = cont["dentro_treino"] + cont["limpo"]
    cont["n_min_limpo"] = _N_MIN_LIMPO
    cont["aviso_n_limpo"] = cont["limpo"] < _N_MIN_LIMPO
    if cont["aviso_n_limpo"]:
        logger.warning("janela limpa estatisticamente fraca: N_limpo=%d < %d; "
                       "considere estender o backtest para incluir mais de 2026.",
                       cont["limpo"], _N_MIN_LIMPO)
    if registrar:  # pragma: no cover - I/O
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.Series(cont).to_json(RESULTS_DIR / "event_count.json", indent=2)
    return cont


# Limiares da taxa de degradação (v4-P5). REVISAR: regras de bolso.
_DEGRAD_RESSALVA = 0.05   # > 5% → IC reportado com ressalva
_DEGRAD_INVALIDA = 0.15   # > 15% → calibração considerada inválida


def calibrar(agent, journal, eventos: Iterable[tuple], registrar: bool = True,
             com_baseline: bool = True) -> dict:
    """Roda o ECON sobre `eventos` [(ticker, data_limite), ...] e reporta o IC.

    Reporta (tudo com mitigação ativa de viés):
    - IC GLOBAL e por SEGMENTO de exposição ao treino (DEFESA 1), com IC95 bootstrap
      (v4-C2), em excesso SIMPLES (primário) e AJUSTADO POR BETA (secundário, v4-P4);
    - GAP contra um baseline de sentimento lexical (v4-C3);
    - relatório de DEGRADAÇÕES por tipo + taxa (v4-P5);
    - contagem de eventos por segmento (v4-C1) e flags de justificativa (P8).

    CRITÉRIO DE PARADA do loop de prompt (≤10 iterações; v4-C6):
    - Sucesso: IC do segmento LIMPO > IC_META (0,15) com IC95 não cruzando zero.
    - Overfit (PARAR já): IC `dentro_treino` sobe mas IC `limpo` NÃO sobe (ou cai)
      entre iterações → o prompt está "colando" na memória do modelo.
    - A cada iteração, faça bump de `_PROMPT_VERSION` (invalida o cache) e registre
      com `registrar_iteracao_prompt(...)`.

    Requer ANTHROPIC_API_KEY + amostra de eventos.
    """
    eventos = list(eventos)
    contagem = contar_eventos_por_segmento(eventos)
    degrad = {"sem_chave": 0, "erro_api": 0, "malformada": 0, "total": 0}
    n_aval = 0
    linhas = []

    for ticker, data_limite in eventos:
        noticias = journal.get_noticias(ticker, data_limite)
        if not noticias:
            continue
        n_aval += 1
        score = agent.avaliar(ticker, data_limite, noticias_override=noticias)

        tipo_degrad = classificar_degradacao(score)
        if tipo_degrad is not None:
            degrad[tipo_degrad] = degrad.get(tipo_degrad, 0) + 1
            degrad["total"] += 1
            continue

        excesso = _retorno_excesso_5d(journal, ticker, data_limite)
        excesso_beta = _retorno_excesso_5d(journal, ticker, data_limite, ajuste_beta=True)
        if excesso is None:
            continue
        linhas.append({
            "ticker": ticker,
            "data": data_limite,
            "segmento": segmentar_por_exposicao(data_limite),
            "score": score.score_total,
            "score_baseline": baseline_sentimento_simples(noticias) if com_baseline else float("nan"),
            "excesso_5d": excesso,
            "excesso_5d_beta": excesso_beta,
            "flags_justificativa": auditar_justificativa(score.justificativa),
        })

    df = pd.DataFrame(linhas)
    taxa_degrad = degrad["total"] / n_aval if n_aval else 0.0
    relatorio = {
        "n": len(df),
        "contagem_eventos": contagem,
        "ic_global": calcular_ic_com_ic(df["score"], df["excesso_5d"]) if not df.empty else None,
        "ic_global_beta": calcular_ic_com_ic(df["score"], df["excesso_5d_beta"]) if not df.empty else None,
        "ic_por_segmento": {},
        "baseline": {}, "degradacao": {}, "n_justificativas_flagueadas": 0, "meta": IC_META,
    }
    relatorio["degradacao"] = {
        **degrad, "n_avaliacoes": n_aval, "taxa": taxa_degrad,
        "ressalva": taxa_degrad > _DEGRAD_RESSALVA,
        "invalida": taxa_degrad > _DEGRAD_INVALIDA,
    }
    if not df.empty:
        for seg, g in df.groupby("segmento"):
            r = calcular_ic_com_ic(g["score"], g["excesso_5d"])
            r["rotulo"] = _ROTULO_SEGMENTO[seg]
            relatorio["ic_por_segmento"][seg] = r
        if com_baseline:
            ic_econ = calcular_ic(df["score"], df["excesso_5d"])
            ic_base = calcular_ic(df["score_baseline"], df["excesso_5d"])
            relatorio["baseline"] = {
                "ic_econ": ic_econ, "ic_baseline": ic_base,
                "gap": ic_econ - ic_base if pd.notna(ic_econ) and pd.notna(ic_base) else float("nan"),
            }
        relatorio["n_justificativas_flagueadas"] = int(df["flags_justificativa"].apply(bool).sum())

    if registrar:
        _salvar(df, relatorio, "calibracao")
        _salvar_json(relatorio["degradacao"], "degradation_report")
        _salvar_json(contagem, "event_count")
    return relatorio


def _par_do_setor(ticker: str, data_limite: pd.Timestamp) -> Optional[str]:
    """Um par ativo do MESMO setor de `ticker` na data (para o placebo swap)."""
    from config import UNIVERSO_HISTORICO, tickers_ativos

    setor = UNIVERSO_HISTORICO.get(ticker, {}).get("setor")
    if setor is None:
        return None
    for t in tickers_ativos(data_limite):
        if t != ticker and UNIVERSO_HISTORICO.get(t, {}).get("setor") == setor:
            return t
    return None


def teste_placebo(agent, journal, eventos: Iterable[tuple],
                  estrategia: str = "swap") -> dict:
    """Compara IC real vs. IC com a IDENTIDADE da empresa ocultada (DEFESA 2).

    ΔIC = IC_real − IC_placebo. ΔIC grande sugere que parte do sinal vinha da
    memória do modelo sobre a empresa (lookahead), não do mecanismo econômico.

    `estrategia` é o parâmetro de ALTO NÍVEL do teste (não confundir com o `modo` de
    baixo nível de `anonimizar_noticias`, que opera só sobre o TEXTO):
    - "swap": troca A por um par B do setor — contexto INTEIRO vira de B (avalia o
      ticker de B), só o texto factual da notícia é preservado (nome A→B). Internamente
      usa `anonimizar_noticias(..., modo="swap")`.
    - "identidade_pura": `anonimizar_noticias(..., modo="anonimizar")` +
      `incluir_contexto_fundamental=False` (oculta fundamentos/setor/macro).
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

        if estrategia == "swap":
            par = _par_do_setor(ticker, data_limite)
            if par is None:
                continue  # sem par do setor, não dá para fazer o swap completo
            nome_par = TICKER_PARA_NOME.get(par, par)
            noticias_swap = anonimizar_noticias(noticias, nome, modo="swap", substituto=nome_par)
            # ticker=par → _montar_contexto busca fundamentos/setor/macro DE B
            s_placebo = agent.avaliar(par, data_limite, noticias_override=noticias_swap)
        elif estrategia == "identidade_pura":
            anon = anonimizar_noticias(noticias, nome, modo="anonimizar")
            s_placebo = agent.avaliar(ticker, data_limite, noticias_override=anon,
                                      nome_override=_ANON_PLACEHOLDER,
                                      incluir_contexto_fundamental=False)
        else:
            raise ValueError(f"estrategia inválida: {estrategia!r} "
                             "(use 'swap' ou 'identidade_pura').")

        s_real = agent.avaliar(ticker, data_limite, noticias_override=noticias)
        reais.append(s_real.score_total)
        placebos.append(s_placebo.score_total)
        excessos.append(excesso)

    ic_real = calcular_ic(reais, excessos)
    ic_placebo = calcular_ic(placebos, excessos)
    delta = ic_real - ic_placebo if (pd.notna(ic_real) and pd.notna(ic_placebo)) else float("nan")
    return {"n": len(excessos), "ic_real": ic_real, "ic_placebo": ic_placebo,
            "delta_ic": delta, "estrategia": estrategia}


# ── v4-C5: comparação de modelos (Haiku vs Sonnet) — PRIORIDADE quando houver chave
# Gaps de decisão (REVISAR): adotar Sonnet se ganho de IC limpo > _GAP_ADOTAR_SONNET;
# ficar com Haiku se < _GAP_FICAR_HAIKU; faixa intermediária = decisão humana.
_GAP_ADOTAR_SONNET = 0.05
_GAP_FICAR_HAIKU = 0.03


def comparar_modelos(journal, eventos: Iterable[tuple],
                     modelos=("claude-haiku-4-5-20251001", "claude-sonnet-4-6"),
                     cache_dir=Path("data/cache")) -> dict:
    """Roda os MESMOS eventos no MESMO prompt em cada modelo e compara (v4-C5).

    Reporta, por modelo, IC (com IC95 bootstrap) e — onde estimável — custo; e a
    concordância (% de scores com |Δ| < 0.2). Recomenda-se um subset da janela
    LIMPA (pós-jul/2025) para evitar cola de memória diferencial entre modelos.
    Gated em ANTHROPIC_API_KEY.
    """
    from agents.econ import EconAgent

    eventos = list(eventos)
    por_modelo = {}
    scores_por_modelo = {}
    for modelo in modelos:
        agent = EconAgent(journal=journal, model=modelo, cache_dir=cache_dir)
        scores, excessos = [], []
        for ticker, data_limite in eventos:
            noticias = journal.get_noticias(ticker, data_limite)
            if not noticias:
                continue
            s = agent.avaliar(ticker, data_limite, noticias_override=noticias)
            if classificar_degradacao(s) is not None:
                continue
            exc = _retorno_excesso_5d(journal, ticker, data_limite)
            if exc is None:
                continue
            scores.append(s.score_total)
            excessos.append(exc)
        scores_por_modelo[modelo] = scores
        por_modelo[modelo] = {"ic": calcular_ic_com_ic(scores, excessos), "n": len(scores)}

    # Concordância par-a-par (só se os modelos cobriram os mesmos eventos)
    concordancia = None
    chaves = list(scores_por_modelo)
    if len(chaves) == 2:
        a, b = scores_por_modelo[chaves[0]], scores_por_modelo[chaves[1]]
        if a and len(a) == len(b):
            concordancia = float(sum(abs(x - y) < 0.2 for x, y in zip(a, b)) / len(a))

    ics = {m: por_modelo[m]["ic"]["ic"] for m in modelos if m in por_modelo}
    return {"por_modelo": por_modelo, "concordancia_lt_0_2": concordancia,
            "gap_adotar_sonnet": _GAP_ADOTAR_SONNET, "gap_ficar_haiku": _GAP_FICAR_HAIKU,
            "ics": ics}


# ── v4-C6: log do histórico de iterações de prompt ─────────────────────────────


def registrar_iteracao_prompt(versao: str, ic_dentro: float, ic_limpo: float,
                              decisao: str, resumo_diff: str = "") -> None:  # pragma: no cover
    """Acrescenta uma linha ao histórico de iterações do prompt (v4-C6).

    Uma linha por iteração em `calibration/results/prompt_iterations.jsonl`:
    versão do prompt, IC dentro_treino, IC limpo, decisão (continuar/parar) e um
    resumo do diff. Permite ver se o IC limpo evolui (ok) ou estagna enquanto o
    IC dentro sobe (overfit → parar).
    """
    import json as _json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    linha = {"versao": versao, "ic_dentro_treino": ic_dentro, "ic_limpo": ic_limpo,
             "decisao": decisao, "resumo_diff": resumo_diff}
    with open(RESULTS_DIR / "prompt_iterations.jsonl", "a") as f:
        f.write(_json.dumps(linha, ensure_ascii=False) + "\n")


def _salvar(df: pd.DataFrame, relatorio: dict, prefixo: str) -> None:  # pragma: no cover
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(RESULTS_DIR / f"{prefixo}_detalhe.csv", index=False)
    pd.Series(relatorio).to_json(RESULTS_DIR / f"{prefixo}_relatorio.json", indent=2,
                                 force_ascii=False)


def _salvar_json(obj: dict, nome: str) -> None:  # pragma: no cover
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(obj).to_json(RESULTS_DIR / f"{nome}.json", indent=2, force_ascii=False)


# ── TODOs de calibração (executar quando houver chave + amostra) ───────────────
# TODO(v4-P6/econ): medir sensibilidade do score a _MAX_CONTEUDO_CHARS (rodar com
#   truncamento maior num subconjunto; se o score muda pouco, o valor é defensável).
# TODO(C4): chamar diagnosticar_colinearidade com os fundamentos por evento e salvar
#   colinearity_report.json (medir a colinearidade IMPLÍCITA da Opção A — v4-P3).
# TODO: montar a amostra curada de eventos (GDELT 2020-2021 + jul-dez/2025); congelar
#       o prompt após ≤10 iterações (bump de _PROMPT_VERSION a cada iteração).
