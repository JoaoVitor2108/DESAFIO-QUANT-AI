"""Baseline da calibração do ECON — Etapa 1: mede o prompt ATUAL, não o muda.

Estabelece o número que as iterações de prompt (Etapa 2) precisam superar: o IC
de Spearman entre `score_total` e o retorno beta-ajustado 5du, sobre o dataset
Bloomberg unificado (2024-2025). Não itera prompt, não roda placebo, não roda
mini-gate — isso é Etapa 2+.

Decisões travadas desta rodada:
  - Amostra: dataset unificado inteiro, janela 2024-01-01 a 2025-12-31.
  - DOIS ICs: "completo" (2024-2025, mais poder, contaminação parcial em 2024) e
    "limpo" (`data_noticia_mais_recente >= 2025-08-01`, pós training cutoff do
    Haiku 4.5 — mesma fronteira do gate de custo).
  - Block bootstrap por data (blocos de 5du, 10k iterações, seed 42), importado
    de `econ_calibration` — mesma rotina do gate.
  - Bloomberg-only: GDELT e NewsAPI neutralizados (IP-flakiness), igual ao gate.
  - Só Haiku 4.5; Sonnet foi descartado no gate (ΔIC = -0.0298, commit 85c69f9).

DEDUPLICAÇÃO POR CONJUNTO DE NOTÍCIAS (padrão, `--sem-dedup` desliga): o lookback
de 7 dias faz o MESMO conjunto de notícias reaparecer em vários pregões vizinhos
do mesmo ticker. Essas repetições não são observações independentes (limitação
registrada no relatório do gate) e não trazem informação nova — só custo. Mantemos
a PRIMEIRA ocorrência de cada configuração, que é o pregão em que a notícia chega.

Execução: `python calibration/baseline_econ.py --dry-run` (grátis: amostra +
estimativa de custo) e depois sem a flag (requer ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import FUSO, tickers_ativos
from calibration.econ_calibration import (
    BLOCO_DIAS_UTEIS,
    _DEGRAD_INVALIDA,
    _DEGRAD_RESSALVA,
    CUTOFF_LIMPO,
    IC_META,
    N_BOOTSTRAP,
    SEED_BOOTSTRAP,
    TRAINING_CUTOFF,
    _retorno_excesso_5d,
    adicionar_blocos,
    baseline_sentimento_simples,
    bootstrap_ic_bloco,
    calcular_ic,
    chave_conjunto_noticias,
    classificar_degradacao,
    eh_evento_limpo,
    estimar_custo_usd,
)
from calibration.exec_infra import (
    _CALLS,
    CHECKPOINT_A_CADA,
    Checkpoint,
    MacroIndisponivel,
    avaliar_com_retry,
    calendario_pregoes,
    custo_da_chamada,
    instalar_captura,
    prefetch_macro,
)

logger = logging.getLogger("baseline_econ")

# ── Parâmetros travados ───────────────────────────────────────────────────────

MODELO = "claude-haiku-4-5-20251001"

JANELA_INICIO = pd.Timestamp("2024-01-01", tz=FUSO)
JANELA_FIM = pd.Timestamp("2025-12-31 23:59:59", tz=FUSO)

# Baseline só MEDE o prompt atual; não itera. O cap era US$3,00 e a estimativa
# dos 636 eventos deu US$3,02 (0,7% acima) — elevado a US$3,50 com autorização
# explícita, para não amputar a amostra que R1 pede por causa de 5 eventos. A
# estimativa é teto: os tokens/chamada vêm do gate, que rodou só nos 4 tickers
# com MAIS notícia. A trava de runtime segue valendo e interrompe a rodada.
CUSTO_HARD_CAP_USD = 3.50
SLEEP_ENTRE_CHAMADAS_S = 1.0     # rate-limit safe
N_MIN_EVENTOS = 150              # abaixo disso, suspeitar de bug de filtro
N_MIN_POR_GRUPO = 10             # piso p/ reportar IC de um ticker/mês isolado
N_PIORES_CASOS = 10
LATENCIA_ALERTA_S = 8.0          # acima disso, sinalizar (não bloqueia)

RESULTS_DIR = Path(__file__).parent / "results"
RELATORIO_MD = RESULTS_DIR / "RELATORIO_CALIBRACAO_ECON.md"
COMPLETO_CSV = RESULTS_DIR / "calibracao_baseline_completo.csv"
INTERMEDIARIO_CSV = RESULTS_DIR / "calibracao_baseline_intermediario.csv"
AMOSTRAGEM_JSON = RESULTS_DIR / "calibracao_baseline_amostragem.json"

# Versão de prompt cuja rodada produziu os artefatos da Etapa 1 (já pagos).
VERSAO_BASELINE = "2026-06-econA"


def _slug(versao: str) -> str:
    """Versão de prompt como pedaço seguro de nome de arquivo."""
    return re.sub(r"[^0-9A-Za-z._-]+", "-", versao).strip("-")


def caminho_checkpoint(versao: str) -> Path:
    """Checkpoint POR VERSÃO DE PROMPT.

    A chave interna do checkpoint é (modelo, evento_id) — não inclui o prompt.
    Se duas iterações dividissem o arquivo, a segunda retomaria os scores da
    primeira e reportaria ΔIC = 0 sem nenhum sinal de erro. A separação vem daqui.
    O baseline mantém o nome original: aquele arquivo custou US$2,79.
    """
    if versao == VERSAO_BASELINE:
        return INTERMEDIARIO_CSV
    return RESULTS_DIR / f"calibracao_{_slug(versao)}_intermediario.csv"


def caminho_completo(versao: str) -> Path:
    """CSV de resultado por versão de prompt (mesma razão do checkpoint)."""
    if versao == VERSAO_BASELINE:
        return COMPLETO_CSV
    return RESULTS_DIR / f"calibracao_{_slug(versao)}_completo.csv"


# ── Etapa A — amostragem de eventos ───────────────────────────────────────────


def amostrar_eventos(journal, janela_inicio: pd.Timestamp, janela_fim: pd.Timestamp,
                     dedup: bool = True, n_max: int = 0) -> tuple[list[dict], dict]:
    """Enumera os eventos (ticker-dia com notícia E com target) da janela.

    Determinístico: percorre pregões em ordem cronológica e tickers em ordem
    alfabética — sem sorteio, a amostra é o dataset inteiro filtrado. Com
    `dedup`, mantém só a PRIMEIRA ocorrência de cada conjunto de notícias por
    ticker. `n_max > 0` corta a amostra após N eventos (teto de custo).
    """
    from agents.econ import _MAX_NOTICIAS

    pregoes = calendario_pregoes(janela_inicio, janela_fim)
    eventos: list[dict] = []
    vistos: set[str] = set()
    n_candidatos = n_sem_noticia = n_duplicados = n_sem_y = 0

    for data in pregoes:
        for ticker in sorted(tickers_ativos(data)):
            n_candidatos += 1
            if n_max and len(eventos) >= n_max:
                continue
            try:
                noticias = journal.get_noticias(ticker, data)
            except Exception as e:  # fonte instável → pula, não aborta
                logger.warning("get_noticias falhou %s %s: %s", ticker, data.date(), e)
                continue
            if not noticias:
                n_sem_noticia += 1
                continue

            # O ECON só envia as _MAX_NOTICIAS primeiras: a chave de dedup tem de
            # descrever o conjunto REALMENTE avaliado, não o coletado.
            noticias = noticias[:_MAX_NOTICIAS]
            chave = chave_conjunto_noticias(ticker, noticias)
            if dedup and chave in vistos:
                n_duplicados += 1
                continue

            y = _retorno_excesso_5d(journal, ticker, data, ajuste_beta=True)
            if y is None:
                n_sem_y += 1
                continue

            vistos.add(chave)
            data_noticia = max(n.publicado_em for n in noticias)
            eventos.append({
                "ticker": ticker,
                "data": data,
                "noticias": noticias,
                "y": y,
                "data_noticia": data_noticia,
                "limpo": eh_evento_limpo(data_noticia),
                "score_lexical": baseline_sentimento_simples(noticias, apenas_titulo=True),
            })

    for i, ev in enumerate(eventos):
        ev["evento_id"] = i

    diag = {
        "n_eventos": len(eventos),
        "n_limpos": sum(ev["limpo"] for ev in eventos),
        "n_candidatos": n_candidatos,
        "n_sem_noticia": n_sem_noticia,
        "n_duplicados_descartados": n_duplicados,
        "n_sem_y": n_sem_y,
        "dedup": dedup,
        "n_max": n_max,
        "cutoff_limpo": str(CUTOFF_LIMPO),
    }
    return eventos, diag


# ── Fechamento offline a partir do checkpoint ─────────────────────────────────


def carregar_de_checkpoint(journal, caminho: Path | None = None,
                           modelo: str = MODELO,
                           versao: str = VERSAO_BASELINE) -> tuple[pd.DataFrame, dict]:
    """Reconstrói o dataset da rodada a partir do checkpoint, SEM tocar a rede.

    Existe porque re-derivar a amostra depende de yfinance (target) e BCB (macro),
    e ambos são intermitentes: numa janela de throttling a re-amostragem devolve
    um conjunto de eventos DIFERENTE, o que desalinharia o `evento_id` posicional
    e invalidaria a rodada já paga. O que o checkpoint não guarda —
    `data_noticia`, `score_lexical` — é recomputado das notícias, que são LOCAIS
    (CSV Bloomberg). O `y` já está gravado e não é recalculado.
    """
    from agents.econ import _MAX_NOTICIAS

    caminho = caminho if caminho is not None else caminho_checkpoint(versao)
    prev = pd.read_csv(caminho)
    prev = prev[prev["modelo"] == modelo]
    # O checkpoint é append-only: uma reavaliação grava uma linha NOVA para o
    # mesmo evento_id. Vale a última — sem isso o evento entraria duas vezes no IC.
    prev = prev.drop_duplicates(subset=["evento_id"], keep="last")
    linhas, n_sem_noticia = [], 0

    for _, r in prev.iterrows():
        data = pd.Timestamp(r["data"])
        noticias = journal.get_noticias(str(r["ticker"]), data)[:_MAX_NOTICIAS]
        if not noticias:
            n_sem_noticia += 1
            logger.warning("checkpoint sem notícia recuperável: %s %s",
                           r["ticker"], data.date())
            continue
        data_noticia = max(n.publicado_em for n in noticias)
        chamou_api = _foi_avaliado(r)
        linhas.append({
            "evento_id": int(r["evento_id"]), "ticker": str(r["ticker"]),
            "data": data, "data_noticia": data_noticia, "y": float(r["y_realizado"]),
            "limpo": eh_evento_limpo(data_noticia),
            "score_lexical": baseline_sentimento_simples(noticias, apenas_titulo=True),
            "score": float(r["score"]), "n_noticias": len(noticias),
            "confianca": float(r.get("confianca", float("nan"))),
            "tem_evento": bool(r.get("tem_evento", True)),
            # `or ""` não serve: NaN é truthy e viraria a string literal "nan".
            "justificativa": ("" if pd.isna(r.get("justificativa"))
                              else str(r.get("justificativa"))),
            "degradacao": (r.get("degradacao") if pd.notna(r.get("degradacao"))
                           else (None if chamou_api else "sem_chamada_api")),
            "tokens_in": int(r["tokens_in"]), "tokens_out": int(r["tokens_out"]),
            "latencia_llm_s": float(r["latencia_llm_s"]),
            "custo_usd": float(r["custo_usd"]), "chamou_api": chamou_api,
        })

    df = pd.DataFrame(linhas)
    diag = _carregar_diag_amostragem()
    diag.update({"n_avaliados": len(df), "n_limpos_avaliados": int(df["limpo"].sum()),
                 "n_sem_noticia_recuperavel": n_sem_noticia})
    return df, diag


def recuperar_justificativas(journal, df: pd.DataFrame, alvos: pd.DataFrame,
                             custo_acumulado: list[float],
                             checkpoint: Checkpoint | None = None) -> pd.DataFrame:
    """Re-busca a justificativa dos eventos em `alvos` que estejam sem ela.

    O checkpoint antigo não persistia justificativa, e é justamente ela que
    orienta a iteração de prompt da Etapa 2. Reavalia com o MESMO prompt e modelo
    (temperature=0), então o texto corresponde ao score já medido; o SCORE
    permanece o do checkpoint — não sobrescrevemos medição com re-medição.
    """
    from agents.econ import EconAgent, _MAX_NOTICIAS

    agent = EconAgent(journal=journal, model=MODELO)
    out = df.copy()
    for idx, r in alvos.iterrows():
        if not pd.isna(r.get("justificativa")) and str(r["justificativa"]).strip():
            continue
        noticias = journal.get_noticias(r["ticker"], pd.Timestamp(r["data"]))[:_MAX_NOTICIAS]
        if not noticias:
            continue
        antes = len(_CALLS)
        try:
            score = agent.avaliar(r["ticker"], pd.Timestamp(r["data"]),
                                  noticias_override=noticias)
        except Exception as e:
            logger.warning("re-busca de justificativa falhou %s %s: %s",
                           r["ticker"], r["data"], e)
            continue
        rec = _CALLS[-1] if len(_CALLS) > antes else None
        custo_acumulado[0] += custo_da_chamada(rec, MODELO)
        out.loc[idx, "justificativa"] = score.justificativa
        out.loc[idx, "score_rebusca"] = score.score_total
        if checkpoint is not None:
            # Persistir: senão a justificativa se perde na próxima regeração do
            # relatório e a re-busca vira trabalho repetido a cada `--finalizar`.
            # O SCORE gravado continua sendo o medido na rodada, não o re-medido.
            checkpoint.feitos.pop((MODELO, int(r["evento_id"])), None)
            checkpoint.registrar({
                "evento_id": int(r["evento_id"]), "ticker": r["ticker"],
                "data": pd.Timestamp(r["data"]).isoformat(),
                "y_realizado": float(r["y"]), "modelo": MODELO,
                "score": float(r["score"]), "latencia_llm_s": r.get("latencia_llm_s"),
                "tokens_in": int(r["tokens_in"]), "tokens_out": int(r["tokens_out"]),
                "custo_usd": float(r["custo_usd"]), "confianca": score.confianca,
                "tem_evento": True, "degradacao": None,
                "justificativa": score.justificativa,
            })
    if checkpoint is not None:
        checkpoint.flush()
    return out


def _foi_avaliado(r) -> bool:
    """A linha representa uma opinião REAL do LLM?

    `tokens_in == 0` sozinho NÃO serve como sinal de falha: um acerto do cache em
    disco do EconAgent também não gasta token, e é uma avaliação legítima. Quando
    o checkpoint traz o diagnóstico (schema novo), ele é a autoridade —
    `degradacao` preenchida é falha, justificativa preenchida é avaliação. Só em
    linhas de schema antigo caímos na heurística de tokens.
    """
    if pd.notna(r.get("degradacao")) and str(r.get("degradacao")).strip():
        return False
    justificativa = r.get("justificativa")
    if pd.notna(justificativa) and str(justificativa).strip():
        return True
    return int(r["tokens_in"]) > 0


def reavaliar_degradadas(journal, checkpoint: Checkpoint, df: pd.DataFrame,
                         custo_acumulado: list[float]) -> int:
    """Re-avalia os eventos que degradaram (não chegaram ao LLM) e regrava.

    Reconstrói o evento a partir do próprio checkpoint — sem re-amostrar, que é o
    passo dependente de yfinance/BCB e, num dia de throttling, devolveria outro
    conjunto. Só toca as linhas degradadas: as válidas já custaram e não mudam.
    """
    from agents.econ import EconAgent, _MAX_NOTICIAS

    agent = EconAgent(journal=journal, model=MODELO)
    alvos = df[~df["chamou_api"].astype(bool)]
    n_ok = 0
    for _, r in alvos.iterrows():
        data = pd.Timestamp(r["data"])
        noticias = journal.get_noticias(r["ticker"], data)[:_MAX_NOTICIAS]
        if not noticias:
            continue
        ev = {"ticker": r["ticker"], "data": data, "noticias": noticias}
        antes = len(_CALLS)
        try:
            score = avaliar_com_retry(agent, ev, degradou=_degradou)
        except Exception as e:
            logger.warning("reavaliação falhou %s %s: %s", r["ticker"], data.date(), e)
            continue
        rec = _CALLS[-1] if len(_CALLS) > antes else None
        if rec is None:  # degradou de novo → deixa como está, será reportado
            continue
        custo_acumulado[0] += custo_da_chamada(rec, MODELO)
        checkpoint.feitos.pop((MODELO, int(r["evento_id"])), None)
        checkpoint.registrar({
            "evento_id": int(r["evento_id"]), "ticker": r["ticker"],
            "data": data.isoformat(), "y_realizado": float(r["y"]), "modelo": MODELO,
            "score": score.score_total, "latencia_llm_s": rec.get("latencia_s"),
            "tokens_in": rec.get("input_tokens") or 0,
            "tokens_out": rec.get("output_tokens") or 0,
            "custo_usd": custo_da_chamada(rec, MODELO),
            "confianca": score.confianca, "tem_evento": score.tem_evento,
            "degradacao": classificar_degradacao(score),
            "justificativa": score.justificativa,
        })
        n_ok += 1
        if n_ok % CHECKPOINT_A_CADA == 0:
            print(f"    reavaliadas {n_ok}/{len(alvos)} | "
                  f"custo acum US$ {custo_acumulado[0]:.4f}", flush=True)
        time.sleep(SLEEP_ENTRE_CHAMADAS_S)
    checkpoint.flush()
    return n_ok


def _degradou(score) -> bool:
    """Avaliação que não representa opinião do LLM (degradação graciosa)."""
    return classificar_degradacao(score) is not None


def _carregar_diag_amostragem() -> dict:
    """Contadores da etapa de amostragem, se a rodada os tiver salvo. Ausentes →
    o relatório declara 'n/d' em vez de inventar número."""
    import json

    if AMOSTRAGEM_JSON.exists():
        return json.loads(AMOSTRAGEM_JSON.read_text())
    return {"n_eventos": None, "n_limpos": None, "n_candidatos": None,
            "n_sem_noticia": None, "n_duplicados_descartados": None,
            "n_sem_y": None, "dedup": True, "n_max": 0,
            "cutoff_limpo": str(CUTOFF_LIMPO)}


# ── Etapa B — avaliação (paga) ────────────────────────────────────────────────


def avaliar_eventos(journal, eventos: list[dict], custo_acumulado: list[float],
                    checkpoint: Checkpoint) -> list[dict]:
    """Avalia todos os eventos com o prompt ATUAL, capturando score, tokens e
    latência isolada do LLM. Retoma do checkpoint; retry com backoff (pula o
    evento após esgotar); hard cap levanta RuntimeError (o caller reporta)."""
    from agents.econ import EconAgent

    agent = EconAgent(journal=journal, model=MODELO)
    linhas = []
    n_novos = 0

    for ev in eventos:
        eid = ev["evento_id"]
        if checkpoint.ja_feito(MODELO, eid):
            prev = checkpoint.linha_feita(MODELO, eid)
            if _confere_identidade(ev, prev):
                linhas.append(_linha_retomada(ev, prev))
                custo_acumulado[0] += float(prev["custo_usd"])
                continue
            # `evento_id` é POSICIONAL: se a amostragem mudou entre execuções
            # (yfinance instável muda quem tem target), o id aponta para outro
            # ticker-dia. Reavaliar é caro; colar o score errado é pior.
            logger.warning("Checkpoint desalinhado no evento %d (%s %s ≠ %s %s); "
                           "reavaliando", eid, ev["ticker"], ev["data"].date(),
                           prev.get("ticker"), prev.get("data"))

        antes = len(_CALLS)
        try:
            score = avaliar_com_retry(agent, ev)
        except Exception as e:  # esgotou retries → pula, NÃO aborta a rodada
            logger.warning("avaliar DESISTIU %s %s: %s", ev["ticker"], ev["data"].date(), e)
            continue

        rec = _CALLS[-1] if len(_CALLS) > antes else None
        custo = custo_da_chamada(rec, MODELO)
        custo_acumulado[0] += custo
        linhas.append({
            **_campos_do_evento(ev),
            "score": score.score_total,
            "confianca": score.confianca,
            "tem_evento": score.tem_evento,
            "n_noticias": score.n_noticias,
            "justificativa": score.justificativa,
            "tokens_in": (rec or {}).get("input_tokens") or 0,
            "tokens_out": (rec or {}).get("output_tokens") or 0,
            "latencia_llm_s": (rec or {}).get("latencia_s", float("nan")),
            "custo_usd": custo,
            "chamou_api": rec is not None,
            "degradacao": classificar_degradacao(score),
        })
        checkpoint.registrar({
            "evento_id": eid, "ticker": ev["ticker"], "data": ev["data"].isoformat(),
            "y_realizado": ev["y"], "modelo": MODELO, "score": score.score_total,
            "latencia_llm_s": (rec or {}).get("latencia_s", float("nan")),
            "tokens_in": (rec or {}).get("input_tokens") or 0,
            "tokens_out": (rec or {}).get("output_tokens") or 0,
            "custo_usd": custo,
            "confianca": score.confianca, "tem_evento": score.tem_evento,
            "degradacao": classificar_degradacao(score),
            "justificativa": score.justificativa,
        })
        n_novos += 1
        if n_novos % CHECKPOINT_A_CADA == 0:
            # flush explícito: com stdout redirecionado a arquivo o Python
            # bufferiza, e o progresso de uma rodada de horas ficaria invisível.
            print(f"    +{n_novos}/{len(eventos)} avaliações | "
                  f"custo acum US$ {custo_acumulado[0]:.4f}", flush=True)

        if custo_acumulado[0] > CUSTO_HARD_CAP_USD:
            checkpoint.flush()
            raise RuntimeError(
                f"HARD CAP estourado: US$ {custo_acumulado[0]:.2f} > "
                f"US$ {CUSTO_HARD_CAP_USD:.2f}. Parando — checkpoint salvo, dá pra retomar."
            )
        time.sleep(SLEEP_ENTRE_CHAMADAS_S)

    checkpoint.flush()
    return linhas


def conferir_alinhamento(eventos: list[dict], checkpoint: Checkpoint,
                         modelo: str = MODELO) -> dict:
    """Quantas linhas do checkpoint ainda descrevem o evento que o `evento_id`
    aponta depois de uma nova amostragem.

    Pré-voo de uma retomada: `evento_id` é posicional, então se o yfinance
    derrubar um ticker a amostra inteira desloca e o checkpoint deixa de valer.
    Melhor abortar de graça do que descobrir isso reavaliando 600 eventos pagos.
    """
    total = alinhados = 0
    for ev in eventos:
        if not checkpoint.ja_feito(modelo, ev["evento_id"]):
            continue
        total += 1
        alinhados += _confere_identidade(ev, checkpoint.linha_feita(modelo, ev["evento_id"]))
    return {"n_checkpoint": total, "n_alinhados": alinhados,
            "n_desalinhados": total - alinhados,
            "taxa": alinhados / total if total else 1.0}


def _confere_identidade(ev: dict, prev: dict) -> bool:
    """A linha do checkpoint descreve MESMO este evento? Compara ticker e data —
    `evento_id` sozinho não basta, porque é posicional."""
    return (str(prev.get("ticker")) == ev["ticker"]
            and pd.Timestamp(prev.get("data")) == ev["data"])


def _campos_do_evento(ev: dict) -> dict:
    return {
        "evento_id": ev["evento_id"], "ticker": ev["ticker"], "data": ev["data"],
        "data_noticia": ev["data_noticia"], "y": ev["y"], "limpo": ev["limpo"],
        "score_lexical": ev["score_lexical"],
    }


def _linha_retomada(ev: dict, prev: dict) -> dict:
    """Linha reconstruída do checkpoint.

    Campos de diagnóstico (confiança, justificativa, degradação) vêm do próprio
    checkpoint quando gravados. Linhas de um checkpoint em schema antigo não os
    têm — nesse caso ficam neutros, e o relatório mostra a lacuna em vez de
    inventar valor."""
    def _ou(campo, default):
        v = prev.get(campo)
        return default if v is None or (isinstance(v, float) and pd.isna(v)) else v

    return {
        **_campos_do_evento(ev),
        "score": float(prev["score"]),
        "confianca": float(_ou("confianca", float("nan"))),
        "tem_evento": bool(_ou("tem_evento", True)),
        "n_noticias": len(ev["noticias"]),
        "justificativa": str(_ou("justificativa", "")),
        "tokens_in": int(prev["tokens_in"]), "tokens_out": int(prev["tokens_out"]),
        "latencia_llm_s": float(prev["latencia_llm_s"]),
        "custo_usd": float(prev["custo_usd"]), "chamou_api": True,
        "degradacao": _ou("degradacao", None),
    }


# ── Etapa C — métricas ────────────────────────────────────────────────────────


def separar_validas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Divide o dataset entre avaliações REAIS e degradações, com a taxa e o
    veredito de validade.

    Uma degradação tem score 0.0 sem que o LLM tenha opinado: mantê-la no IC
    injeta um bloco de zeros correlacionado no tempo (a falha dura enquanto a API
    está ruim) e enviesa a medição para zero."""
    if "chamou_api" not in df.columns:
        return df, {"n_total": len(df), "n_validas": len(df), "n_degradadas": 0,
                    "taxa": 0.0, "ressalva": False, "invalida": False}
    validas = df[df["chamou_api"].astype(bool)]
    n_degrad = len(df) - len(validas)
    taxa = n_degrad / len(df) if len(df) else 0.0
    return validas, {
        "n_total": len(df), "n_validas": len(validas), "n_degradadas": n_degrad,
        "taxa": taxa,
        "ressalva": taxa > _DEGRAD_RESSALVA, "invalida": taxa > _DEGRAD_INVALIDA,
        "limiar_ressalva": _DEGRAD_RESSALVA, "limiar_invalida": _DEGRAD_INVALIDA,
    }


def calcular_metricas(df: pd.DataFrame) -> dict:
    """IC completo, IC limpo, IC lexical (B0) e GAP — todos por block bootstrap,
    sobre as avaliações VÁLIDAS (degradações não entram)."""
    df, _ = separar_validas(df)
    dfb = adicionar_blocos(df, "data", data_inicio=JANELA_INICIO)
    limpo = dfb[dfb["limpo"]]

    def _ic(sub: pd.DataFrame, col: str) -> dict:
        if len(sub) < 2:
            return {"ic": float("nan"), "ic95_low": float("nan"),
                    "ic95_high": float("nan"), "n": len(sub), "n_blocos": 0}
        return bootstrap_ic_bloco(sub, col, "y")

    econ_completo, econ_limpo = _ic(dfb, "score"), _ic(limpo, "score")
    lex_completo, lex_limpo = _ic(dfb, "score_lexical"), _ic(limpo, "score_lexical")
    return {
        "econ_completo": econ_completo, "econ_limpo": econ_limpo,
        "lexical_completo": lex_completo, "lexical_limpo": lex_limpo,
        "gap_completo": econ_completo["ic"] - lex_completo["ic"],
        "gap_limpo": econ_limpo["ic"] - lex_limpo["ic"],
        "cobertura_lexical": float((df["score_lexical"] != 0).mean()),
    }


def diagnosticar(df: pd.DataFrame) -> dict:
    """Onde o ECON erra: IC por ticker, por mês, e os piores casos individuais.
    Degradações ficam de fora — elas dizem sobre a API, não sobre o prompt."""
    df, _ = separar_validas(df)
    por_ticker = _ic_por_grupo(df, df["ticker"])
    meses = df["data"].apply(lambda d: pd.Timestamp(d).strftime("%Y-%m"))
    por_mes = _ic_por_grupo(df, meses)

    # Discordância de RANK (mesma moeda do Spearman): score no topo com retorno no
    # fundo (ou o inverso) dá produto negativo grande.
    d = df.copy()
    rs = d["score"].rank(pct=True) - 0.5
    ry = d["y"].rank(pct=True) - 0.5
    d["discordancia"] = rs * ry
    piores = d.nsmallest(N_PIORES_CASOS, "discordancia")
    return {
        "ic_por_ticker": por_ticker,
        "ic_por_mes": por_mes,
        "piores_casos": piores,
        "n_min_por_grupo": N_MIN_POR_GRUPO,
    }


def _ic_por_grupo(df: pd.DataFrame, chave) -> pd.DataFrame:
    """IC por grupo, com a razão de um IC indefinido explicitada.

    Spearman é indefinido quando o score não varia dentro do grupo — e isso não é
    ruído de amostra pequena, é um ACHADO: o prompt devolveu a mesma nota para
    todos os eventos do período. Reportar 'nan' esconderia justamente isso."""
    linhas = []
    for nome, g in df.groupby(chave):
        if len(g) < N_MIN_POR_GRUPO:
            continue
        ic = calcular_ic(g["score"], g["y"])
        obs = ""
        if not (ic == ic):
            obs = (f"score constante ({g['score'].iloc[0]:+.2f}) em todo o grupo"
                   if g["score"].nunique() == 1 else "IC indefinido")
        linhas.append({"grupo": nome, "n": len(g), "ic": ic,
                       "zeros": int((g["score"] == 0).sum()), "obs": obs})
    out = pd.DataFrame(linhas)
    return out.sort_values("ic", na_position="last") if not out.empty else out


# ── Etapa D — relatório ───────────────────────────────────────────────────────


def _fmt_ic(r: dict) -> str:
    if not (r["ic"] == r["ic"]):
        return "n/d (amostra insuficiente)"
    cruza = r["ic95_low"] <= 0 <= r["ic95_high"]
    marca = "cruza zero" if cruza else "**não cruza zero**"
    return (f"{r['ic']:+.4f} [{r['ic95_low']:+.4f}, {r['ic95_high']:+.4f}] "
            f"— {marca} (n={r['n']}, blocos={r['n_blocos']})")


def _veredito(ic: float) -> str:
    if not (ic == ic):
        return "INDETERMINADO"
    if ic > IC_META:
        return f"SUFICIENTE (> {IC_META})"
    if ic >= 0.10:
        return f"ACEITÁVEL, MAS MELHORAR (0.10–{IC_META})"
    if ic >= 0:
        return "PRECISA DE ITERAÇÃO SIGNIFICATIVA (< 0.10)"
    return "⚠️ NEGATIVO — prompt confuso ou bug metodológico"


def gerar_relatorio(df: pd.DataFrame, diag: dict, met: dict, dgn: dict,
                    versao_prompt: str, custo_total: float, parcial: bool) -> str:
    df_todos = df
    df, degrad = separar_validas(df)
    lat = df["latencia_llm_s"].dropna()
    tk = df["ticker"].value_counts()
    meses = df["data"].apply(lambda d: pd.Timestamp(d).strftime("%Y-%m")).value_counts().sort_index()

    L = [f"# Calibração do ECON — baseline do prompt `{versao_prompt}`\n"]
    if parcial:
        n_amostrado = diag.get("n_eventos")
        alvo = f"{n_amostrado} eventos amostrados" if n_amostrado else "a amostra completa"
        L.append(f"> ⚠️ **RODADA PARCIAL** — {len(df_todos)} de {alvo} foram avaliados. "
                 "Ver 'Limitações' para o motivo e o que isso afeta.\n")
    if degrad["invalida"]:
        L.append(f"> ⛔ **CALIBRAÇÃO COM VALIDADE COMPROMETIDA** — "
                 f"{degrad['n_degradadas']} de {degrad['n_total']} avaliações "
                 f"({degrad['taxa']:.1%}) NÃO chegaram ao LLM e voltaram como "
                 f"score neutro (degradação graciosa do EconAgent). O limiar de "
                 f"invalidez é {degrad['limiar_invalida']:.0%}. As degradações "
                 "foram EXCLUÍDAS das métricas abaixo, mas o baseline precisa ser "
                 "re-rodado antes de virar régua da Etapa 2.\n")
    elif degrad["ressalva"]:
        L.append(f"> ⚠️ **RESSALVA** — {degrad['taxa']:.1%} de degradações "
                 f"(limiar {degrad['limiar_ressalva']:.0%}); excluídas das métricas.\n")
    L.append("Etapa 1: mede o prompt ATUAL sem alterá-lo. Estabelece o número que "
             "as iterações da Etapa 2 precisam superar.\n")
    L.append(f"- Data/hora: {pd.Timestamp.now(tz=FUSO).isoformat(timespec='seconds')}")
    L.append("- Fonte de notícias: Bloomberg-only (GDELT/NewsAPI neutralizados)\n")

    L.append("## 1. Parâmetros\n")
    L.append(f"- Modelo: `{MODELO}` (Sonnet descartado no gate de custo, ΔIC = -0.0298)")
    L.append(f"- Versão do prompt: `{versao_prompt}` (NÃO modificada nesta etapa)")
    L.append(f"- Janela: {JANELA_INICIO.date()} → {JANELA_FIM.date()}")
    L.append(f"- N avaliado: {len(df_todos)} (de {diag['n_eventos']} eventos "
             f"amostrados) | **N válido para métricas: {len(df)}**")
    L.append("- Métrica: IC de Spearman(score_total, retorno beta-ajustado 5du)")
    L.append(f"- Bootstrap: block-by-date, blocos de {BLOCO_DIAS_UTEIS} dias úteis, "
             f"{N_BOOTSTRAP} iterações, seed={SEED_BOOTSTRAP}")
    L.append(f"- Fronteira LIMPA: `data_noticia_mais_recente >= {CUTOFF_LIMPO}` "
             f"(pós training cutoff do Haiku 4.5, {TRAINING_CUTOFF.date()})")
    L.append(f"- Hard cap: US$ {CUSTO_HARD_CAP_USD:.2f} | "
             f"**custo real: US$ {custo_total:.4f}** "
             f"({100 * custo_total / CUSTO_HARD_CAP_USD:.1f}% do cap)")
    L.append(f"- Deduplicação por conjunto de notícias: "
             f"{'ATIVA' if diag['dedup'] else 'desligada'}\n")

    def _n(chave) -> str:
        v = diag.get(chave)
        return "n/d" if v is None else str(v)

    L.append("## 2. Amostra\n")
    L.append(f"- Candidatos (ticker-dia): {_n('n_candidatos')}; sem notícia: "
             f"{_n('n_sem_noticia')}; duplicados descartados: "
             f"{_n('n_duplicados_descartados')}; sem target y: {_n('n_sem_y')}")
    L.append(f"- Eventos amostrados: {_n('n_eventos')} — dos quais LIMPOS "
             f"(notícia ≥ {CUTOFF_LIMPO.date()}): {_n('n_limpos')}")
    L.append(f"- Eventos AVALIADOS (base deste relatório): {len(df)} — "
             f"limpos: {int(df['limpo'].sum())}")
    if diag.get("fonte"):
        L.append(f"- Procedência dos contadores de amostragem: {diag['fonte']}")
    L.append(f"- Tickers cobertos: {df['ticker'].nunique()}")
    L.append("- Por mês: " + ", ".join(f"{m}={c}" for m, c in meses.items()))
    L.append("- Por ticker: " + ", ".join(f"{t}={c}" for t, c in tk.items()))
    L.append(f"- Dispersão do score: média {df['score'].mean():+.4f}, "
             f"desvio {df['score'].std():.4f}, min {df['score'].min():+.2f}, "
             f"max {df['score'].max():+.2f}, zeros {int((df['score'] == 0).sum())}")
    L.append(f"- Dispersão do target y: média {df['y'].mean():+.4f}, "
             f"desvio {df['y'].std():.4f}\n")

    L.append(f"## 3. Baseline — prompt `{versao_prompt}`\n")
    L.append("| Métrica | IC de Spearman [IC95] |")
    L.append("|---|---|")
    L.append(f"| **IC completo** (2024-2025) | {_fmt_ic(met['econ_completo'])} |")
    L.append(f"| **IC limpo** (notícia ≥ {CUTOFF_LIMPO.date()}) | {_fmt_ic(met['econ_limpo'])} |")
    L.append(f"| IC lexical B0 — completo | {_fmt_ic(met['lexical_completo'])} |")
    L.append(f"| IC lexical B0 — limpo | {_fmt_ic(met['lexical_limpo'])} |")
    L.append("")
    L.append(f"- **GAP (ECON − lexical), completo: {met['gap_completo']:+.4f}**")
    L.append(f"- **GAP (ECON − lexical), limpo: {met['gap_limpo']:+.4f}**")
    L.append(f"- Cobertura do léxico B0 (eventos com score ≠ 0): "
             f"{met['cobertura_lexical']:.1%}")
    L.append(f"- Veredito (IC completo): **{_veredito(met['econ_completo']['ic'])}**")
    L.append(f"- Veredito (IC limpo): **{_veredito(met['econ_limpo']['ic'])}**")
    L.append(f"- Latência do LLM: mediana {lat.median():.2f}s | "
             f"P95 {lat.quantile(0.95):.2f}s" if len(lat) else "- Latência: n/d")
    L.append(f"- Taxa de fallback (`tem_evento=False`): "
             f"{(~df['tem_evento'].astype(bool)).mean():.1%}")
    L.append(f"- **Taxa de degradação (não chegou ao LLM): "
             f"{degrad['taxa']:.1%}** ({degrad['n_degradadas']}/{degrad['n_total']}) "
             f"— limiares: ressalva > {degrad['limiar_ressalva']:.0%}, "
             f"inválida > {degrad['limiar_invalida']:.0%}")
    L.append(f"- Tokens médios: {df['tokens_in'].mean():.0f} in / "
             f"{df['tokens_out'].mean():.0f} out | custo médio US$ "
             f"{df['custo_usd'].mean():.6f}/chamada\n")
    if len(lat) and lat.median() > LATENCIA_ALERTA_S:
        L.append(f"> ⚠️ Latência mediana {lat.median():.1f}s > {LATENCIA_ALERTA_S:.0f}s "
                 "— possível rate limit ou degradação do modelo. Não bloqueia.\n")

    L.append("## 4. Diagnóstico — onde o ECON está errando\n")
    L.extend(_tabela_grupo(dgn["ic_por_ticker"], "Ticker",
                           f"IC por ticker (só grupos com n ≥ {N_MIN_POR_GRUPO}, pior primeiro)"))
    L.extend(_tabela_grupo(dgn["ic_por_mes"], "Mês",
                           f"IC por mês (só grupos com n ≥ {N_MIN_POR_GRUPO}, pior primeiro)"))
    L.append(f"\n### {N_PIORES_CASOS} casos de maior discordância de rank\n")
    L.append("Score no topo com retorno no fundo (ou o inverso) — a mesma moeda do "
             "Spearman. `justificativa` truncada.\n")
    L.append("| ticker | data | score | y | justificativa |")
    L.append("|---|---|---|---|---|")
    for _, r in dgn["piores_casos"].iterrows():
        just = str(r.get("justificativa", "")).replace("|", "/")[:110]
        L.append(f"| {r['ticker']} | {pd.Timestamp(r['data']).date()} | "
                 f"{r['score']:+.2f} | {r['y']:+.4f} | {just} |")

    L.append("\n## 5. Próximos passos (Etapa 2 — iteração de prompt)\n")
    L.extend(_eixos_de_iteracao(df, met, dgn))

    L.append("\n## Limitações desta rodada\n")
    L.append("- **Contaminação parcial em 2024-2025 até jul/2025**: está dentro do "
             "training cutoff do Haiku 4.5. Por isso o IC limpo é reportado à parte "
             "— é o número honesto; o completo é teto otimista com mais poder.")
    L.append("- **Concentração por ticker**: PETR4/ITUB4/BBDC4/VALE3 dominam o "
             "dataset Bloomberg. O IC agregado pesa mais esses nomes.")
    L.append("- **Deduplicação muda a unidade de observação** para (ticker, "
             "configuração de notícias), não (ticker, dia). Reduz custo e a "
             "dependência serial, mas não a elimina — daí o block bootstrap.")
    if parcial:
        ultimo = pd.Timestamp(df["data"].max()).date()
        L.append(f"- **Rodada interrompida em {ultimo}**: a execução travou horas "
                 "em rate-limit do BCB SGS (`get_macro` cacheia por data, então "
                 "centenas de eventos viraram centenas de buscas da série inteira) "
                 "e, na retomada, o yfinance entrou em throttling — re-amostrar "
                 "devolveria um conjunto de eventos diferente. O relatório foi "
                 "fechado a partir do checkpoint. Efeito prático: **dez/2025 fica "
                 "sub-representado** na janela limpa. Corrigido para as próximas "
                 "rodadas com `prefetch_macro` (uma busca por rodada).")
    if degrad["n_degradadas"]:
        deg = df_todos[~df_todos["chamou_api"].astype(bool)]
        per = (f"{pd.Timestamp(deg['data'].min()).date()} a "
               f"{pd.Timestamp(deg['data'].max()).date()}")
        L.append(f"- **{degrad['n_degradadas']} avaliações não chegaram ao LLM** "
                 f"(eventos de {per}): o EconAgent degrada graciosamente — devolve "
                 "score neutro 0.0 com aviso, sem levantar exceção — e o retry da "
                 "rodada só capturava exceção, então a falha entrou no dataset "
                 "disfarçada de avaliação. Detectadas por `tokens_in == 0` e "
                 "excluídas das métricas. `avaliar_com_retry` passou a aceitar um "
                 "predicado de degradação para reagir a isso.")
    if (df["justificativa"].astype(str).str.strip() == "").any():
        n_sem = int((df["justificativa"].astype(str).str.strip() == "").sum())
        L.append(f"- **{n_sem} avaliações sem justificativa persistida**: o "
                 "checkpoint original não gravava esse campo (schema corrigido "
                 "depois). Onde a tabela de piores casos mostra justificativa, ela "
                 "foi RE-BUSCADA com o mesmo prompt e modelo.")
    L.append("- Target usa Close AJUSTADO (`_retorno_excesso_5d`), enquanto o "
             "MATH&ML usa Close_raw; diferença esperada é pequena.")
    csv = caminho_completo(versao_prompt)
    L.append(f"\nCSV completo: `{csv.relative_to(csv.parents[2])}`")
    return "\n".join(L) + "\n"


def _tabela_grupo(tab: pd.DataFrame, rotulo: str, titulo: str) -> list[str]:
    if tab.empty:
        return [f"\n### {titulo}\n", "_Nenhum grupo atingiu o piso de amostra._"]
    linhas = [f"\n### {titulo}\n",
              f"| {rotulo} | n | scores zerados | IC | observação |",
              "|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        ic = "n/d" if not (r["ic"] == r["ic"]) else f"{r['ic']:+.4f}"
        linhas.append(f"| {r['grupo']} | {int(r['n'])} | {int(r['zeros'])} | "
                      f"{ic} | {r['obs']} |")
    return linhas


def _eixos_de_iteracao(df: pd.DataFrame, met: dict, dgn: dict) -> list[str]:
    """Sugestões de eixo de prompt derivadas do que os dados mostram."""
    eixos = []
    if abs(df["score"].std()) < 0.20:
        eixos.append("**Dispersão do score**: o desvio-padrão está baixo — o prompt "
                     "empurra tudo para perto de zero. Instruir uso da escala cheia "
                     "[-1,+1] e ancorar exemplos de -0.8 e +0.8 tende a aumentar a "
                     "variância de rank, que é o que o Spearman enxerga.")
    if met["gap_completo"] < 0.03:
        eixos.append("**GAP baixo contra o léxico**: o ECON ainda não está fazendo "
                     "muito além de dicionário de sentimento. Reforçar o pedido de "
                     "MECANISMO (efeito em caixa/margem/múltiplo) e penalizar "
                     "explicitamente a leitura de tom da manchete.")
    pior = dgn["ic_por_ticker"]
    if not pior.empty and pior.iloc[0]["ic"] < 0:
        nomes = ", ".join(pior[pior["ic"] < 0]["grupo"].head(4))
        eixos.append(f"**Tickers com IC negativo** ({nomes}): inspecionar as "
                     "justificativas desses nomes — pode ser setor em que o prompt "
                     "inverte o sinal (ex.: notícia de dividendo vs. de capex).")
    if (df["score"] == 0).mean() > 0.15:
        eixos.append("**Excesso de score exatamente zero**: parcela grande de "
                     "eventos vira neutro. Investigar se é notícia sem conteúdo "
                     "material (então o filtro do JOURNAL é o alvo) ou timidez do "
                     "prompt (então o alvo é a instrução).")
    eixos.append("**Horizonte explícito**: reforçar que a janela é de 5 dias úteis "
                 "e que notícia já precificada no dia deve pontuar perto de zero.")
    eixos.append("**Protocolo da Etapa 2**: uma mudança por iteração, bump de "
                 "`_PROMPT_VERSION` (invalida o cache), re-rodar esta mesma rotina e "
                 "comparar contra este baseline. Parar em IC > 0.15, 10 iterações ou "
                 "hard cap de US$15.")
    return [f"{i}. {e}" for i, e in enumerate(eixos, 1)]


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Baseline da calibração do ECON (Etapa 1)")
    p.add_argument("--dry-run", action="store_true",
                   help="amostra e estima o custo, sem gastar nada")
    p.add_argument("--n", type=int, default=0, help="teto de eventos (0 = todos)")
    p.add_argument("--sem-dedup", action="store_true",
                   help="não deduplica eventos por conjunto de notícias")
    p.add_argument("--data-inicio", default=None, help="YYYY-MM-DD (default 2024-01-01)")
    p.add_argument("--data-fim", default=None, help="YYYY-MM-DD (default 2025-12-31)")
    p.add_argument("--cascata-completa", action="store_true",
                   help="usa GDELT+NewsAPI além do Bloomberg (default: Bloomberg-only)")
    p.add_argument("--sem-relatorio", action="store_true",
                   help="roda e grava os CSVs, mas NÃO escreve o relatório — usado "
                        "pela Etapa 2, que monta um relatório multi-versão")
    p.add_argument("--finalizar", action="store_true",
                   help="gera o relatório a partir do checkpoint, sem rede nem API")
    p.add_argument("--reavaliar-degradadas", action="store_true",
                   help="com --finalizar: reavalia os eventos que não chegaram ao "
                        "LLM (score 0.0 por degradação graciosa)")
    p.add_argument("--recuperar-justificativas", action="store_true",
                   help="com --finalizar: re-busca a justificativa dos piores casos "
                        "(custo baixo; só onde o checkpoint antigo não a gravou)")
    return p.parse_args(argv)


def _finalizar(journal, versao_prompt: str, recuperar: bool = False,
               reavaliar: bool = False) -> int:
    """Fecha a rodada offline: métricas e relatório a partir do checkpoint."""
    print("\n[F] Reconstruindo dataset do checkpoint (sem rede, sem API)...", flush=True)
    df, diag = carregar_de_checkpoint(journal, versao=versao_prompt)
    if df.empty:
        print("[F] ⛔ checkpoint vazio ou sem linhas do modelo — nada a finalizar.")
        return 6
    custo_total = float(df["custo_usd"].sum())
    n_amostrado = diag.get("n_eventos")
    parcial = bool(n_amostrado) and len(df) < n_amostrado
    print(f"[F] {len(df)} avaliações ({int(df['limpo'].sum())} limpas) | "
          f"custo já gasto US$ {custo_total:.4f}", flush=True)

    if reavaliar and (~df["chamou_api"].astype(bool)).any():
        gasto = [0.0]
        instalar_captura()
        n_alvo = int((~df["chamou_api"].astype(bool)).sum())
        print(f"[F] Reavaliando {n_alvo} eventos degradados...", flush=True)
        cp = Checkpoint(caminho_checkpoint(versao_prompt))
        n_ok = reavaliar_degradadas(journal, cp, df, gasto)
        print(f"[F] {n_ok}/{n_alvo} recuperados | custo US$ {gasto[0]:.4f}", flush=True)
        df, diag = carregar_de_checkpoint(journal, versao=versao_prompt)
        custo_total = float(df["custo_usd"].sum())

    dgn = diagnosticar(df)
    if recuperar:
        gasto = [0.0]
        instalar_captura()
        print(f"[F] Re-buscando justificativa dos {len(dgn['piores_casos'])} "
              "piores casos...", flush=True)
        df = recuperar_justificativas(journal, df, dgn["piores_casos"], gasto,
                                      Checkpoint(caminho_checkpoint(versao_prompt)))
        custo_total += gasto[0]
        dgn = diagnosticar(df)  # recomputa com as justificativas preenchidas
        print(f"[F] re-busca custou US$ {gasto[0]:.4f}", flush=True)

    df.to_csv(caminho_completo(versao_prompt), index=False)
    met = calcular_metricas(df)
    md = gerar_relatorio(df, diag, met, dgn, versao_prompt, custo_total, parcial)
    RELATORIO_MD.write_text(md, encoding="utf-8")
    print("\n" + "=" * 72)
    print(md)
    print("=" * 72)
    print(f"Relatório: {RELATORIO_MD}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    from agents.econ import _PROMPT_VERSION
    from agents.journal import JournalAgent

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    journal = JournalAgent()
    if not args.cascata_completa:  # R4 — Bloomberg-only, igual ao gate
        journal.gdelt.buscar = lambda *a, **k: []
        journal.newsapi.buscar = lambda *a, **k: []

    inicio = pd.Timestamp(args.data_inicio, tz=FUSO) if args.data_inicio else JANELA_INICIO
    fim = (pd.Timestamp(args.data_fim, tz=FUSO) + pd.Timedelta(hours=23, minutes=59, seconds=59)
           if args.data_fim else JANELA_FIM)

    print("=" * 72)
    print(f"BASELINE DA CALIBRAÇÃO DO ECON — prompt `{_PROMPT_VERSION}` (não modificado)")
    print(f"  modelo={MODELO} | janela={inicio.date()}..{fim.date()} | "
          f"dedup={not args.sem_dedup} | dry_run={args.dry_run}")
    print("=" * 72)

    if args.finalizar:
        return _finalizar(journal, _PROMPT_VERSION,
                          args.recuperar_justificativas, args.reavaliar_degradadas)

    print("\n[A] Amostrando eventos...", flush=True)
    eventos, diag = amostrar_eventos(journal, inicio, fim,
                                     dedup=not args.sem_dedup, n_max=args.n)
    print(f"[A] eventos={diag['n_eventos']} (limpos={diag['n_limpos']}) | "
          f"candidatos={diag['n_candidatos']} sem_noticia={diag['n_sem_noticia']} "
          f"duplicados={diag['n_duplicados_descartados']} sem_y={diag['n_sem_y']}", flush=True)

    if diag["n_eventos"] < N_MIN_EVENTOS:
        print(f"[A] ⚠️ N={diag['n_eventos']} < piso {N_MIN_EVENTOS}. Suspeito de bug "
              "de filtro (o gate teve 318 eventos em janela MAIS CURTA). ABORTANDO.")
        return 3

    # Persistido para que `--finalizar` reporte a amostragem real, e não "n/d".
    import json as _json
    AMOSTRAGEM_JSON.write_text(_json.dumps(diag, indent=2, ensure_ascii=False))

    estimativa = estimar_custo_usd(diag["n_eventos"], MODELO)
    print(f"\n[EST] Custo estimado: US$ {estimativa:.4f} "
          f"({diag['n_eventos']} eventos) | hard cap US$ {CUSTO_HARD_CAP_USD:.2f}")
    if estimativa > CUSTO_HARD_CAP_USD:
        print(f"[EST] ⛔ Estimativa ACIMA do hard cap. PARANDO antes de gastar. "
              f"Reduza a amostra (--n {int(CUSTO_HARD_CAP_USD / estimar_custo_usd(1, MODELO))}) "
              "ou eleve o cap conscientemente.")
        return 4
    if args.dry_run:
        print("[DRY-RUN] Nada foi gasto. Rode sem --dry-run para executar.", flush=True)
        return 0

    # Macro uma vez só: `get_macro` cacheia por DATA, e centenas de eventos viram
    # centenas de buscas da série inteira no BCB SGS → rate-limit por IP (foi o
    # que travou a primeira execução desta rodada por horas).
    try:
        prefetch_macro(journal, fim)
    except MacroIndisponivel as e:
        print(f"[MACRO] ⛔ {e}")
        return 5

    checkpoint = Checkpoint(caminho_checkpoint(_PROMPT_VERSION))
    alinhamento = conferir_alinhamento(eventos, checkpoint)
    if alinhamento["n_desalinhados"]:
        print(f"[ALINHAMENTO] ⛔ {alinhamento['n_desalinhados']} de "
              f"{alinhamento['n_checkpoint']} linhas do checkpoint apontam para "
              "outro ticker-dia — a amostragem mudou desde a rodada anterior "
              "(yfinance instável?). Retomar reavaliaria eventos já pagos. "
              "ABORTANDO. Use --finalizar para fechar com o que já existe.")
        return 7
    print(f"[ALINHAMENTO] ✓ {alinhamento['n_alinhados']} eventos do checkpoint "
          f"conferem; faltam {len(eventos) - alinhamento['n_checkpoint']} a avaliar",
          flush=True)
    instalar_captura()
    custo = [0.0]
    parcial = False
    print(f"\n[B] Avaliando {len(eventos)} eventos com {MODELO}...", flush=True)
    try:
        linhas = avaliar_eventos(journal, eventos, custo, checkpoint)
    except RuntimeError as e:
        print(f"\n[HARD CAP] {e}")
        return 2
    print(f"[B] {len(linhas)} avaliações | custo real US$ {custo[0]:.4f}")

    df = pd.DataFrame(linhas)
    if len(df) < len(eventos):
        parcial = True
    df.to_csv(caminho_completo(_PROMPT_VERSION), index=False)

    if args.sem_relatorio:
        print(f"\n[C] --sem-relatorio: CSV gravado em "
              f"{caminho_completo(_PROMPT_VERSION).name}; relatório é da Etapa 2.")
        print(f"Custo real total: US$ {custo[0]:.4f}")
        return 0

    print("\n[C] Métricas + diagnóstico...", flush=True)
    met = calcular_metricas(df)
    dgn = diagnosticar(df)
    md = gerar_relatorio(df, diag, met, dgn, _PROMPT_VERSION, custo[0], parcial)
    RELATORIO_MD.write_text(md, encoding="utf-8")

    print("\n" + "=" * 72)
    print(md)
    print("=" * 72)
    print(f"Custo real total: US$ {custo[0]:.4f} "
          f"({100 * custo[0] / CUSTO_HARD_CAP_USD:.1f}% do cap)")
    print(f"Relatório: {RELATORIO_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
