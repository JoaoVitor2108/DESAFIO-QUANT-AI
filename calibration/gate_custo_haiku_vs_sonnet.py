"""Gate de custo Haiku vs Sonnet — decisão única, sem iteração de prompt.

Responde UMA pergunta: "vale a pena pagar 3x mais pelo Sonnet 4.6 vs Haiku 4.5?".
NÃO é calibração completa: não itera prompt, não roda placebo, não roda auditoria
de justificativas. Reusa os helpers PUROS e testados de
`calibration/econ_calibration.py` (IC de Spearman, block bootstrap por data,
target beta-ajustado 5du sem lookahead, beta setorial, classificação de
degradação, segmentação por cutoff) e a infra de rodada paga de
`calibration/exec_infra.py` (captura de custo/latência, checkpoint, retry), e
adiciona só o que falta para o gate:

  - ΔIC pareado (mesmos blocos reamostrados para os dois modelos → covariância
    preservada);
  - captura de custo real (tokens × preço), latência ISOLADA do LLM (só
    `messages.create`, não `avaliar` inteiro) e taxa de fallback;
  - critério de decisão automatizado (zona cinza → Haiku por custo-benefício);
  - relatório em `calibration/results/RELATORIO_GATE_CUSTO.md`.

tz: o ECON opera NATIVAMENTE em tz-aware America/Sao_Paulo — NÃO há adaptador
`_para_sp` aqui (isso é do PROGRAM). Todos os `data_limite` são tz-aware SP.

Execução: `python calibration/gate_custo_haiku_vs_sonnet.py` (requer ANTHROPIC_API_KEY).
Os helpers puros abaixo são testados offline em `tests/test_gate_custo.py`.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import FUSO, tickers_ativos
from calibration.econ_calibration import (
    BLOCO_DIAS_UTEIS,
    CUTOFF_LIMPO,
    N_BOOTSTRAP,
    PRECOS,
    SEED_BOOTSTRAP,
    TRAINING_CUTOFF,
    _retorno_excesso_5d,
    adicionar_blocos,
    bootstrap_ic_bloco,
    calcular_ic,
    classificar_degradacao,
    segmentar_por_exposicao,
)
from calibration.exec_infra import (
    _CALLS,
    CHECKPOINT_A_CADA,
    MAX_RETRY_AVALIAR,
    Checkpoint as _Checkpoint,
    avaliar_com_retry as _avaliar_com_retry,
    calendario_pregoes as _calendario_pregoes,
    instalar_captura as _instalar_captura,
)

logger = logging.getLogger("gate_custo")

# ── Parâmetros travados do experimento ────────────────────────────────────────

MODELO_HAIKU = "claude-haiku-4-5-20251001"
MODELO_SONNET = "claude-sonnet-4-6"

N_ALVO = 100
N_MIN = 40  # piso: abaixo disso o IC95 do ΔIC fica largo demais p/ decidir → aborta
# Janela LIMPA do Haiku 4.5: estritamente PÓS training cutoff (2025-07-31). Jul/2025
# está no treino → contamina. Fronteira exata: data_noticia_mais_recente >= CUTOFF_LIMPO.
JANELA_INICIO = pd.Timestamp("2025-08-01", tz=FUSO)
JANELA_FIM = pd.Timestamp("2025-12-31 23:59:59", tz=FUSO)

SEED_AMOSTRA = 42

SLEEP_ENTRE_CHAMADAS_S = 1.0     # rate-limit safe (ambos os modelos são pagos)
CUSTO_HARD_CAP_USD = 8.0         # 2x a estimativa (~US$2); estourar = algo errado
MAX_PROBES = 5000                # teto de sondagens (>= candidatos p/ exaurir todos os eventos)
CREDITO_INICIAL_USD = 10.0       # crédito no console Anthropic (para o relatório de gasto)

# Critério de decisão (idêntico aos gaps de econ_calibration v4-C5).
GAP_ADOTAR_SONNET = 0.05
GAP_FICAR_HAIKU = 0.03

RESULTS_DIR = Path(__file__).parent / "results"
RELATORIO_MD = RESULTS_DIR / "RELATORIO_GATE_CUSTO.md"
COMPLETO_CSV = RESULTS_DIR / "gate_custo_resultado_completo.csv"
INTERMEDIARIO_CSV = RESULTS_DIR / "gate_custo_intermediario.csv"


# ── Funções PURAS (testadas offline) ──────────────────────────────────────────


def decidir_modelo(delta_ic: float) -> str:
    """Critério travado: ΔIC (Sonnet − Haiku) > 0.05 → 'sonnet'; caso contrário
    'haiku' (cobre ΔIC < 0.03 E a zona cinza 0.03 ≤ ΔIC ≤ 0.05, ambos → Haiku por
    custo-benefício)."""
    return "sonnet" if delta_ic > GAP_ADOTAR_SONNET else "haiku"


def bootstrap_delta_ic_pareado(
    df: pd.DataFrame, col_haiku: str, col_sonnet: str, col_y: str,
    n_iter: int = N_BOOTSTRAP, seed: int = SEED_BOOTSTRAP,
) -> dict:
    """ΔIC = IC_sonnet − IC_haiku com IC95 via block bootstrap PAREADO: a cada
    iteração reamostra os MESMOS blocos e calcula os dois ICs sobre as mesmas
    linhas, preservando a covariância entre os modelos. Requer coluna 'bloco'."""
    sh = df[col_haiku].to_numpy(dtype="float64")
    ss = df[col_sonnet].to_numpy(dtype="float64")
    y = df[col_y].to_numpy(dtype="float64")
    blocos = df["bloco"].to_numpy()
    unicos = np.unique(blocos)
    grupos = [np.where(blocos == b)[0] for b in unicos]
    n_b = len(unicos)
    delta_pt = calcular_ic(ss, y) - calcular_ic(sh, y)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_iter):
        pick = rng.integers(0, n_b, n_b)
        idx = np.concatenate([grupos[p] for p in pick])
        a = calcular_ic(sh[idx], y[idx])
        b = calcular_ic(ss[idx], y[idx])
        if a == a and b == b:
            deltas.append(b - a)
    deltas = np.array(deltas) if deltas else np.array([delta_pt])
    return {
        "delta": float(delta_pt),
        "ic95_low": float(np.percentile(deltas, 2.5)),
        "ic95_high": float(np.percentile(deltas, 97.5)),
    }


def eh_fallback(score) -> bool:
    """Taxa de fallback (métrica secundária): tem_evento=False, confianca<0.3, ou
    avisos != []."""
    return (not score.tem_evento) or (score.confianca < 0.3) or bool(score.avisos)


# ── Etapa A — amostragem de eventos limpos ────────────────────────────────────


def amostrar_eventos(
    journal, n_alvo: int = N_ALVO,
    janela_inicio: pd.Timestamp | None = None,
    janela_fim: pd.Timestamp | None = None,
    cutoff: pd.Timestamp | None = None,
) -> tuple[list[dict], dict]:
    """Amostra até `n_alvo` eventos LIMPOS na janela: ticker-dia com notícia E com
    target beta-ajustado 5du disponível. Determinístico (SEED_AMOSTRA).

    janela_inicio/janela_fim/cutoff default às constantes do módulo (ago-dez/2025).
    Retorna (eventos, diag). Cada evento: {ticker, data, noticias, y}. `diag` traz
    contagens de sondagem para o relatório. Se não alcançar `n_alvo`, PARA e reporta
    quantos conseguiu (via `diag`), sem abortar."""
    janela_inicio = janela_inicio if janela_inicio is not None else JANELA_INICIO
    janela_fim = janela_fim if janela_fim is not None else JANELA_FIM
    cutoff = cutoff if cutoff is not None else CUTOFF_LIMPO
    pregoes = _calendario_pregoes(janela_inicio, janela_fim)
    candidatos = []
    for data in pregoes:
        for ticker in tickers_ativos(data):
            candidatos.append((ticker, data))
    rng = np.random.default_rng(SEED_AMOSTRA)
    rng.shuffle(candidatos)

    eventos: list[dict] = []
    n_probes = n_sem_noticia = n_sem_y = n_noticia_pre_cutoff = 0
    for ticker, data in candidatos:
        if len(eventos) >= n_alvo or n_probes >= MAX_PROBES:
            break
        n_probes += 1
        try:
            noticias = journal.get_noticias(ticker, data)
        except Exception as e:  # rede/fonte instável → pula, não aborta (grac.)
            logger.warning("get_noticias falhou %s %s: %s", ticker, data.date(), e)
            continue
        if not noticias:
            n_sem_noticia += 1
            continue
        # Fronteira LIMPA exata: a notícia MAIS RECENTE do evento deve ser
        # >= 2025-08-01 00:00 SP (estritamente pós training cutoff do Haiku).
        data_noticia = max(n.publicado_em for n in noticias)
        if data_noticia < cutoff:
            n_noticia_pre_cutoff += 1
            continue
        y = _retorno_excesso_5d(journal, ticker, data, ajuste_beta=True)
        if y is None:
            n_sem_y += 1
            continue
        eventos.append({"ticker": ticker, "data": data, "noticias": noticias,
                        "y": y, "data_noticia": data_noticia})

    diag = {
        "n_eventos": len(eventos),
        "n_probes": n_probes,
        "n_candidatos": len(candidatos),
        "n_sem_noticia": n_sem_noticia,
        "n_noticia_pre_cutoff": n_noticia_pre_cutoff,
        "n_sem_y": n_sem_y,
        "alcancou_alvo": len(eventos) >= n_alvo,
        "cutoff_limpo": str(cutoff),
    }
    return eventos, diag


# ── Etapas B/C — avaliar cada evento com cada modelo ──────────────────────────


def _linha_de_checkpoint(ev: dict, modelo: str, prev: dict) -> dict:
    """Reconstrói a linha (para o relatório) de um evento já avaliado (retomada).

    Campos numéricos vêm do checkpoint; `fallback`/`degradacao` não são
    persistidos (colunas fixas do checkpoint) → assumidos neutros para linhas
    retomadas. Só afeta a métrica SECUNDÁRIA de fallback num cenário de crash."""
    return {
        "evento_id": ev["evento_id"], "ticker": ev["ticker"], "data": ev["data"],
        "y": ev["y"], "segmento": segmentar_por_exposicao(ev["data"]),
        "score": float(prev["score"]),
        "input_tokens": int(prev["tokens_in"]), "output_tokens": int(prev["tokens_out"]),
        "latencia_llm_s": float(prev["latencia_llm_s"]), "custo_usd": float(prev["custo_usd"]),
        "chamou_api": True, "fallback": False, "degradacao": None,
    }


def avaliar_modelo(journal, modelo: str, eventos: list[dict],
                   custo_acumulado: list[float], checkpoint: _Checkpoint) -> list[dict]:
    """Avalia todos os eventos com `modelo`, capturando score, tokens e latência
    ISOLADA do LLM. Retoma do checkpoint; retry com backoff (pula após
    MAX_RETRY_AVALIAR falhas, sem abortar); grava checkpoint e loga progresso a
    cada CHECKPOINT_A_CADA. Hard cap: RuntimeError (o caller reporta e para)."""
    from agents.econ import EconAgent

    agent = EconAgent(journal=journal, model=modelo)
    preco = PRECOS[modelo]
    linhas = []
    n_novos = 0
    for ev in eventos:
        eid = ev["evento_id"]
        if checkpoint.ja_feito(modelo, eid):
            prev = checkpoint.linha_feita(modelo, eid)
            linhas.append(_linha_de_checkpoint(ev, modelo, prev))
            custo_acumulado[0] += float(prev["custo_usd"])
            continue

        antes = len(_CALLS)
        try:
            score = _avaliar_com_retry(agent, ev)
        except Exception as e:  # esgotou retries → pula (grac.), NÃO aborta
            logger.warning("avaliar DESISTIU %s %s (%s) após %d tentativas: %s",
                           ev["ticker"], ev["data"].date(), modelo, MAX_RETRY_AVALIAR, e)
            continue

        rec = _CALLS[-1] if len(_CALLS) > antes else None
        it = (rec or {}).get("input_tokens") or 0
        ot = (rec or {}).get("output_tokens") or 0
        custo = it / 1e6 * preco["in"] + ot / 1e6 * preco["out"]
        custo_acumulado[0] += custo
        linhas.append({
            "evento_id": eid,
            "ticker": ev["ticker"],
            "data": ev["data"],
            "y": ev["y"],
            "segmento": segmentar_por_exposicao(ev["data"]),
            "score": score.score_total,
            "input_tokens": it,
            "output_tokens": ot,
            "latencia_llm_s": (rec or {}).get("latencia_s", float("nan")),
            "custo_usd": custo,
            "chamou_api": rec is not None,
            "fallback": eh_fallback(score),
            "degradacao": classificar_degradacao(score),
        })
        checkpoint.registrar({
            "evento_id": eid, "ticker": ev["ticker"], "data": ev["data"].isoformat(),
            "y_realizado": ev["y"], "modelo": modelo, "score": score.score_total,
            "latencia_llm_s": (rec or {}).get("latencia_s", float("nan")),
            "tokens_in": it, "tokens_out": ot, "custo_usd": custo,
        })
        n_novos += 1
        if n_novos % CHECKPOINT_A_CADA == 0:
            print(f"    [{modelo}] +{n_novos} avaliações | custo acum US$ {custo_acumulado[0]:.4f}")

        if custo_acumulado[0] > CUSTO_HARD_CAP_USD:
            checkpoint.flush()
            raise RuntimeError(
                f"HARD CAP de custo estourado: US$ {custo_acumulado[0]:.2f} > "
                f"US$ {CUSTO_HARD_CAP_USD:.2f}. Parando — checkpoint salvo, dá pra retomar."
            )
        time.sleep(SLEEP_ENTRE_CHAMADAS_S)

    checkpoint.flush()
    return linhas


# ── Etapa G — relatório ───────────────────────────────────────────────────────


def _resumo_modelo(linhas: list[dict], modelo: str) -> dict:
    df = pd.DataFrame(linhas)
    validas = df[df["chamou_api"]]
    lat = validas["latencia_llm_s"].dropna()
    return {
        "modelo": modelo,
        "n": len(df),
        "custo_total_usd": float(df["custo_usd"].sum()),
        "custo_medio_usd": float(df["custo_usd"].mean()) if len(df) else float("nan"),
        "lat_mediana_s": float(lat.median()) if len(lat) else float("nan"),
        "lat_p90_s": float(lat.quantile(0.90)) if len(lat) else float("nan"),
        "lat_p95_s": float(lat.quantile(0.95)) if len(lat) else float("nan"),
        "taxa_fallback": float(df["fallback"].mean()) if len(df) else float("nan"),
    }


def gerar_relatorio(df_merged: pd.DataFrame, diag: dict, res_h: dict, res_s: dict,
                    ic_h: dict, ic_s: dict, delta: dict, decisao: str,
                    parcial: bool, custo_total: float, n_alvo: int,
                    modo_todos: bool) -> str:
    tk = df_merged["ticker"].value_counts()
    meses = df_merged["data"].apply(lambda d: pd.Timestamp(d).strftime("%Y-%m")).value_counts().sort_index()
    concentrados = tk[tk > 5]

    def _ic(r):
        return f"{r['ic']:+.4f} [{r['ic95_low']:+.4f}, {r['ic95_high']:+.4f}]"

    linhas = []
    linhas.append("# Gate de custo — Haiku 4.5 vs Sonnet 4.6\n")
    if parcial:
        linhas.append(f"> ⚠️ **AMOSTRA PARCIAL** — não alcançou o alvo de {n_alvo} eventos "
                      f"limpos (conseguiu {diag['n_eventos']}). Ver diagnóstico de amostragem.\n")
    linhas.append(f"- Data/hora de execução: {pd.Timestamp.now(tz=FUSO).isoformat(timespec='seconds')}")
    linhas.append(f"- Fonte de notícias: {'cascata completa' if diag.get('cascata_completa') else 'Bloomberg-only'}\n")
    linhas.append("## Parâmetros do experimento\n")
    _n_alvo_txt = "TODOS os eventos limpos" if modo_todos else str(n_alvo)
    linhas.append(f"- N alvo: {_n_alvo_txt} | N mínimo: {N_MIN} | N efetivo: {len(df_merged)}")
    linhas.append(f"- Janela: {JANELA_INICIO.date()} → {JANELA_FIM.date()}")
    linhas.append(f"- Fronteira LIMPA (exata): `data_noticia_mais_recente >= "
                  f"{CUTOFF_LIMPO}` (estritamente pós training cutoff do Haiku 4.5, "
                  f"{TRAINING_CUTOFF.date()}). Jul/2025 EXCLUÍDO por estar no treino.")
    linhas.append(f"- Modelos: `{MODELO_HAIKU}` vs `{MODELO_SONNET}`")
    linhas.append(f"- Métrica: IC Spearman(score, retorno beta-ajustado 5du)")
    linhas.append(f"- Bootstrap: block-by-date, blocos de {BLOCO_DIAS_UTEIS} dias úteis, "
                  f"{N_BOOTSTRAP} iterações, seed={SEED_BOOTSTRAP}")
    linhas.append(f"- Seed da amostra: {SEED_AMOSTRA}\n")

    linhas.append("## Amostra\n")
    linhas.append(f"- Sondagens de notícia: {diag['n_probes']} (de {diag['n_candidatos']} "
                  f"candidatos); sem notícia: {diag['n_sem_noticia']}; "
                  f"notícia pré-cutoff (< {CUTOFF_LIMPO.date()}, descartadas): "
                  f"{diag.get('n_noticia_pre_cutoff', 0)}; sem target y: {diag['n_sem_y']}")
    linhas.append(f"- Segmento (cutoff Haiku {TRAINING_CUTOFF.date()}): "
                  + ", ".join(f"{k}={v}" for k, v in
                              df_merged['segmento'].value_counts().to_dict().items()))
    linhas.append("- Distribuição por mês: "
                  + ", ".join(f"{m}={c}" for m, c in meses.items()))
    linhas.append("- Distribuição por ticker: "
                  + ", ".join(f"{t}={c}" for t, c in tk.items()))
    if len(concentrados):
        linhas.append(f"- ⚠️ Tickers com > 5 eventos (concentração): "
                      + ", ".join(f"{t}={c}" for t, c in concentrados.items()))
    else:
        linhas.append("- Concentração: nenhum ticker com > 5 eventos.")
    linhas.append("")

    for res, ic, nome in ((res_h, ic_h, "Haiku 4.5"), (res_s, ic_s, "Sonnet 4.6")):
        linhas.append(f"## Resultado — {nome} (`{res['modelo']}`)\n")
        linhas.append(f"- IC Spearman: {_ic(ic)}  (n_blocos={ic['n_blocos']})")
        linhas.append(f"- Custo total: US$ {res['custo_total_usd']:.4f} | "
                      f"médio/chamada: US$ {res['custo_medio_usd']:.6f}")
        linhas.append(f"- Latência LLM: mediana {res['lat_mediana_s']:.2f}s | "
                      f"P90 {res['lat_p90_s']:.2f}s | P95 {res['lat_p95_s']:.2f}s")
        linhas.append(f"- Taxa de fallback: {res['taxa_fallback']:.1%}\n")

    linhas.append("## ΔIC (Sonnet − Haiku)\n")
    linhas.append(f"- ΔIC = {delta['delta']:+.4f} "
                  f"[{delta['ic95_low']:+.4f}, {delta['ic95_high']:+.4f}] "
                  "(bootstrap pareado, mesmos blocos)\n")

    linhas.append("## Decisão automatizada\n")
    linhas.append(f"- Critério: ΔIC > {GAP_ADOTAR_SONNET} → Sonnet; "
                  f"< {GAP_FICAR_HAIKU} → Haiku; zona cinza → Haiku (custo-benefício).")
    zona_cinza = GAP_FICAR_HAIKU <= delta["delta"] <= GAP_ADOTAR_SONNET
    if zona_cinza:
        linhas.append(f"- ⚠️ **ZONA CINZA** (0.03 ≤ ΔIC ≤ 0.05): modelos ~equivalentes "
                      "na tarefa; decisão Haiku por custo-benefício.")
    linhas.append(f"- **DECISÃO: {decisao.upper()}**\n")

    linhas.append("## Custo real gasto\n")
    pct_cap = 100.0 * custo_total / CUSTO_HARD_CAP_USD if CUSTO_HARD_CAP_USD else float("nan")
    linhas.append(f"- **Custo real total: US$ {custo_total:.4f}**")
    linhas.append(f"- Percentual do hard cap (US$ {CUSTO_HARD_CAP_USD:.0f}): {pct_cap:.1f}%")
    linhas.append(f"- Crédito restante estimado: US$ {CREDITO_INICIAL_USD - custo_total:.2f} "
                  f"(de US$ {CREDITO_INICIAL_USD:.0f} iniciais)\n")

    lat_max = max(res_h["lat_mediana_s"], res_s["lat_mediana_s"])
    if lat_max > 10:
        linhas.append("> ⚠️ **Latência LLM mediana > 10s** — um backtest oficial "
                      "(2000+ chamadas) levaria horas. Não bloqueia a decisão do gate, "
                      "mas é dado relevante para o planejamento do backtest.\n")

    linhas.append("## Limitações\n")
    linhas.append(f"- **Amostra restrita a ago-dez/2025** (janela limpa do Haiku 4.5, "
                  f"estritamente pós training cutoff {TRAINING_CUTOFF.date()}; fronteira "
                  f"exata `data_noticia_mais_recente >= {CUTOFF_LIMPO}`). N pequeno e "
                  "uma única janela → não generaliza para outros regimes.")
    linhas.append("- **Cutoff do Sonnet 4.6 pode DIFERIR do Haiku 4.5.** Se for mais "
                  "recente, ago-dez/2025 pode estar (parcialmente) no treino do Sonnet "
                  "→ risco de contaminação diferencial (Sonnet 'lembrar' de eventos que "
                  "o Haiku não viu). Não há como corrigir dentro do gate.")
    linhas.append("- **Bootstrap block-by-date** (blocos de 5 dias úteis), não i.i.d.: "
                  "retornos por data são correlacionados, então reamostrar linhas "
                  "subestima o IC95. O ΔIC deste gate usa bootstrap PAREADO (mesmos "
                  "blocos p/ os dois modelos). A divergência com "
                  "`econ_calibration.py::comparar_modelos()` (que usava i.i.d.) foi "
                  "reconciliada: o block bootstrap agora é canônico em "
                  "`econ_calibration.py` e o gate importa de lá.")
    linhas.append("- Sem iteração de prompt, sem placebo, sem auditoria de "
                  "justificativas (fora do escopo deste gate — é decisão única).")
    linhas.append("- Target usa Close AJUSTADO (via `_retorno_excesso_5d`), enquanto "
                  "o MATH&ML usa Close_raw; diferença de retorno esperada é pequena.")
    linhas.append("- Eventos se sobrepõem: `get_noticias` usa lookback de 7 dias, então "
                  "a mesma notícia é pontuada em ticker-dias vizinhos (não são "
                  "observações independentes; o block bootstrap por data mitiga).")

    linhas.append("\n## Anexo — dados brutos (amostra)\n")

    def _tabela(sub: pd.DataFrame) -> list[str]:
        out = ["| evento_id | ticker | data | y_realizado | score_haiku | score_sonnet |",
               "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            out.append(f"| {int(r['evento_id'])} | {r['ticker']} | "
                       f"{pd.Timestamp(r['data']).date()} | {r['y']:+.4f} | "
                       f"{r['score_haiku']:+.4f} | {r['score_sonnet']:+.4f} |")
        return out

    ordenado = df_merged.sort_values("evento_id")
    linhas.append("**Primeiros 10 eventos:**\n")
    linhas.extend(_tabela(ordenado.head(10)))
    linhas.append("\n**Últimos 10 eventos:**\n")
    linhas.extend(_tabela(ordenado.tail(10)))
    linhas.append(f"\nCSV completo: `{COMPLETO_CSV.relative_to(COMPLETO_CSV.parents[2])}`")
    return "\n".join(linhas) + "\n"


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Gate de custo Haiku 4.5 vs Sonnet 4.6")
    p.add_argument("--n", type=int, default=0,
                   help="teto de eventos avaliados (0 = TODOS os eventos limpos)")
    p.add_argument("--data-inicio", default=None, help="YYYY-MM-DD (default: 2025-08-01)")
    p.add_argument("--data-fim", default=None, help="YYYY-MM-DD (default: 2025-12-31)")
    p.add_argument("--cascata-completa", action="store_true",
                   help="usa GDELT+NewsAPI além do Bloomberg (default: Bloomberg-only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    from agents.journal import JournalAgent

    _instalar_captura()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    journal = JournalAgent()

    # Bloomberg-only por padrão: rápido, determinístico e é um gate Bloomberg. GDELT
    # tem throttle de 5s/probe e está IP-flaky → cascata completa custaria horas.
    if not args.cascata_completa:
        journal.gdelt.buscar = lambda *a, **k: []
        journal.newsapi.buscar = lambda *a, **k: []

    janela_inicio = pd.Timestamp(args.data_inicio, tz=FUSO) if args.data_inicio else JANELA_INICIO
    janela_fim = (pd.Timestamp(args.data_fim, tz=FUSO) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                  if args.data_fim else JANELA_FIM)
    modo_todos = args.n <= 0
    n_alvo = 10**9 if modo_todos else args.n

    print("=" * 72)
    print("GATE DE CUSTO — Haiku 4.5 vs Sonnet 4.6")
    print(f"  fonte={'cascata completa' if args.cascata_completa else 'Bloomberg-only'} | "
          f"N={'todos' if modo_todos else n_alvo} | "
          f"janela={janela_inicio.date()}..{janela_fim.date()}")
    print("=" * 72)

    # Etapa A
    print("\n[A] Amostrando eventos limpos...")
    eventos, diag = amostrar_eventos(journal, n_alvo, janela_inicio, janela_fim)
    diag["cascata_completa"] = args.cascata_completa
    if modo_todos:
        diag["alcancou_alvo"] = True   # modo 'todos': o alvo é tudo que existe
    print(f"[A] eventos={diag['n_eventos']} probes={diag['n_probes']} "
          f"sem_noticia={diag['n_sem_noticia']} sem_y={diag['n_sem_y']} "
          f"alcancou_alvo={diag['alcancou_alvo']}")
    if diag["n_eventos"] < N_MIN:
        print(f"[A] N={diag['n_eventos']} abaixo do piso {N_MIN} — IC95 do ΔIC seria "
              "muito largo pra decisão significativa. ABORTANDO. Reporte a João.")
        return 3
    if not diag["alcancou_alvo"]:
        print(f"[A] ⚠️ PARCIAL: {diag['n_eventos']} eventos (< alvo {n_alvo}). "
              "Relatório marcado como PARCIAL.")

    # evento_id estável (ordem determinística) — chave do checkpoint e do merge.
    eventos.sort(key=lambda e: (e["data"], e["ticker"]))
    for i, ev in enumerate(eventos):
        ev["evento_id"] = i

    checkpoint = _Checkpoint(INTERMEDIARIO_CSV)
    custo = [0.0]
    try:
        print(f"\n[B] Avaliando {len(eventos)} eventos com Haiku...")
        linhas_h = avaliar_modelo(journal, MODELO_HAIKU, eventos, custo, checkpoint)
        print(f"[B] Haiku: {len(linhas_h)} avaliações | custo acum US$ {custo[0]:.4f}")
        print(f"\n[C] Avaliando {len(eventos)} eventos com Sonnet...")
        linhas_s = avaliar_modelo(journal, MODELO_SONNET, eventos, custo, checkpoint)
        print(f"[C] Sonnet: {len(linhas_s)} avaliações | custo acum US$ {custo[0]:.4f}")
    except RuntimeError as e:
        print(f"\n[HARD CAP] {e}")
        return 2

    # Merge por evento_id — só eventos avaliados nos DOIS modelos entram no IC.
    def _pref(linhas, sufixo):
        d = pd.DataFrame(linhas).rename(columns={
            "score": f"score_{sufixo}",
            "input_tokens": f"tokens_in_{sufixo}",
            "output_tokens": f"tokens_out_{sufixo}",
            "latencia_llm_s": f"latencia_{sufixo}_s",
            "custo_usd": f"custo_{sufixo}_usd",
            "fallback": f"fallback_{sufixo}",
        })
        cols = ["evento_id", f"score_{sufixo}", f"tokens_in_{sufixo}",
                f"tokens_out_{sufixo}", f"latencia_{sufixo}_s",
                f"custo_{sufixo}_usd", f"fallback_{sufixo}"]
        if sufixo == "haiku":
            cols += ["ticker", "data", "y", "segmento"]
        return d[cols]

    dh = _pref(linhas_h, "haiku")
    ds = _pref(linhas_s, "sonnet")
    merged = pd.merge(dh, ds, on="evento_id", how="inner")
    merged = adicionar_blocos(merged, "data", data_inicio=janela_inicio)

    # Etapas D/E — IC por modelo (block bootstrap) + ΔIC pareado
    ic_h = bootstrap_ic_bloco(merged, "score_haiku", "y")
    ic_s = bootstrap_ic_bloco(merged, "score_sonnet", "y")
    delta = bootstrap_delta_ic_pareado(merged, "score_haiku", "score_sonnet", "y")

    # Etapa F — decisão
    decisao = decidir_modelo(delta["delta"])

    res_h = _resumo_modelo(linhas_h, MODELO_HAIKU)
    res_s = _resumo_modelo(linhas_s, MODELO_SONNET)

    # Persistência (só resultados agregados — NUNCA payloads/notícias brutas)
    merged.to_csv(COMPLETO_CSV, index=False)
    md = gerar_relatorio(merged, diag, res_h, res_s, ic_h, ic_s, delta, decisao,
                         parcial=not diag["alcancou_alvo"], custo_total=custo[0],
                         n_alvo=n_alvo, modo_todos=modo_todos)
    RELATORIO_MD.write_text(md, encoding="utf-8")

    print("\n" + "=" * 72)
    print(md)
    print("=" * 72)
    print(f"Custo real total: US$ {custo[0]:.4f} "
          f"({100 * custo[0] / CUSTO_HARD_CAP_USD:.1f}% do cap US$ {CUSTO_HARD_CAP_USD:.0f})")
    print(f"Relatório: {RELATORIO_MD}")
    print(f"CSV completo: {COMPLETO_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
