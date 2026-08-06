"""Infra de execução de rodadas PAGAS de calibração (compartilhada).

Extraída de `gate_custo_haiku_vs_sonnet.py` para ser reusada pela calibração do
ECON sem copiar código: as duas rodadas gastam dinheiro real e precisam das
mesmas garantias — captura de custo/latência a partir do `usage` do SDK,
checkpoint incremental que sobrevive a crash, retry com backoff e calendário de
pregões da B3.

Nada aqui decide nada sobre o experimento (amostra, métrica, critério de parada):
isso é responsabilidade de quem chama. Aqui só existe encanamento.

Os helpers são exercidos offline por `tests/test_gate_custo.py` (via reexport no
gate) — nenhum toca a API.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from config import FUSO
from calibration.econ_calibration import PRECOS

logger = logging.getLogger(__name__)

MAX_RETRY_AVALIAR = 3     # tentativas por evento antes de desistir dele
CHECKPOINT_A_CADA = 10    # grava checkpoint incremental a cada N avaliações


# ── Instrumentação do SDK (não toca agents/econ.py) ───────────────────────────

# Uma entrada por `messages.create`: {"latencia_s", "input_tokens", "output_tokens"}
_CALLS: list[dict] = []


def instalar_captura() -> None:
    """Envolve `Messages.create` para cronometrar SÓ a chamada ao LLM (não a
    coleta de notícias) e capturar `usage`. Idempotente."""
    import anthropic.resources.messages as _m

    if getattr(_m.Messages.create, "_captura_instalada", False):
        return
    _orig = _m.Messages.create

    def _wrap(self, *a, **k):
        t0 = time.perf_counter()
        resp = _orig(self, *a, **k)
        dt = time.perf_counter() - t0
        u = getattr(resp, "usage", None)
        _CALLS.append({
            "latencia_s": dt,
            "input_tokens": getattr(u, "input_tokens", None) if u else None,
            "output_tokens": getattr(u, "output_tokens", None) if u else None,
        })
        return resp

    _wrap._captura_instalada = True
    _m.Messages.create = _wrap


def custo_da_chamada(registro: dict | None, modelo: str) -> float:
    """Custo real em USD de uma chamada, a partir do `usage` capturado.

    `registro` None (nenhuma chamada nova — cache hit do EconAgent) → custo zero,
    que é o custo verdadeiro."""
    preco = PRECOS[modelo]
    entrada = (registro or {}).get("input_tokens") or 0
    saida = (registro or {}).get("output_tokens") or 0
    return entrada / 1e6 * preco["in"] + saida / 1e6 * preco["out"]


# ── Checkpoint incremental ────────────────────────────────────────────────────


class Checkpoint:
    """Checkpoint incremental (long format) para retomar sem re-avaliar.

    Grava a cada `a_cada` avaliações (modo append). Na retomada, pares
    (modelo, evento_id) já presentes são pulados. O EconAgent tem cache próprio
    (TTL 24h), mas o checkpoint evita reconstruir contexto e sobrevive a um crash
    no meio de uma rodada paga."""

    # `justificativa`/`confianca`/`tem_evento`/`degradacao` são persistidos porque
    # são exatamente o material do diagnóstico de prompt: sem eles, uma retomada
    # devolve números sem o raciocínio que explica os erros.
    COLS = ["evento_id", "ticker", "data", "y_realizado", "modelo", "score",
            "latencia_llm_s", "tokens_in", "tokens_out", "custo_usd",
            "confianca", "tem_evento", "degradacao", "justificativa"]

    def __init__(self, path: Path, a_cada: int = CHECKPOINT_A_CADA) -> None:
        self.path = Path(path)
        self.a_cada = a_cada
        self.feitos: dict[tuple[str, int], dict] = {}
        self._buffer: list[dict] = []
        if self.path.exists():
            prev = pd.read_csv(self.path)
            for _, r in prev.iterrows():
                self.feitos[(str(r["modelo"]), int(r["evento_id"]))] = r.to_dict()
            logger.info("Checkpoint: %d avaliações carregadas de %s (retomada)",
                        len(self.feitos), self.path.name)
            self._migrar_schema(prev)

    def _migrar_schema(self, prev: pd.DataFrame) -> None:
        """Reescreve o arquivo no schema atual se ele foi gravado num anterior.

        Sem isso, o append de linhas novas (mais colunas) contra um header antigo
        produziria um CSV desalinhado — perdendo a rodada inteira já paga."""
        if list(prev.columns) == self.COLS:
            return
        logger.warning("Checkpoint %s em schema antigo (%d colunas); migrando para %d",
                       self.path.name, len(prev.columns), len(self.COLS))
        prev.reindex(columns=self.COLS).to_csv(self.path, index=False)

    def ja_feito(self, modelo: str, evento_id: int) -> bool:
        return (modelo, int(evento_id)) in self.feitos

    def linha_feita(self, modelo: str, evento_id: int) -> dict:
        return self.feitos[(modelo, int(evento_id))]

    def registrar(self, row: dict) -> None:
        # `.get`: chamadores que não preenchem as colunas de diagnóstico (o gate
        # de custo, por exemplo) continuam válidos — gravam vazio, não quebram.
        self.feitos[(row["modelo"], int(row["evento_id"]))] = row
        self._buffer.append({k: row.get(k) for k in self.COLS})
        if len(self._buffer) >= self.a_cada:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        escrever_header = not self.path.exists()
        pd.DataFrame(self._buffer, columns=self.COLS).to_csv(
            self.path, mode="a", header=escrever_header, index=False)
        self._buffer = []


# ── Retry com backoff ─────────────────────────────────────────────────────────


def avaliar_com_retry(agent, ev: dict, max_tentativas: int = MAX_RETRY_AVALIAR,
                      degradou=None):
    """`agent.avaliar` com backoff exponencial (2s, 4s, ...). Levanta a última
    exceção se todas as tentativas falharem — o caller então pula o evento.

    `degradou(score) -> bool` é ESSENCIAL: o EconAgent degrada graciosamente e
    NUNCA levanta exceção, então um erro de API volta como score neutro (0.0) com
    um aviso. Sem esse predicado, o retry nunca dispara e a falha entra no dataset
    disfarçada de avaliação legítima — foi assim que 101 eventos viraram zeros
    silenciosos na primeira rodada do baseline.
    """
    for tentativa in range(1, max_tentativas + 1):
        try:
            score = agent.avaliar(ev["ticker"], ev["data"], noticias_override=ev["noticias"])
            if degradou is None or not degradou(score):
                return score
            motivo = "avaliação degradada (sem chamada real ao LLM)"
        except Exception as e:
            if tentativa >= max_tentativas:
                raise
            motivo = f"exceção ({e})"
        if tentativa >= max_tentativas:
            return score  # esgotou: devolve o degradado, o caller decide o que fazer
        espera = 2 ** tentativa
        logger.warning("avaliar %s %s tentativa %d/%d: %s; retry em %ds",
                       ev["ticker"], ev["data"].date(), tentativa, max_tentativas,
                       motivo, espera)
        time.sleep(espera)


# ── Pré-fetch de macro ────────────────────────────────────────────────────────


class MacroIndisponivel(RuntimeError):
    """O pré-fetch de macro não trouxe série utilizável — falhar alto, porque
    seguir sem macro mudaria silenciosamente o payload enviado ao LLM."""


def prefetch_macro(journal, data_fim: pd.Timestamp, serie_exigida: str = "selic_diaria"):
    """Troca `journal.get_macro` por um servidor de FATIAS em memória.

    Motivo: `get_macro` cacheia por DATA, então uma rodada de centenas de eventos
    dispara centenas de buscas da série inteira no BCB SGS e leva rate-limit por
    IP (mesma armadilha que o MATH&ML resolveu com `_prefetch`). Aqui buscamos uma
    única vez até `data_fim` e servimos cada evento com a série cortada na sua
    própria `data_limite`.

    ⚠️ ANTI-LOOKAHEAD: a fatia entregue é sempre `índice <= data_limite`, idêntica
    ao que uma chamada direta devolveria — o range maior é só I/O, nunca chega ao
    payload. Coberto por `tests/test_exec_infra.py`.

    Devolve a função original (para restaurar). Levanta `MacroIndisponivel` se a
    série exigida vier vazia: melhor abortar do que avaliar centenas de eventos
    com um contexto silenciosamente diferente do resto da rodada.
    """
    base = journal.get_macro(data_fim)
    serie = base.get(serie_exigida)
    if serie is None or serie.empty:
        raise MacroIndisponivel(
            f"pré-fetch de macro sem '{serie_exigida}' até {data_fim.date()} — "
            "BCB SGS/FRED indisponíveis. Abortando em vez de degradar a rodada."
        )

    original = journal.get_macro

    def _fatiar(data_limite: pd.Timestamp) -> dict:
        return {
            nome: s if s.empty else s[s.index <= data_limite]
            for nome, s in base.items()
        }

    journal.get_macro = _fatiar
    logger.info("Macro pré-buscada até %s (%d séries) — get_macro agora fatia em memória",
                data_fim.date(), len(base))
    return original


# ── Calendário ────────────────────────────────────────────────────────────────


def calendario_pregoes(inicio: pd.Timestamp, fim: pd.Timestamp) -> list[pd.Timestamp]:
    """Pregões B3 (BMF) na janela, tz-aware SP às 23:59 — captura o dia inteiro de
    notícia, com entrada no fechamento do dia."""
    import pandas_market_calendars as mcal

    bmf = mcal.get_calendar("BMF")
    sched = bmf.schedule(start_date=inicio.date(), end_date=fim.date())
    naive = sched.index.tz_localize(None)
    return [pd.Timestamp(d.date(), tz=FUSO) + pd.Timedelta(hours=23, minutes=59)
            for d in naive]
