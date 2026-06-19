"""
EconAgent — análise fundamentalista qualitativa via Claude (tool use).

Recebe dados brutos do JOURNAL (notícia + fundamentos CVM + retornos dos pares +
macro), aciona o Claude como analista fundamentalista de ações brasileiras
(buy-side) e devolve um `ScoreEcon`: score [-1,+1] de impacto esperado no retorno
em excesso ao Ibovespa nos próximos 5 dias úteis, com componentes desagregados
para o MATH&ML.

Princípios herdados do JOURNAL:
- Anti-lookahead: o JOURNAL já corta todos os dados em `data_limite` (tz-aware SP);
  o ECON apenas repassa o texto ex-ante ao LLM e o instrui a raciocinar como se
  fosse essa data.
- Degradação graciosa: sem chave / erro de API → score neutro + aviso, nunca
  levanta exceção (o backtest não pode quebrar por uma falha de fonte).
- Event-driven: sem notícia relevante → score neutro SEM chamar o Claude (custo).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from config import FUSO, TICKER_PARA_NOME, UNIVERSO_HISTORICO
from agents.journal import JournalAgent, _DiskCache, _validate_aware
from agents.sources.noticia import Noticia

logger = logging.getLogger(__name__)

MODELO_PADRAO = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024

# Versão do prompt/schema. Entra na chave de cache: ao iterar o prompt na
# calibração (≤10 iterações), avaliações antigas NÃO podem ser reusadas, senão
# compararíamos IC de um prompt novo com saídas cacheadas do prompt antigo.
# REGRA: faça bump sempre que `_SYSTEM_PROMPT` ou `_TOOL`/schema mudarem.
_PROMPT_VERSION = "2024-06-econA"

# Teto de notícias enviadas ao LLM. Justificativa: a janela operacional é de 7
# dias por ticker; após o filtro de whitelist do JOURNAL sobram poucas notícias
# relevantes, e o JOURNAL já as ordena por peso de fonte e recência — 8 cobre os
# eventos materiais sem inflar contexto/custo.
_MAX_NOTICIAS = 8
# Truncamento do corpo de cada notícia. Justificativa: a estrutura jornalística
# põe o fato no lead + 1º parágrafo; ~800 chars capturam isso. O GDELT muitas
# vezes nem traz corpo. (TODO calibração: medir sensibilidade do score a este valor.)
_MAX_CONTEUDO_CHARS = 800

# Limiar da checagem de sanidade (Opção A): sob a semântica nova, `score_total`
# deve ficar próximo de `comp_noticia` (sua base). Divergência maior gera aviso.
_DIVERGENCIA_MAX = 0.5


# ── Contrato de saída ─────────────────────────────────────────────────────────


@dataclass
class ScoreEcon:
    ticker: str
    data_referencia: pd.Timestamp        # = data_limite (tz-aware SP)
    score_total: float                   # [-1, +1] impacto da NOTÍCIA (Opção A) — é o sinal principal
    comp_noticia: float                  # [-1, +1] base de score_total (devem ficar próximos)
    comp_saude_financeira: float         # [-1, +1] CONTEXTO considerado, não parcela somada
    comp_setorial: float                 # [-1, +1] CONTEXTO considerado, não parcela somada
    comp_macro: float                    # [-1, +1] CONTEXTO considerado, não parcela somada
    confianca: float                     # [0, 1]
    tem_evento: bool                     # houve notícia relevante?
    n_noticias: int
    justificativa: str                   # raciocínio curto
    modelo: str                          # rastreabilidade
    avisos: list[str] = field(default_factory=list)


# ── Schema da ferramenta (tool use forçado, sem parsing de texto livre) ───────


_TOOL = {
    "name": "registrar_avaliacao",
    "description": (
        "Registra a avaliação econômica estruturada da ação. O 'score_total' é a "
        "nota principal: o impacto esperado da(s) NOTÍCIA(S) no retorno EM EXCESSO "
        "ao Ibovespa nos próximos 5 dias úteis (-1 muito negativo, 0 neutro, +1 "
        "muito positivo). Os componentes de saúde financeira, setorial e macro são "
        "o CONTEXTO que você considerou para calibrar essa leitura — NÃO são "
        "parcelas somadas ao score_total."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "score_total": {"type": "number", "minimum": -1, "maximum": 1,
                            "description": "Nota principal: impacto da(s) notícia(s) no "
                                           "retorno em excesso ao Ibovespa em 5 dias úteis. "
                                           "Deve refletir essencialmente componente_noticia."},
            "componente_noticia": {"type": "number", "minimum": -1, "maximum": 1,
                                   "description": "Impacto isolado da(s) notícia(s) — base do score_total."},
            "componente_saude_financeira": {"type": "number", "minimum": -1, "maximum": 1,
                                            "description": "CONTEXTO: qualidade dos fundamentos (TTM). Não somar ao total."},
            "componente_setorial": {"type": "number", "minimum": -1, "maximum": 1,
                                    "description": "CONTEXTO: momento do setor. Não somar ao total."},
            "componente_macro": {"type": "number", "minimum": -1, "maximum": 1,
                                 "description": "CONTEXTO: vento macro (Selic, IPCA, câmbio). Não somar ao total."},
            "confianca": {"type": "number", "minimum": 0, "maximum": 1,
                          "description": "Confiança na avaliação em [0, 1]. Notícias "
                                         "contraditórias entre si devem REDUZIR este valor."},
            "justificativa": {"type": "string",
                              "description": "Raciocínio econômico em 1-3 frases."},
        },
        "required": [
            "score_total", "componente_noticia", "componente_saude_financeira",
            "componente_setorial", "componente_macro", "confianca", "justificativa",
        ],
    },
}

_SYSTEM_PROMPT = (
    "Você é um analista fundamentalista sênior de ações brasileiras (buy-side). "
    "Avalie o MECANISMO econômico da notícia — efeito esperado em caixa, margem, "
    "posição competitiva ou múltiplo — e NÃO o tom superficial do texto. "
    "O 'score_total' é sua nota PRINCIPAL e mede SÓ o impacto da(s) notícia(s) no "
    "retorno EM EXCESSO ao Ibovespa nos próximos 5 dias úteis (-1 muito negativo, "
    "0 neutro, +1 muito positivo); ele deve refletir essencialmente o "
    "'componente_noticia'. "
    "Saúde financeira (fundamentos TTM), momento setorial e cenário macro são o "
    "CONTEXTO que você usa para calibrar a leitura da notícia (ex.: a mesma notícia "
    "pesa mais numa empresa frágil) — você os reporta nos campos próprios, mas eles "
    "NÃO são parcelas somadas ao score_total. "
    "Desconte ruído sem efeito fundamental (ex.: política genérica). "
    "MÚLTIPLAS NOTÍCIAS: pondere pelo impacto fundamental e pela confiabilidade da "
    "fonte; notícias contraditórias entre si devem REDUZIR a 'confianca'. "
    "ANTI-LOOKAHEAD: raciocine APENAS com os dados fornecidos, como se a data de "
    "hoje fosse a data_limite informada; jamais use conhecimento de fatos "
    "posteriores a essa data. Responda EXCLUSIVAMENTE chamando a ferramenta "
    "registrar_avaliacao."
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _clamp(x, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return float(max(lo, min(hi, v)))


def _hash_noticias(noticias: list[Noticia]) -> str:
    """Hash determinístico do conjunto de notícias (para a chave de cache)."""
    chaves = sorted(f"{n.fonte}|{n.publicado_em.isoformat()}|{n.titulo}" for n in noticias)
    raw = "||".join(chaves)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _neutro(ticker, data_limite, n_noticias, justificativa, avisos, modelo) -> ScoreEcon:
    return ScoreEcon(
        ticker=ticker,
        data_referencia=data_limite,
        score_total=0.0,
        comp_noticia=0.0,
        comp_saude_financeira=0.0,
        comp_setorial=0.0,
        comp_macro=0.0,
        confianca=0.0,
        tem_evento=n_noticias > 0,
        n_noticias=n_noticias,
        justificativa=justificativa,
        modelo=modelo,
        avisos=list(avisos),
    )


# ── EconAgent ─────────────────────────────────────────────────────────────────


class EconAgent:
    def __init__(
        self,
        journal: Optional[JournalAgent] = None,
        client=None,
        model: str = MODELO_PADRAO,
        cache_dir: Path = Path("data/cache"),
    ) -> None:
        self.journal = journal if journal is not None else JournalAgent(cache_dir=cache_dir)
        self.model = model
        self._client = client          # injeção de dependência (testes); None → lazy
        self._client_resolvido = client is not None
        self._cache = _DiskCache(cache_dir)

    # ── Cliente Anthropic (lazy) ──────────────────────────────────────────────

    def _get_client(self):
        """Devolve o cliente Anthropic ou None se a chave/lib não estiverem disponíveis."""
        if self._client is not None:
            return self._client
        if self._client_resolvido:
            return None
        self._client_resolvido = True
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY ausente; ECON degrada para neutro.")
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except Exception as e:
            logger.warning("Falha ao inicializar cliente Anthropic: %s", e)
            self._client = None
        return self._client

    # ── API pública ───────────────────────────────────────────────────────────

    def avaliar(
        self,
        ticker: str,
        data_limite: pd.Timestamp,
        lookback_days: int = 7,
        noticias_override: Optional[list[Noticia]] = None,
        nome_override: Optional[str] = None,
    ) -> ScoreEcon:
        """Avalia o impacto econômico das notícias de `ticker` até `data_limite`.

        `noticias_override` e `nome_override` são hooks de calibração (default
        None = comportamento normal): permitem ao teste de placebo fornecer
        notícias já anonimizadas e substituir o nome da empresa, reusando todo o
        pipeline. Fundamentos/macro continuam vindo do ticker real (preserva o
        mecanismo econômico; esconde a identidade da empresa).
        """
        _validate_aware(data_limite, "data_limite")

        # 1. Notícias (já anti-lookahead no JOURNAL) — ou as fornecidas (placebo)
        if noticias_override is not None:
            noticias = noticias_override
        else:
            noticias = self.journal.get_noticias(ticker, data_limite, lookback_days)

        # 2. Sem evento → neutro SEM chamar o Claude
        if not noticias:
            return _neutro(ticker, data_limite, 0, "sem notícia relevante", [], self.model)

        noticias = noticias[:_MAX_NOTICIAS]

        # 3. Cache por (ticker, data, modelo, versão do prompt, conjunto de notícias).
        # nome_override entra na chave: placebo e avaliação real não compartilham cache.
        cache_key = {
            "t": ticker,
            "dl": str(data_limite.date()),
            "m": self.model,
            "pv": _PROMPT_VERSION,
            "nome": nome_override or "",
            "h": _hash_noticias(noticias),
        }
        cached = self._cache.get("econ_avaliar", cache_key)
        if cached is not None:
            return cached

        # 4. Cliente disponível?
        client = self._get_client()
        if client is None:
            return _neutro(
                ticker, data_limite, len(noticias),
                "avaliação indisponível (sem chave da API)",
                ["ANTHROPIC_API_KEY ausente ou cliente indisponível; score neutro."],
                self.model,
            )

        # 5. Coletar contexto fundamental/macro/setorial (tolerante a falha)
        avisos: list[str] = []
        contexto = self._montar_contexto(ticker, data_limite, noticias, avisos,
                                         nome_override=nome_override)

        # 6. Chamar o Claude (tool use forçado) e parsear
        try:
            resposta = client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                temperature=0,
                system=_SYSTEM_PROMPT,
                tools=[_TOOL],
                tool_choice={"type": "tool", "name": _TOOL["name"]},
                messages=[{"role": "user", "content": contexto}],
            )
        except Exception as e:
            logger.warning("Chamada ao Claude falhou para %s: %s", ticker, e)
            return _neutro(
                ticker, data_limite, len(noticias),
                "avaliação indisponível (erro de API)",
                avisos + [f"erro na chamada ao Claude: {e}"],
                self.model,
            )

        score = self._parsear(ticker, data_limite, noticias, resposta, avisos)
        if score is None:
            # Resposta malformada (sem tool_use): degrada sem cachear, para não
            # contaminar 24h de chamadas com uma falha transitória do modelo.
            return _neutro(
                ticker, data_limite, len(noticias),
                "avaliação indisponível (resposta sem tool_use)",
                avisos + ["resposta do Claude não contém bloco tool_use válido."],
                self.model,
            )
        self._cache.set("econ_avaliar", cache_key, score)
        return score

    # ── Internos ──────────────────────────────────────────────────────────────

    def _montar_contexto(self, ticker, data_limite, noticias, avisos: list[str],
                         nome_override: Optional[str] = None) -> str:
        """Monta o payload textual (JSON) enviado ao LLM a partir do JOURNAL.

        `nome_override` (placebo): substitui o nome da empresa apresentado ao LLM,
        sem mudar a fonte dos fundamentos/macro (que vêm do ticker real).
        """
        empresa = nome_override or TICKER_PARA_NOME.get(ticker, ticker)
        setor = UNIVERSO_HISTORICO.get(ticker, {}).get("setor")
        if setor is None:
            try:
                setor = self.journal.get_setor(ticker)
            except Exception as e:
                avisos.append(f"setor indisponível: {e}")

        payload = {
            "data_limite": str(data_limite),
            "ticker": ticker,
            "empresa": empresa,
            "setor": setor,
            "noticias": [
                {
                    "titulo": n.titulo,
                    "fonte": n.fonte,
                    "peso_fonte": n.peso_fonte,
                    "publicado_em": str(n.publicado_em),
                    "conteudo": (n.conteudo or "")[:_MAX_CONTEUDO_CHARS],
                }
                for n in noticias
            ],
        }

        # Fundamentos (tolerante a falha)
        try:
            f = self.journal.get_fundamentals(ticker, data_limite)
            payload["fundamentos"] = {
                "pl": f.pl, "pvp": f.pvp, "roe": f.roe,
                "margem_liquida": f.margem_liquida,
                "divida_liquida_ebitda": f.divida_liquida_ebitda,
                "receita": f.receita, "lucro_liquido": f.lucro_liquido,
                "periodicidade": f.periodicidade,
            }
            if setor is None:
                payload["setor"] = f.setor
        except Exception as e:
            avisos.append(f"fundamentos indisponíveis: {e}")

        # Macro (últimos valores ≤ data_limite)
        try:
            macro = self.journal.get_macro(data_limite)
            payload["macro"] = {
                nome: float(serie.iloc[-1])
                for nome, serie in macro.items()
                if serie is not None and not serie.empty and pd.notna(serie.iloc[-1])
            }
        except Exception as e:
            avisos.append(f"macro indisponível: {e}")

        # Retornos do setor
        try:
            if setor:
                payload["retornos_setor"] = self.journal.get_retornos_setor(setor, data_limite)
        except Exception as e:
            avisos.append(f"retornos do setor indisponíveis: {e}")

        return (
            "Avalie a ação com base nos dados abaixo (todos ex-ante à data_limite). "
            "Responda chamando a ferramenta registrar_avaliacao.\n\n"
            + json.dumps(payload, ensure_ascii=False, default=str, indent=2)
        )

    def _parsear(self, ticker, data_limite, noticias, resposta, avisos: list[str]) -> Optional[ScoreEcon]:
        """Extrai o bloco tool_use e monta o ScoreEcon; None se a resposta for inválida.

        Retornar None (em vez de um neutro pronto) permite ao chamador distinguir
        sucesso de falha e NÃO cachear a falha (ver BUG-1 / cache-poisoning).
        """
        dados = None
        for bloco in getattr(resposta, "content", []) or []:
            if getattr(bloco, "type", None) == "tool_use" and getattr(bloco, "name", None) == _TOOL["name"]:
                dados = getattr(bloco, "input", None)
                break

        if not isinstance(dados, dict):
            return None

        score_total = _clamp(dados.get("score_total"), -1, 1)
        comp_noticia = _clamp(dados.get("componente_noticia"), -1, 1)

        # P7 — sanidade leve (Opção A): score_total deve refletir comp_noticia.
        # Divergência grande não é fatal (degradação graciosa), mas é sinalizada.
        if abs(score_total - comp_noticia) > _DIVERGENCIA_MAX:
            msg = (f"divergência score_total ({score_total:+.2f}) vs "
                   f"componente_noticia ({comp_noticia:+.2f}) > {_DIVERGENCIA_MAX}")
            logger.warning("%s: %s", ticker, msg)
            avisos = avisos + [msg]

        return ScoreEcon(
            ticker=ticker,
            data_referencia=data_limite,
            score_total=score_total,
            comp_noticia=comp_noticia,
            comp_saude_financeira=_clamp(dados.get("componente_saude_financeira"), -1, 1),
            comp_setorial=_clamp(dados.get("componente_setorial"), -1, 1),
            comp_macro=_clamp(dados.get("componente_macro"), -1, 1),
            confianca=_clamp(dados.get("confianca"), 0, 1),
            tem_evento=True,
            n_noticias=len(noticias),
            justificativa=str(dados.get("justificativa", "")).strip(),
            modelo=self.model,
            avisos=avisos,
        )
