"""
Tipos compartilhados pelas fontes de notícia do JOURNAL.

Isolado em módulo próprio para evitar import circular: tanto journal.py
quanto as fontes especializadas (gdelt.py, newsapi.py) importam daqui, sem
que as fontes precisem importar journal.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Noticia:
    titulo: str
    conteudo: str
    url: str
    publicado_em: pd.Timestamp  # timezone-aware, America/Sao_Paulo
    fonte: str                  # domínio da fonte, ex: "bloomberg.com"
    peso_fonte: float           # peso da whitelist (1.0 = máxima confiabilidade)
    ticker: Optional[str] = None  # query/ticker que originou a busca


def peso_para_url(alvo: str, whitelist: dict[str, float]) -> Optional[float]:
    """Retorna o peso da whitelist se algum domínio casar com `alvo`.

    `alvo` pode ser a URL completa ou o domínio do artigo (ou ambos
    concatenados). Casamento por substring — cobre www., subdomínios e
    variações de caminho. Retorna None se fora da whitelist.
    """
    for dominio, peso in whitelist.items():
        if dominio in alvo:
            return peso
    return None
