from __future__ import annotations

import pandas as pd

FUSO = "America/Sao_Paulo"
LAG_FUNDAMENTALS_DIAS = 45
HORARIO_CORTE_B3 = "17:05"
INICIO_WARMUP = pd.Timestamp("2019-01-01", tz=FUSO)
FIM_BACKTEST = pd.Timestamp("2024-12-31 23:59:59", tz=FUSO)


def _ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str, tz=FUSO)


# Universo histórico: composição do IBOV ao longo de 2019-2024.
# entrada=None  → ticker já estava no índice em 01/01/2019.
# saida=None    → ticker ainda estava no índice em 31/12/2024.
# Todos os timestamps são timezone-aware em America/Sao_Paulo.
UNIVERSO_HISTORICO: dict[str, dict] = {
    # ── petroleo_gas ─────────────────────────────────────────────────────────
    "PETR4.SA": {
        "setor": "petroleo_gas",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    "PRIO3.SA": {
        "setor": "petroleo_gas",
        "entrada": _ts("2020-09-08"),
        "saida": None,
        "confianca": "alta",
        "fonte": "PetroRio ingressou no IBOV em 08/09/2020 (Money Times)",
    },
    # ── mineracao ─────────────────────────────────────────────────────────────
    "VALE3.SA": {
        "setor": "mineracao",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    "CMIN3.SA": {
        "setor": "mineracao",
        "entrada": _ts("2021-02-18"),
        "saida": None,
        "confianca": "alta",
        "fonte": "IPO CSN Mineração 18/02/2021",
    },
    # ── industria ─────────────────────────────────────────────────────────────
    "WEGE3.SA": {
        "setor": "industria",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "WEG no IBOV desde 2016",
    },
    "GGBR4.SA": {
        "setor": "industria",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019 (siderurgia/metalurgia)",
    },
    # ── bancos ────────────────────────────────────────────────────────────────
    "ITUB4.SA": {
        "setor": "bancos",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    "BBDC4.SA": {
        "setor": "bancos",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    "BBAS3.SA": {
        "setor": "bancos",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    "BPAC11.SA": {
        "setor": "bancos",
        "entrada": _ts("2019-10-02"),
        "saida": None,
        "confianca": "alta",
        "fonte": "BTG Pactual entrou no IBOV em 02/10/2019",
    },
    # ── alimentos_bebidas ─────────────────────────────────────────────────────
    "ABEV3.SA": {
        "setor": "alimentos_bebidas",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    "JBSS3.SA": {
        "setor": "alimentos_bebidas",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    # ── energia_eletrica ──────────────────────────────────────────────────────
    "ELET3.SA": {
        "setor": "energia_eletrica",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019; privatizada jun/2022, ticker mantido",
    },
    "EGIE3.SA": {
        "setor": "energia_eletrica",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    # ── papel_celulose ────────────────────────────────────────────────────────
    "SUZB3.SA": {
        "setor": "papel_celulose",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019; fusão Suzano+Fibria concluída jan/2019",
    },
    "KLBN11.SA": {
        "setor": "papel_celulose",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    # ── telecom ───────────────────────────────────────────────────────────────
    "VIVT3.SA": {
        "setor": "telecom",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    # ── saude ─────────────────────────────────────────────────────────────────
    "RDOR3.SA": {
        "setor": "saude",
        "entrada": _ts("2021-09-06"),
        "saida": None,
        "confianca": "alta",
        "fonte": "Inclusão no IBOV 06/09/2021; IPO 10/12/2020",
    },
    # ── varejo ────────────────────────────────────────────────────────────────
    "LREN3.SA": {
        "setor": "varejo",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    "MGLU3.SA": {
        "setor": "varejo",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    # Survivorship bias — padrão "colapso abrupto": Americanas entrou em RJ em
    # jan/2023. O modelo treina com AMER3 durante a janela ativa e não projeta
    # além de saida. AMER3 (B2W renomeada) começou a negociar em 19/07/2021;
    # LAME4 foi incorporada e deslistada em 24/01/2022.
    "AMER3.SA": {
        "setor": "varejo",
        "entrada": _ts("2021-07-19"),
        "saida": _ts("2023-01-12"),
        "confianca": "alta",
        "fonte": "AMER3 início 19/07/2021; recuperação judicial 12/01/2023",
    },
    # ── construcao ────────────────────────────────────────────────────────────
    "CYRE3.SA": {
        "setor": "construcao",
        "entrada": None,
        "saida": None,
        "confianca": "alta",
        "fonte": "B3 composição jan/2019",
    },
    # ── tecnologia ────────────────────────────────────────────────────────────
    "TOTS3.SA": {
        "setor": "tecnologia",
        "entrada": _ts("2020-01-06"),
        "saida": None,
        "confianca": "alta",
        "fonte": "Totvs entrou no IBOV em 06/01/2020",
    },
    # ── outros ────────────────────────────────────────────────────────────────
    # Survivorship bias — padrão "lenta agonia": IRB revelou fraude contábil em
    # fev/2020 e caiu ~80%, mas só foi excluído do IBOV no rebalanceamento de
    # jan/2023. Demonstra que ticker permanece no universo até a saída formal,
    # mesmo durante crise prolongada — não julgamento qualitativo.
    "IRBR3.SA": {
        "setor": "outros",
        "entrada": None,
        "saida": _ts("2023-01-02"),
        "confianca": "media",
        "fonte": "Excluído IBOV jan/2023; escândalo contábil fev/2020",
    },
}


def tickers_ativos(data: pd.Timestamp) -> list[str]:
    """Retorna tickers cuja janela [entrada, saida] contém `data`.

    Levanta ValueError se `data` for naive (sem timezone).
    """
    if data.tzinfo is None:
        raise ValueError(
            f"tickers_ativos requer timestamp timezone-aware; recebeu naive: {data}"
        )
    result = []
    for ticker, info in UNIVERSO_HISTORICO.items():
        entrada = info["entrada"] if info["entrada"] is not None else INICIO_WARMUP
        saida = info["saida"] if info["saida"] is not None else FIM_BACKTEST
        if entrada <= data <= saida:
            result.append(ticker)
    return result


# Mapa setor → [tickers] derivado do universo completo (não filtrado por data).
# Para iteração time-aware, usar tickers_ativos(data) filtrado por setor.
MAPA_SETORES: dict[str, list[str]] = {}
for _ticker, _info in UNIVERSO_HISTORICO.items():
    MAPA_SETORES.setdefault(_info["setor"], []).append(_ticker)
