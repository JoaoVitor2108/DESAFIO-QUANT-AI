"""
CVMSource — leitor de dados abertos da CVM (Comissão de Valores Mobiliários).

Fonte primária de fundamentals para o sistema JEMPO. Substitui yfinance
(cobertura limitada a ~4 trimestres) por dados históricos oficiais desde 2011.

Chave de precisão: DT_RECEB (data de recebimento na CVM) como marco de
disponibilidade pública — mais rigoroso que o lag heurístico de 45 dias.

Armadilhas evitadas:
  - encoding="latin-1" (NÃO utf-8 — CVM usa ISO-8859-1)
  - sep=";" e decimal="," (padrão brasileiro)
  - Versão máxima por (CD_CVM, DT_REFER, CD_CONTA, ORDEM_EXERC) — evita duplicatas
  - Multiplicar VL_CONTA × 1000 (valores em milhares de reais)
  - Timestamps do CSV são naive; localizados para America/Sao_Paulo ao retornar
  - DFP cobre Q4 (exercício anual); ITR cobre Q1–Q3
"""
from __future__ import annotations

import logging
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_URL_ITR = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/"
_URL_DFP = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/"
_URL_CAD = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"

_ENC = "latin-1"
_SEP = ";"
_DEC = ","
_FUSO = "America/Sao_Paulo"

# Colunas preservadas ao parsear os CSVs (descarta metadados desnecessários)
_KEEP = [
    "CD_CVM", "CNPJ_CIA", "DT_REFER", "DT_RECEB", "VERSAO",
    "ORDEM_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA",
]

# Templates de nome de arquivo dentro dos ZIPs
# HDR = arquivo índice que contém DT_RECEB por (CD_CVM, DT_REFER, VERSAO)
_ITR_TEMPLATES = {
    "HDR": "itr_cia_aberta_{ano}.csv",              # índice: DT_RECEB por filing
    "BPA": "itr_cia_aberta_BPA_con_{ano}.csv",
    "BPP": "itr_cia_aberta_BPP_con_{ano}.csv",
    "DRE": "itr_cia_aberta_DRE_con_{ano}.csv",
    "DFC": "itr_cia_aberta_DFC_MI_con_{ano}.csv",
    "CAP": "itr_cia_aberta_composicao_capital_{ano}.csv",
}
_DFP_TEMPLATES = {
    "HDR": "dfp_cia_aberta_{ano}.csv",
    "BPA": "dfp_cia_aberta_BPA_con_{ano}.csv",
    "BPP": "dfp_cia_aberta_BPP_con_{ano}.csv",
    "DRE": "dfp_cia_aberta_DRE_con_{ano}.csv",
    "DFC": "dfp_cia_aberta_DFC_MI_con_{ano}.csv",
    "CAP": "dfp_cia_aberta_composicao_capital_{ano}.csv",
}


def _localize(ts) -> Optional[pd.Timestamp]:
    """Converte timestamp naive ou aware para America/Sao_Paulo."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    t = pd.Timestamp(ts)
    if pd.isna(t):
        return None
    return t.tz_localize(_FUSO) if t.tzinfo is None else t.tz_convert(_FUSO)


class CVMSource:
    """Leitor de dados abertos da CVM com cache local em parquet."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cvm",
        ttl_dias: int = 90,
    ) -> None:
        self.raw_dir = Path(cache_dir) / "raw"
        self.processed_dir = Path(cache_dir) / "processed"
        self.ttl = timedelta(days=ttl_dias)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._cadastro: Optional[pd.DataFrame] = None

    # ── Download ──────────────────────────────────────────────────────────────

    def baixar_ano(self, ano: int, tipo: str = "ITR") -> Path:
        """Baixa o ZIP do ano (ITR ou DFP) e salva em data/cvm/raw/."""
        if tipo not in ("ITR", "DFP"):
            raise ValueError(f"tipo deve ser 'ITR' ou 'DFP'; recebeu {tipo!r}")
        filename = f"{tipo.lower()}_cia_aberta_{ano}.zip"
        url = (_URL_ITR if tipo == "ITR" else _URL_DFP) + filename
        destino = self.raw_dir / filename

        if destino.exists():
            age = datetime.now() - datetime.fromtimestamp(destino.stat().st_mtime)
            if age < self.ttl:
                return destino

        logger.info("Baixando %s (%.0f MB esperado)...", url, 0)
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        with open(destino, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65_536):
                f.write(chunk)
        logger.info("Salvo: %s (%.1f MB)", destino.name, destino.stat().st_size / 1e6)
        return destino

    def listar_csvs(self, ano: int, tipo: str = "ITR") -> list[str]:
        """Lista CSVs dentro do ZIP sem extrair — útil para inspecionar estrutura."""
        with zipfile.ZipFile(self.baixar_ano(ano, tipo)) as zf:
            return sorted(zf.namelist())

    # ── Cadastro ──────────────────────────────────────────────────────────────

    def get_cadastro(self) -> pd.DataFrame:
        """Baixa e cacheia cad_cia_aberta.csv."""
        if self._cadastro is not None:
            return self._cadastro
        cad_path = self.raw_dir / "cad_cia_aberta.csv"
        if not cad_path.exists() or (
            datetime.now() - datetime.fromtimestamp(cad_path.stat().st_mtime) > self.ttl
        ):
            logger.info("Baixando cadastro CVM...")
            resp = requests.get(_URL_CAD, timeout=60)
            resp.raise_for_status()
            cad_path.write_bytes(resp.content)
        df = pd.read_csv(cad_path, encoding=_ENC, sep=_SEP, dtype={"CD_CVM": int, "CNPJ_CIA": str})
        self._cadastro = df
        return df

    def buscar_por_nome(self, termo: str) -> pd.DataFrame:
        """Busca empresas no cadastro por substring no nome (case-insensitive)."""
        cad = self.get_cadastro()
        mask = (
            cad["DENOM_SOCIAL"].str.contains(termo, case=False, na=False)
            | cad["DENOM_COMERC"].str.contains(termo, case=False, na=False)
        )
        return cad[mask][["CD_CVM", "CNPJ_CIA", "DENOM_SOCIAL", "DENOM_COMERC", "SIT"]].copy()

    def get_cnpj_por_cd_cvm(self) -> dict[int, str]:
        """Retorna mapeamento CD_CVM (int) → CNPJ (str)."""
        cad = self.get_cadastro()
        return dict(zip(cad["CD_CVM"].astype(int), cad["CNPJ_CIA"]))

    # ── Demonstrativos ────────────────────────────────────────────────────────

    def carregar_demonstrativos(self, ano: int, tipo: str = "ITR") -> dict[str, pd.DataFrame]:
        """Parseia e cacheia os demonstrativos do ano em parquet.

        Retorna dict com chaves BPA, BPP, DRE, DFC, CAP.
        Aplica: encoding=latin-1, sep=";", decimal=",".
        Filtra versão máxima por (CD_CVM, DT_REFER, CD_CONTA, ORDEM_EXERC).
        """
        templates = _ITR_TEMPLATES if tipo == "ITR" else _DFP_TEMPLATES
        result: dict[str, pd.DataFrame] = {}
        pendentes: list[str] = []

        # Verificar cache parquet para cada tipo de demonstrativo
        for key in templates:
            p = self.processed_dir / f"{tipo}_{ano}_{key}.parquet"
            if p.exists():
                age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
                if age < self.ttl:
                    try:
                        result[key] = pd.read_parquet(p)
                        continue
                    except Exception:
                        pass
            pendentes.append(key)

        if not pendentes:
            return result

        zip_path = self.baixar_ano(ano, tipo)
        logger.info("Parseando %s %d (%s)...", tipo, ano, ", ".join(pendentes))

        with zipfile.ZipFile(zip_path) as zf:
            arquivos = set(zf.namelist())
            for key in pendentes:
                fname = templates[key].format(ano=ano)
                if fname not in arquivos:
                    logger.debug("%s ausente em %s", fname, zip_path.name)
                    continue

                with zf.open(fname) as f:
                    df = pd.read_csv(
                        f,
                        encoding=_ENC,
                        sep=_SEP,
                        decimal=_DEC,
                        dtype=str,
                        low_memory=False,
                    )

                # Selecionar colunas relevantes
                if key == "HDR":
                    df = self._parsear_hdr(df)
                elif key == "CAP":
                    df = self._parsear_cap(df)
                else:
                    cols = [c for c in _KEEP if c in df.columns]
                    df = df[cols].copy()
                    df = self._parsear_demo(df)

                # Salvar parquet
                p = self.processed_dir / f"{tipo}_{ano}_{key}.parquet"
                df.to_parquet(p, index=False)
                result[key] = df
                logger.debug("Parquet: %s (%d linhas)", p.name, len(df))

        return result

    def _parsear_demo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parseia tipos e deduplica por versão máxima."""
        for col in ("DT_REFER", "DT_RECEB"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"], errors="coerce")
        df["VERSAO"] = pd.to_numeric(df["VERSAO"], errors="coerce").fillna(0).astype(int)
        df["CD_CVM"] = pd.to_numeric(df["CD_CVM"], errors="coerce")

        # Manter apenas a versão máxima por (CD_CVM, DT_REFER, CD_CONTA, ORDEM_EXERC)
        grp = [c for c in ("CD_CVM", "DT_REFER", "CD_CONTA", "ORDEM_EXERC") if c in df.columns]
        df["_vmax"] = df.groupby(grp)["VERSAO"].transform("max")
        df = df[df["VERSAO"] == df["_vmax"]].drop(columns=["_vmax", "VERSAO"])
        return df.drop_duplicates(subset=grp).reset_index(drop=True)

    def _parsear_hdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parseia o arquivo índice (HDR) que contém DT_RECEB por filing."""
        for col in ("DT_REFER", "DT_RECEB"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df["VERSAO"] = pd.to_numeric(df.get("VERSAO", 0), errors="coerce").fillna(0).astype(int)
        df["CD_CVM"] = pd.to_numeric(df.get("CD_CVM", None), errors="coerce")
        # Manter apenas a versão máxima por (CD_CVM, DT_REFER)
        grp = [c for c in ("CD_CVM", "DT_REFER") if c in df.columns]
        if grp:
            df["_vmax"] = df.groupby(grp)["VERSAO"].transform("max")
            df = df[df["VERSAO"] == df["_vmax"]].drop(columns=["_vmax"])
        cols = [c for c in ("CD_CVM", "CNPJ_CIA", "DT_REFER", "DT_RECEB", "VERSAO") if c in df.columns]
        return df[cols].drop_duplicates(subset=grp).reset_index(drop=True)

    def _parsear_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parseia composicao_capital — estrutura diferente dos demonstrativos."""
        for col in ("DT_REFER",):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        for col in ("QT_ACAO_ORDIN_CAP_INTEGR", "QT_ACAO_PREF_CAP_INTEGR",
                    "QT_ACAO_TOTAL_CAP_INTEGR", "QT_ACAO_ORDIN_TESOURO",
                    "QT_ACAO_PREF_TESOURO", "QT_ACAO_TOTAL_TESOURO"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

        if "VERSAO" in df.columns:
            df["VERSAO"] = pd.to_numeric(df["VERSAO"], errors="coerce").fillna(0).astype(int)
            grp = [c for c in ("CNPJ_CIA", "DT_REFER") if c in df.columns]
            df["_vmax"] = df.groupby(grp)["VERSAO"].transform("max")
            df = df[df["VERSAO"] == df["_vmax"]].drop(columns=["_vmax", "VERSAO"])

        return df.reset_index(drop=True)

    # ── Fundamentals ──────────────────────────────────────────────────────────

    def get_fundamentals(
        self,
        cd_cvm: int,
        data_limite: pd.Timestamp,
    ) -> dict | None:
        """Retorna o último demonstrativo cuja DT_RECEB <= data_limite.

        Itens de fluxo (receita, lucro, EBIT, EBITDA): sempre TTM (12 meses).
        Itens de balanço (ativo, caixa, patrimônio, dívida): point-in-time.

        Para ITR Q1/Q2/Q3, o TTM exige DFP do ano anterior com
        DT_RECEB <= data_limite. Se indisponível: campo fica None + aviso.
        Para DFP anual: TTM = anual diretamente, sem cálculo adicional.

        Retorna None se nenhum demonstrativo disponível até data_limite.
        """
        if data_limite.tzinfo is None:
            raise ValueError("data_limite deve ser timezone-aware")

        anos = list(range(max(2011, data_limite.year - 2), data_limite.year + 1))
        candidatos: list[dict] = []

        for ano in anos:
            for tipo in ("DFP", "ITR"):
                try:
                    demos = self.carregar_demonstrativos(ano, tipo)
                except Exception as e:
                    logger.warning("Falha ao carregar %s %d: %s", tipo, ano, e)
                    continue

                dre = demos.get("DRE", pd.DataFrame())
                if dre.empty or "CD_CVM" not in dre.columns:
                    continue

                empresa = dre[dre["CD_CVM"] == cd_cvm]
                if empresa.empty:
                    continue
                if "ORDEM_EXERC" in empresa.columns:
                    empresa = empresa[empresa["ORDEM_EXERC"] == "ÚLTIMO"]
                if "DT_REFER" not in empresa.columns:
                    continue

                # DT_RECEB vem do HDR (índice de filings)
                hdr = demos.get("HDR", pd.DataFrame())
                dt_receb_map: dict = {}
                if not hdr.empty and "CD_CVM" in hdr.columns and "DT_RECEB" in hdr.columns:
                    for _, row in hdr[hdr["CD_CVM"] == cd_cvm].iterrows():
                        dt_receb_map[row["DT_REFER"]] = row["DT_RECEB"]

                for dt_refer in empresa["DT_REFER"].dropna().unique():
                    dt_refer_sp = _localize(dt_refer)
                    if dt_refer_sp is None:
                        continue
                    dt_receb_raw = dt_receb_map.get(dt_refer)
                    dt_receb_sp = (
                        _localize(dt_receb_raw) if dt_receb_raw is not None
                        else dt_refer_sp + pd.Timedelta(days=45)
                    )
                    if dt_receb_sp <= data_limite:
                        candidatos.append({
                            "dt_refer": dt_refer_sp,
                            "dt_receb": dt_receb_sp,
                            "demos": demos,
                            "tipo": tipo,
                        })

        if not candidatos:
            return None

        candidatos.sort(key=lambda x: (x["dt_receb"], x["dt_refer"]), reverse=True)
        melhor = candidatos[0]

        # Para ITR: buscar DFP_prev com anti-lookahead
        demos_dfp_prev = None
        if melhor["tipo"] == "ITR":
            ano_prev = melhor["dt_refer"].year - 1
            try:
                _dfp = self.carregar_demonstrativos(ano_prev, "DFP")
                hdr_dfp = _dfp.get("HDR", pd.DataFrame())
                if not hdr_dfp.empty and "CD_CVM" in hdr_dfp.columns and "DT_RECEB" in hdr_dfp.columns:
                    emp = hdr_dfp[hdr_dfp["CD_CVM"] == cd_cvm]
                    if not emp.empty:
                        dt_receb_dfp = _localize(emp["DT_RECEB"].iloc[0])
                        if dt_receb_dfp is not None and dt_receb_dfp <= data_limite:
                            demos_dfp_prev = _dfp
            except Exception as e:
                logger.warning("DFP_prev (ano=%d) indisponível: %s", ano_prev, e)

        return self._extrair_metricas(
            cd_cvm=cd_cvm,
            dt_refer=melhor["dt_refer"],
            dt_receb=melhor["dt_receb"],
            demos=melhor["demos"],
            tipo_doc=melhor["tipo"],
            demos_dfp_prev=demos_dfp_prev,
        )

    def _extrair_metricas(
        self,
        cd_cvm: int,
        dt_refer: pd.Timestamp,
        dt_receb: pd.Timestamp,
        demos: dict[str, pd.DataFrame],
        tipo_doc: str = "ITR",
        demos_dfp_prev: dict | None = None,
    ) -> dict:
        """Extrai métricas com TTM para fluxo e point-in-time para balanço.

        Fluxo (TTM):       receita, lucro_liquido, ebit, ebitda_aproximado
        Balanço (PIT):     ativo_total, caixa, patrimonio_liquido,
                           divida_bruta, divida_liquida
        """
        avisos: list[str] = []
        is_dfp = (tipo_doc == "DFP")
        dt_naive = dt_refer.tz_localize(None) if dt_refer.tzinfo else dt_refer

        # Label: DFP = "ANUAL-YYYY", ITR = "Q{n}-YYYY"
        if is_dfp:
            trimestre = f"ANUAL-{dt_refer.year}"
        else:
            trimestre = f"Q{(dt_refer.month - 1) // 3 + 1}-{dt_refer.year}"

        # ── Helpers de filtragem e extração ───────────────────────────────────

        def _filt(df: pd.DataFrame, ordem: str) -> pd.DataFrame:
            if df.empty or "CD_CVM" not in df.columns:
                return pd.DataFrame()
            mask = (df["CD_CVM"] == cd_cvm) & (df["DT_REFER"] == dt_naive)
            if "ORDEM_EXERC" in df.columns:
                mask &= df["ORDEM_EXERC"] == ordem
            return df[mask]

        def _v(df: pd.DataFrame, *cds: str, desc: tuple[str, ...] = ()) -> Optional[float]:
            """Extrai VL_CONTA × 1000 por CD_CONTA (ou DS_CONTA como fallback)."""
            if df.empty:
                return None
            for cd in cds:
                if "CD_CONTA" not in df.columns:
                    break
                rows = df[df["CD_CONTA"] == cd]
                if not rows.empty:
                    val = rows["VL_CONTA"].iloc[0]
                    if pd.notna(val):
                        return float(val) * 1000.0
            for t in desc:
                if "DS_CONTA" not in df.columns:
                    break
                rows = df[df["DS_CONTA"].str.contains(t, case=False, na=False)]
                if not rows.empty:
                    val = rows["VL_CONTA"].iloc[0]
                    if pd.notna(val):
                        return float(val) * 1000.0
            return None

        # Corrente: ÚLTIMO e PENÚLTIMO do mesmo filing
        bpa_u = _filt(demos.get("BPA", pd.DataFrame()), "ÚLTIMO")
        bpp_u = _filt(demos.get("BPP", pd.DataFrame()), "ÚLTIMO")
        dre_u = _filt(demos.get("DRE", pd.DataFrame()), "ÚLTIMO")
        dfc_u = _filt(demos.get("DFC", pd.DataFrame()), "ÚLTIMO")
        dre_p = _filt(demos.get("DRE", pd.DataFrame()), "PENÚLTIMO")  # mesmo período, ano-1
        dfc_p = _filt(demos.get("DFC", pd.DataFrame()), "PENÚLTIMO")

        # DFP_prev: frames anuais do ano anterior (apenas para ITR)
        dre_dfp = dfc_dfp = pd.DataFrame()
        if not is_dfp and demos_dfp_prev is not None:
            dre_raw = demos_dfp_prev.get("DRE", pd.DataFrame())
            dfc_raw = demos_dfp_prev.get("DFC", pd.DataFrame())
            # Encontrar DT_REFER do DFP_prev para esta empresa
            emp = dre_raw[dre_raw["CD_CVM"] == cd_cvm] if not dre_raw.empty else pd.DataFrame()
            if "ORDEM_EXERC" in emp.columns:
                emp = emp[emp["ORDEM_EXERC"] == "ÚLTIMO"]
            if not emp.empty:
                dt_dfp_naive = emp["DT_REFER"].iloc[0]
                def _filt_dfp(df: pd.DataFrame) -> pd.DataFrame:
                    if df.empty or "CD_CVM" not in df.columns:
                        return pd.DataFrame()
                    mask = (df["CD_CVM"] == cd_cvm) & (df["DT_REFER"] == dt_dfp_naive)
                    if "ORDEM_EXERC" in df.columns:
                        mask &= df["ORDEM_EXERC"] == "ÚLTIMO"
                    return df[mask]
                dre_dfp = _filt_dfp(dre_raw)
                dfc_dfp = _filt_dfp(dfc_raw)

        # ── TTM helper ────────────────────────────────────────────────────────

        def _ttm(ytd, pen, dfp_anual, nome: str) -> tuple[Optional[float], Optional[str]]:
            """TTM = YTD_atual + (DFP_prev − PENÚLTIMO_YTD). None + aviso se faltar peça."""
            if is_dfp:
                return ytd, None          # DFP já é 12 meses = TTM direto
            if ytd is None:
                return None, f"{nome}: YTD atual ausente"
            if pen is None:
                return None, f"{nome}: PENÚLTIMO ausente no ITR (mesmo período ano anterior)"
            if dfp_anual is None:
                return None, f"{nome}: DFP do ano anterior indisponível ou ainda não publicado"
            return ytd + (dfp_anual - pen), None

        # ── Resultado base ────────────────────────────────────────────────────

        _PIT = "point_in_time"
        _TTM = "TTM"

        resultado: dict = {
            "dt_refer":    dt_refer,
            "dt_receb":    dt_receb,
            "trimestre":   trimestre,
            "tipo_doc":    tipo_doc,
            "avisos":      avisos,
            "periodicidade": {},
        }

        # ── Balanço — point-in-time (foto na DT_REFER) ────────────────────────

        resultado["ativo_total"] = (
            _v(bpa_u, "1", desc=("Ativo Total", "Total do Ativo"))
        )
        resultado["periodicidade"]["ativo_total"] = _PIT

        c1 = _v(bpa_u, "1.01.01", desc=("Caixa e Equivalentes",)) or 0.0
        c2 = _v(bpa_u, "1.01.02") or 0.0
        resultado["caixa"] = (c1 + c2) if (c1 or c2) else None
        resultado["periodicidade"]["caixa"] = _PIT
        if resultado["caixa"] is None:
            avisos.append("caixa: não encontrado no BPA (1.01.01/1.01.02)")

        resultado["patrimonio_liquido"] = (
            _v(bpp_u, "2.03", "2.03.01",
               desc=("Patrimônio Líquido Consolidado", "Patrimônio Líquido"))
        )
        resultado["periodicidade"]["patrimonio_liquido"] = _PIT
        if resultado["patrimonio_liquido"] is None:
            avisos.append("patrimônio líquido: não encontrado no BPP (2.03)")

        div_cp = _v(bpp_u, "2.01.04", desc=("Empréstimos e Financiamentos",)) or 0.0
        div_lp = _v(bpp_u, "2.02.01") or 0.0
        resultado["divida_bruta"] = (div_cp + div_lp) if (div_cp or div_lp) else None
        resultado["periodicidade"]["divida_bruta"] = _PIT
        if resultado["divida_bruta"] is None:
            avisos.append("dívida bruta: não encontrada no BPP (2.01.04/2.02.01)")

        resultado["divida_liquida"] = (
            resultado["divida_bruta"] - resultado["caixa"]
            if resultado["divida_bruta"] is not None and resultado["caixa"] is not None
            else None
        )
        resultado["periodicidade"]["divida_liquida"] = _PIT

        # ── Fluxo — TTM ───────────────────────────────────────────────────────

        # Receita
        rec_u = _v(dre_u, "3.01", desc=("Receita de Venda", "Receita Líquida",
                                         "Resultado Bruto da Intermediação"))
        rec_p = _v(dre_p, "3.01", desc=("Receita de Venda", "Receita Líquida"))
        rec_d = _v(dre_dfp, "3.01", desc=("Receita de Venda", "Receita Líquida"))
        resultado["receita"], av = _ttm(rec_u, rec_p, rec_d, "receita")
        resultado["periodicidade"]["receita"] = _TTM
        if av:
            avisos.append(av)
        if rec_u is None:
            avisos.append("receita: não encontrada no DRE (3.01)")

        # EBIT
        ebit_u = _v(dre_u, "3.05", desc=("Resultado Antes do Resultado Financeiro",
                                           "Resultado Antes dos Tributos"))
        ebit_p = _v(dre_p, "3.05", desc=("Resultado Antes do Resultado Financeiro",))
        ebit_d = _v(dre_dfp, "3.05", desc=("Resultado Antes do Resultado Financeiro",))
        resultado["ebit"], av = _ttm(ebit_u, ebit_p, ebit_d, "ebit")
        resultado["periodicidade"]["ebit"] = _TTM
        if av:
            avisos.append(av)

        # Lucro líquido (prioridade: 3.11.01 = atribuível à controladora)
        _ll_cds = ("3.11.01", "3.11", "3.13")
        _ll_desc = ("Lucro/Prejuízo Consolidado do Período",
                    "Lucro/Prejuízo do Período", "Resultado Líquido do Período")
        ll_u = _v(dre_u, *_ll_cds, desc=_ll_desc)
        ll_p = _v(dre_p, *_ll_cds, desc=_ll_desc)
        ll_d = _v(dre_dfp, *_ll_cds, desc=_ll_desc)
        resultado["lucro_liquido"], av = _ttm(ll_u, ll_p, ll_d, "lucro_liquido")
        resultado["periodicidade"]["lucro_liquido"] = _TTM
        if av:
            avisos.append(av)
        if ll_u is None:
            avisos.append("lucro_liquido: não encontrado no DRE (3.11/3.11.01)")

        # D&A — também TTM (vem do DFC)
        _da_desc = ("Depreciação e Amortização", "Depreciação,", "Amortização de")
        _da_cds  = ("6.01.01.05", "6.01.01.02")
        da_u = _v(dfc_u, *_da_cds, desc=_da_desc)
        da_p = _v(dfc_p, *_da_cds, desc=_da_desc)
        da_d = _v(dfc_dfp, *_da_cds, desc=_da_desc)
        # D&A no DFC indireto aparece como valor positivo (adição ao lucro)
        da_u = abs(da_u) if da_u is not None else None
        da_p = abs(da_p) if da_p is not None else None
        da_d = abs(da_d) if da_d is not None else None
        da_ttm, av_da = _ttm(da_u, da_p, da_d, "depreciacao_amortizacao")

        # EBITDA = EBIT_TTM + D&A_TTM
        resultado["periodicidade"]["ebitda_aproximado"] = _TTM
        if resultado["ebit"] is not None and da_ttm is not None:
            resultado["ebitda_aproximado"] = resultado["ebit"] + da_ttm
            avisos.append(
                f"ebitda_aproximado = EBIT_TTM + D&A_TTM ({da_ttm/1e6:.0f}M); não é EBITDA oficial"
            )
        elif resultado["ebit"] is not None:
            resultado["ebitda_aproximado"] = resultado["ebit"]
            msg = av_da or "D&A não encontrado no DFC"
            avisos.append(f"ebitda_aproximado: {msg}; usando apenas EBIT_TTM (subestimado)")
        else:
            resultado["ebitda_aproximado"] = None
            avisos.append("ebitda_aproximado: EBIT indisponível")

        return resultado

    # ── Ações em circulação ───────────────────────────────────────────────────

    def get_acoes_em_circulacao(
        self,
        cd_cvm: int,
        data_limite: pd.Timestamp,
        cnpj: Optional[str] = None,
    ) -> int | None:
        """Retorna ações em circulação (total - tesouraria) na data_limite.

        Usa composicao_capital do ITR. Sem DT_RECEB neste arquivo — usa
        DT_REFER <= data_limite como proxy (conservador).
        Requer cnpj para lookup (composicao_capital usa CNPJ, não CD_CVM).
        """
        if data_limite.tzinfo is None:
            raise ValueError("data_limite deve ser timezone-aware")

        # Resolver CNPJ se não fornecido
        if cnpj is None:
            mapa = self.get_cnpj_por_cd_cvm()
            cnpj = mapa.get(cd_cvm)
        if cnpj is None:
            logger.warning("CNPJ não encontrado para CD_CVM=%d", cd_cvm)
            return None

        ano = data_limite.year
        # Tentar o ano da data_limite e o anterior
        for a in (ano, ano - 1):
            try:
                demos = self.carregar_demonstrativos(a, "ITR")
                cap = demos.get("CAP", pd.DataFrame())
                if cap.empty or "CNPJ_CIA" not in cap.columns:
                    continue

                empresa = cap[cap["CNPJ_CIA"] == cnpj].copy()
                if empresa.empty:
                    continue

                dt_limite_naive = data_limite.tz_localize(None)
                empresa = empresa[empresa["DT_REFER"] <= dt_limite_naive]
                if empresa.empty:
                    continue

                empresa = empresa.sort_values("DT_REFER", ascending=False)
                row = empresa.iloc[0]

                total = int(row.get("QT_ACAO_TOTAL_CAP_INTEGR", 0))
                tesouro = int(row.get("QT_ACAO_TOTAL_TESOURO", 0))
                return max(0, total - tesouro)

            except Exception as e:
                logger.warning("get_acoes_em_circulacao ano=%d: %s", a, e)

        return None
