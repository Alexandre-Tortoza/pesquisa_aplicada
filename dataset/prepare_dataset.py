#!/usr/bin/env python3

"""
Merge compacto: Tráfego x População agregada (região + sexo + ano)

- Lê:
  clean/clean_population.csv  (saída do clear_population.py)
  clean/clean_traffic.csv     (saída do clear_traffic.py)

- Padroniza todas as colunas para português em snake_case
- Agrega população por: ['regiao','sexo','ano'] somando todas as idades
- Extrai 'ano' do tráfego a partir de 'day' ou 'year'
- Faz merge por ['regiao','ano'] resultando em ATÉ 2 linhas por registro de tráfego (Homens/Mulheres)
- Exporta:
  merged_dataset_compacto.csv
  data_quality_report_compacto.json

Autor: Alexandre Marques Tortoza Canoa
Versão do Python: 3.13.7
"""

from pathlib import Path
from typing import Optional, Dict, List
import json
import unicodedata
import re

import pandas as pd
import numpy as np

POPULATION_FILE = Path("./clean/clean_population.csv")
TRAFFIC_FILE    = Path("./clean/clean_traffic.csv")

OUTPUT_FILE         = Path("./preparedData/dataset.csv")
QUALITY_REPORT_FILE = Path("./preparedData/quality_report.json")

CSV_SEP   = ";"
ENCODING  = "utf-8"
MIN_YEAR  = 2018

REGIONS = ["center", "north", "south", "east", "west"]

COLUMN_MAP = {
    "region": "regiao",
    "sexo": "sexo",
    "ano": "ano",
    "year": "ano",
    "day": "data",
    "hour": "hora",
    "expressway": "via_expressa",
    "road": "rodovia",
    "congestion_size": "tamanho_congestionamento",
    "populacao": "populacao",
    "population": "populacao",
    "id": "id",
}

def info(msg: str):
    print(f"ⓘ {msg}")

def ok(msg: str):
    print(f"✓ {msg}")

def warn(msg: str):
    print(f"⚠ {msg}")

def header(t: str):
    line = "=" * 72
    print(f"\n{line}\n{t.center(72)}\n{line}")

def print_step(step: int, total: int, description: str) -> None:
    """Imprime um indicador formatado da etapa de processamento."""
    print(f"\n[{step}/{total}] {description}")

def create_ascii_bar_chart(counts: Dict[str, int], max_width: int = 40) -> List[str]:
    """Cria um gráfico de barras horizontal em ASCII."""
    if not counts:
        return []

    max_count = max(counts.values())
    total_count = sum(counts.values())
    lines = []

    for label, count in counts.items():
        if max_count > 0:
            bar_length = int((count / max_count) * max_width)
        else:
            bar_length = 0

        bar = "█" * bar_length
        percentage = (count / total_count * 100) if total_count > 0 else 0

        line = f"   {label:8s}: {bar:<{max_width}s} {count:>6,d} "
        line += f"({percentage:5.1f}%)"
        lines.append(line)

    return lines

def read_csv_safe(path: Path, sep: str = CSV_SEP, encoding: str = ENCODING) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, sep=sep, encoding=encoding)
        ok(f"Lido {path} com {len(df):,} linhas")
        return df
    except FileNotFoundError:
        warn(f"Arquivo não encontrado: {path}")
        return None
    except Exception as e:
        warn(f"Erro ao ler {path}: {e}")
        return None


def normalize_string(s: str) -> str:
    """Remove acentos, minúsculas, converte espaços/caracteres para underscore."""
    nfd = unicodedata.normalize("NFD", s)
    ascii_str = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    ascii_str = ascii_str.lower().strip()
    ascii_str = re.sub(r"[^\w]+", "_", ascii_str)
    ascii_str = re.sub(r"^_+|_+$", "", ascii_str)
    return ascii_str

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes de colunas:
    1. Aplica mapeamento explícito (COLUMN_MAP)
    2. Para colunas não mapeadas, normaliza removendo acentos e convertendo para snake_case
    3. Aplica regra de 'ano' (prioriza 'year' sobre 'day')
    4. Remove coluna 'via' se existir
    """
    df = df.copy()
    
    
    rename_dict = {}
    for old_col in df.columns:
        if old_col in COLUMN_MAP:
            rename_dict[old_col] = COLUMN_MAP[old_col]
        else:
            
            rename_dict[old_col] = normalize_string(old_col)
    
    df = df.rename(columns=rename_dict)
    
    
    if "via" in df.columns:
        df = df.drop(columns=["via"])
    
    return df




def validate_region_col(df: pd.DataFrame, name: str) -> bool:
    """Valida que 'regiao' existe e tem valores válidos."""
    if "regiao" not in df.columns:
        warn(f"{name}: coluna 'regiao' não encontrada")
        return False
    uniq = df["regiao"].dropna().unique()
    invalid = [r for r in uniq if r not in REGIONS]
    if invalid:
        warn(f"{name}: regiões inválidas detectadas: {invalid}")
    valid_count = df["regiao"].isin(REGIONS).sum()
    ok(f"{name}: regiões válidas em {valid_count:,}/{len(df):,} linhas")
    return True

def filter_year(df: pd.DataFrame, col: str, min_year: int = MIN_YEAR) -> pd.DataFrame:
    """Filtra por ano mínimo."""
    if col not in df.columns:
        warn(f"Coluna de ano '{col}' não existe; sem filtro aplicado")
        return df
    before = len(df)
    out = df[df[col] >= min_year].copy()
    removed = before - len(out)
    ok(f"Filtro ano >= {min_year}: {len(out):,} linhas (removidas {removed:,})")
    return out

def filter_valid_regions(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Filtra mantendo apenas regiões válidas."""
    before = len(df)
    out = df[df["regiao"].isin(REGIONS)].copy()
    removed = before - len(out)
    if removed > 0:
        warn(f"{name}: removidas {removed:,} linhas com regiões inválidas")
    ok(f"{name}: {len(out):,} linhas com regiões válidas")
    return out




def describe_numeric(df: pd.DataFrame) -> Dict:
    """Gera estatísticas descritivas para colunas numéricas."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        stats[c] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
        }
    return stats

def generate_statistical_report(merged: pd.DataFrame, traf: pd.DataFrame) -> None:
    """Gera e imprime um relatório estatístico formatado."""
    header("RESUMO ESTATÍSTICO DOS DADOS PROCESSADOS")

    total_records = len(merged)
    print(f"\nTotal de registros (saída): {total_records:,}")
    print(f"Registros de tráfego (entrada): {len(traf):,}")

    if "regiao" in merged.columns:
        region_counts = (merged["regiao"]
                        .value_counts()
                        .sort_index()
                        .to_dict())

        print(f"\nDistribuição por região:")
        bar_chart = create_ascii_bar_chart(region_counts)
        for line in bar_chart:
            print(line)

    if "tamanho_congestionamento" in merged.columns:
        congestion_stats = merged["tamanho_congestionamento"].describe()
        print(f"\nEstatísticas de tamanho de congestionamento (metros):")
        print(f"   Média       : {congestion_stats['mean']:>10,.2f}")
        print(f"   Mediana     : {congestion_stats['50%']:>10,.2f}")
        print(f"   Mínimo      : {congestion_stats['min']:>10,.2f}")
        print(f"   Máximo      : {congestion_stats['max']:>10,.2f}")
        print(f"   Desvio Pad. : {congestion_stats['std']:>10,.2f}")

    if "data" in merged.columns:
        try:
            merged_copy = merged.copy()
            merged_copy["data"] = pd.to_datetime(merged_copy["data"])
            min_date = merged_copy["data"].min()
            max_date = merged_copy["data"].max()
            date_range = (max_date - min_date).days
            print(f"\nPeríodo dos dados:")
            print(f"   Data inicial: {min_date.strftime('%Y-%m-%d')}")
            print(f"   Data final  : {max_date.strftime('%Y-%m-%d')}")
            print(f"   Duração     : {date_range:,} dias")
        except Exception as e:
            warn(f"Erro ao processar datas: {e}")

    if "via_expressa" in merged.columns:
        unique_roads = merged["via_expressa"].nunique()
        print(f"\nVias expressas únicas no dataset: {unique_roads:,}")

    if "sexo" in merged.columns:
        sexo_counts = merged["sexo"].value_counts().sort_index().to_dict()
        print(f"\nDistribuição por sexo:")
        bar_chart = create_ascii_bar_chart(sexo_counts)
        for line in bar_chart:
            print(line)

    if "pop_total" in merged.columns:
        pop_stats = merged["pop_total"].describe()
        print(f"\nEstatísticas de população total:")
        print(f"   Média       : {pop_stats['mean']:>10,.2f}")
        print(f"   Mediana     : {pop_stats['50%']:>10,.2f}")
        print(f"   Mínimo      : {pop_stats['min']:>10,.2f}")
        print(f"   Máximo      : {pop_stats['max']:>10,.2f}")
        print(f"   Desvio Pad. : {pop_stats['std']:>10,.2f}")




def main():
    header("MERGE COMPACTO - TRAFEGO x POPULACAO")

    
    print_step(1, 9, "Carregando dados...")
    pop = read_csv_safe(POPULATION_FILE)
    traf = read_csv_safe(TRAFFIC_FILE)
    if pop is None or traf is None:
        return

    
    print_step(2, 9, "Padronizando nomes de colunas...")
    pop = standardize_columns(pop)
    traf = standardize_columns(traf)
    ok("Colunas padronizadas para português snake_case")

    
    print_step(3, 9, "Validando regiões...")
    if not validate_region_col(pop, "População"):
        return
    if not validate_region_col(traf, "Tráfego"):
        return

    
    pop = filter_valid_regions(pop, "População")
    traf = filter_valid_regions(traf, "Tráfego")

    
    print_step(4, 9, "Garantindo coluna 'ano'...")
    if "ano" not in traf.columns:
        if "data" in traf.columns:
            info("Extraindo 'ano' de 'data' no tráfego...")
            try:
                traf["ano"] = pd.to_datetime(traf["data"]).dt.year
                ok("Coluna 'ano' criada a partir de 'data'")
            except Exception as e:
                warn(f"Falha ao extrair 'ano' de 'data': {e}")
                traf["ano"] = pd.NA
        else:
            warn("Tráfego sem 'ano' nem 'data'; criando 'ano' vazio")
            traf["ano"] = pd.NA

    if "ano" not in pop.columns:
        warn("População sem coluna 'ano'")
        return

    
    print_step(5, 9, "Filtrando por ano mínimo...")
    pop = filter_year(pop, "ano", MIN_YEAR)
    traf = filter_year(traf, "ano", MIN_YEAR)

    
    print_step(6, 9, "Agregando população...")
    req_cols = {"regiao", "sexo", "ano", "populacao"}
    missing = req_cols - set(pop.columns)
    if missing:
        warn(f"População sem colunas necessárias: {sorted(missing)}")
        return

    info("Agregando população por [regiao, sexo, ano]...")
    pop_agg = (
        pop.groupby(["regiao", "sexo", "ano"], as_index=False)["populacao"]
           .sum()
           .rename(columns={"populacao": "pop_total"})
    )
    ok(f"População agregada: {len(pop_agg):,} linhas")

    
    print_step(7, 9, "Validando agregação de população...")
    counts = pop_agg.groupby(["regiao", "ano"]).size().reset_index(name="rows")
    offenders = counts[counts["rows"] > 2]
    if not offenders.empty:
        warn("Há chaves (regiao, ano) com mais de 2 linhas em pop_agg (esperado <=2):")
        print(offenders.sort_values(["regiao", "ano"]).to_string(index=False))
    else:
        ok("Validação passed: max 2 linhas por (regiao, ano)")

    
    print_step(8, 9, "Realizando merge...")
    info("Merge por ['regiao','ano']...")
    merged = traf.merge(
        pop_agg,
        on=["regiao", "ano"],
        how="inner",
        validate="m:m"
    )
    ok(f"Merge concluído: {len(merged):,} linhas (vs tráfego {len(traf):,})")

    
    print_step(9, 9, "Preparando saída final...")

    
    sort_cols = []
    if "data" in merged.columns:
        sort_cols.append("data")
    if "hora" in merged.columns:
        sort_cols.append("hora")
    sort_cols.extend(["regiao", "sexo"])
    
    if sort_cols:
        merged = merged.sort_values(sort_cols, na_position="last")

    
    final_cols = ["data", "hora", "via_expressa", "tamanho_congestionamento", "regiao", "sexo", "pop_total"]
    final_cols = [c for c in final_cols if c in merged.columns]
    merged = merged[final_cols]

    ok(f"Colunas finais selecionadas: {final_cols}")

    
    report = {
        "rows_traffic_in": int(len(traf)),
        "rows_population_in": int(len(pop)),
        "rows_out": int(len(merged)),
        "unique_regioes": int(merged["regiao"].nunique()) if "regiao" in merged.columns else None,
        "year_range_traffic": {
            "min": int(traf["ano"].min()) if "ano" in traf.columns and traf["ano"].notna().any() else None,
            "max": int(traf["ano"].max()) if "ano" in traf.columns and traf["ano"].notna().any() else None,
        },
        "pop_agg_rows": int(len(pop_agg)),
        "numeric_describe": describe_numeric(merged),
        "note": "Cada linha de tráfego foi expandida para no máximo 2 linhas por sexo, usando população total (>=20 anos) por região e ano.",
    }

    
    info("Exportando arquivos...")
    try:
        merged.to_csv(OUTPUT_FILE, sep=CSV_SEP, index=False, encoding=ENCODING)
        ok(f"Dataset salvo em: {OUTPUT_FILE}")
        with open(QUALITY_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        ok(f"Relatório salvo em: {QUALITY_REPORT_FILE}")
    except Exception as e:
        warn(f"Erro ao salvar saídas: {e}")
        return

    
    generate_statistical_report(merged, traf)

    header("PROCESSO CONCLUIDO COM SUCESSO!")
    print(f"\n- Registros de tráfego (filtrado): {len(traf):,}")
    print(f"- Registros de população (filtrado): {len(pop):,}")
    print(f"- Registros saída (esperado ~2x tráfego): {len(merged):,}\n")

if __name__ == "__main__":
    main()