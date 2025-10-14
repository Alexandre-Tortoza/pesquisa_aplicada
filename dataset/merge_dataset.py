#!/usr/bin/env python3

import pandas as pd


def merge_population_and_traffic(population_path: str, traffic_path: str, output_path: str | None = None) -> pd.DataFrame:
    """
    Combina os datasets de população e congestionamento urbano, unificando por ano e região/distrito.

    Etapas executadas:
    1. Carrega os dois arquivos CSV.
    2. Extrai o ano do campo 'day' no dataset de tráfego.
    3. Agrega o tráfego por 'ano' e 'region' (somando o congestionamento total).
    4. Faz o merge com o dataset de população usando o ano e a correspondência entre 'region' e 'distritos'.
    5. (Opcional) Exporta o dataset final.

    Args:
        population_path (str): Caminho do dataset de população filtrado.
        traffic_path (str): Caminho do dataset de congestionamento limpo.
        output_path (str | None): Caminho para salvar o dataset combinado (opcional).

    Returns:
        pd.DataFrame: Dataset combinado com informações de população e tráfego.
    """

    pop_df = pd.read_csv(population_path)
    traffic_df = pd.read_csv(traffic_path)

    pop_df.columns = pop_df.columns.str.strip().str.lower()
    traffic_df.columns = traffic_df.columns.str.strip().str.lower()

    if "day" in traffic_df.columns:
        traffic_df["day"] = pd.to_datetime(traffic_df["day"], errors="coerce")
        traffic_df["ano"] = traffic_df["day"].dt.year

    traffic_grouped = (
        traffic_df.groupby(["ano", "region"], as_index=False)["congestion_size"]
        .sum()
        .rename(columns={"region": "distritos", "congestion_size": "total_congestion"})
    )

    merged_df = pd.merge(
        pop_df,
        traffic_grouped,
        how="left",
        on=["ano", "distritos"]
    )

    if output_path:
        merged_df.to_csv(output_path, index=False)
        print(f"✅ Dataset combinado salvo em: {output_path}")

    return merged_df


if __name__ == "__main__":
    population_file = "./dados_populacao_filtrado.csv"
    traffic_file = "./dados_congestionamento_limpo.csv"
    output_file = "dataset_merged.csv"

    dataset = merge_population_and_traffic(population_file, traffic_file, output_file)

    print(dataset.head())
    print(f"\n📊 Registros combinados: {len(dataset)}")

