#!/usr/bin/env python3

import pandas as pd


def load_and_clean_dataset(file_path: str, output_path: str | None = None) -> pd.DataFrame:
    """
    Carrega, limpa e prepara o dataset de congestionamento urbano para análise.

    Etapas executadas:
    1. Lê o arquivo CSV informado (tentando UTF-8 e fallback para Latin-1).
    2. Remove espaços e padroniza os nomes das colunas em snake_case.
    3. Seleciona apenas as colunas relevantes: 'day', 'hour', 'congestion_size', 'region'.
    4. Converte tipos de dados:
        - 'day' → datetime (YYYY-MM-DD)
        - 'hour' → string
        - 'congestion_size' → numérico (int ou float)
    5. Filtra os registros de 2018 em diante.
    6. (Opcional) Salva o resultado em um novo arquivo CSV, se `output_path` for fornecido.

    Args:
        file_path (str): Caminho do arquivo CSV original.
        output_path (str | None): Caminho para salvar o CSV limpo (opcional).

    Returns:
        pd.DataFrame: DataFrame limpo contendo apenas as colunas essenciais.
    """

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    selected_columns = ["day", "hour", "congestion_size", "region"]
    existing_columns = [col for col in selected_columns if col in df.columns]
    cleaned_df = df[existing_columns].copy()

    if "day" in cleaned_df.columns:
        cleaned_df["day"] = pd.to_datetime(cleaned_df["day"], errors="coerce")

    if "hour" in cleaned_df.columns:
        cleaned_df["hour"] = cleaned_df["hour"].astype(str).str.strip()

    if "congestion_size" in cleaned_df.columns:
        cleaned_df["congestion_size"] = pd.to_numeric(
            cleaned_df["congestion_size"], errors="coerce"
        )

    if "region" in cleaned_df.columns:
        cleaned_df["region"] = cleaned_df["region"].astype(str).str.strip()

    if "day" in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df["day"].dt.year >= 2018].copy()

    # 6. (Opcional) Exporta o DataFrame limpo
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"✅ Dataset limpo salvo em: {output_path}")

    return cleaned_df


if __name__ == "__main__":
    input_file = "./rawData/sp_traffic_congestions.csv"
    output_file = "dados_congestionamento_limpo.csv"

    dataset = load_and_clean_dataset(input_file, output_file)
    print(dataset.head())
    print(f"\n📊 Registros totais após limpeza: {len(dataset)}")

