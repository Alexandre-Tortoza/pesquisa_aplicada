#!/usr/bin/env python3

import pandas as pd


def load_and_clean_population(file_path: str, output_path: str | None = None) -> pd.DataFrame:
    """
    Carrega, limpa e prepara o dataset de população por distrito para análise.

    Etapas executadas:
    1. Lê o arquivo CSV (com fallback de codificação).
    2. Normaliza os nomes das colunas para snake_case.
    3. Seleciona apenas as colunas relevantes.
    4. Converte tipos de dados:
        - 'ano' -> int
        - 'populacao' -> numérico
        - 'idade' -> intervalo tratado para extrair o limite inferior.
    5. Filtra:
        - anos entre 2018 e 2025 (inclusive),
        - faixas etárias com idade mínima > 19 anos.
    6. Remove a coluna auxiliar 'idade_minima' antes de salvar.
    7. (Opcional) Salva o resultado em um novo CSV.

    Args:
        file_path (str): Caminho do arquivo CSV original.
        output_path (str | None): Caminho para salvar o CSV limpo (opcional).

    Returns:
        pd.DataFrame: DataFrame limpo e filtrado.
    """

    try:
        df = pd.read_csv(file_path, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, sep=";", encoding="latin1")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    selected_columns = ["ano", "cod_distr", "distritos", "sexo", "idade", "populacao"]
    existing_columns = [col for col in selected_columns if col in df.columns]
    cleaned_df = df[existing_columns].copy()

    if "ano" in cleaned_df.columns:
        cleaned_df["ano"] = pd.to_numeric(cleaned_df["ano"], errors="coerce").astype("Int64")

    if "populacao" in cleaned_df.columns:
        cleaned_df["populacao"] = pd.to_numeric(cleaned_df["populacao"], errors="coerce")

    if "idade" in cleaned_df.columns:
        cleaned_df["idade_minima"] = (
            cleaned_df["idade"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    if "ano" in cleaned_df.columns:
        cleaned_df = cleaned_df[
            (cleaned_df["ano"] >= 2018) & (cleaned_df["ano"] <= 2025)
        ]

    if "idade_minima" in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df["idade_minima"] > 19]

    cleaned_df = cleaned_df.drop(columns=["idade_minima"], errors="ignore")

    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"✅ Dataset limpo salvo em: {output_path}")

    return cleaned_df


if __name__ == "__main__":
    input_file = "./rawData/estimativa_pop_idade_sexo_msp.csv"
    output_file = "dados_populacao_filtrado.csv"

    dataset = load_and_clean_population(input_file, output_file)

    print(dataset.head())
    print(f"\n📊 Registros totais após filtros: {len(dataset)}")

