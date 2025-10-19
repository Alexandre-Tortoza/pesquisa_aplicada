#!/usr/bin/env python3
"""
    Script de processamento de dados de congestionamentos de São Paulo.

    Este script realiza as seguintes operações:
    - Carrega dados de congestionamentos com informações de localização e tamanho
    - Normaliza textos (remove acentos e espaços extras)
    - Valida e formata campos de data e hora
    - Remove registros duplicados
    - Normaliza nomes de vias e regiões
    - Gera relatório estatístico abrangente
    - Exporta os dados processados para CSV

    Autor: Alexandre Marques Tortoza Canoa
    Versão do Python: 3.13.7
"""

import json
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

INPUT_FILE: Path = Path("./rawData/sp_traffic_congestions.csv")
OUTPUT_FILE: Path = Path("./clean/clean_traffic.csv")
INVALID_RECORDS_FILE: Path = Path("zInvalid_traffic_records.json")
INPUT_CSV_SEPARATOR: str = ","
OUTPUT_CSV_SEPARATOR: str = ";"
INPUT_ENCODING: str = "iso-8859-1"
OUTPUT_ENCODING: str = "utf-8"

REGIONS: List[str] = ["center", "north", "south", "east", "west"]


def remove_accents(text: str) -> str:
    """
        Remove acentos de strings usando normalização Unicode.

        Esta função converte caracteres acentuados para seus equivalentes ASCII.
        Por exemplo: 'São Paulo' -> 'Sao Paulo'

        Args:
            text: String que pode conter caracteres acentuados.

        Returns:
            String sem acentos, ou o valor original se não for uma string.

        Examples:
            >>> remove_accents("Marginal Tietê")
            'Marginal Tiete'
            >>> remove_accents("Av. Paulista")
            'Av. Paulista'
    """
    if isinstance(text, str):
        normalized = unicodedata.normalize("NFKD", text)
        return normalized.encode("ASCII", "ignore").decode("utf-8")
    return text


def normalize_text(text: str) -> str:
    """
        Normaliza texto removendo acentos e espaços extras.

        Args:
            text: String a ser normalizada.

        Returns:
            String normalizada sem acentos e com espaços simples.

        Examples:
            >>> normalize_text("Marginal  Tietê  ")
            'Marginal Tiete'
    """
    if isinstance(text, str):
        text = remove_accents(text)
        text = " ".join(text.split())
        return text.strip()
    return text


def normalize_region(region: str) -> str:
    """
        Normaliza nomes de regiões para o formato padronizado.

        Args:
            region: Nome da região (pode estar em diversos formatos).

        Returns:
            Nome da região normalizado em minúsculas.

        Examples:
            >>> normalize_region("CENTER")
            'center'
            >>> normalize_region("  South  ")
            'south'
    """
    if isinstance(region, str):
        normalized = normalize_text(region).lower()
        if normalized in REGIONS:
            return normalized
    return region


def validate_datetime_fields(
    day: str,
    hour: str
) -> Tuple[bool, Optional[datetime]]:
    """
        Valida se os campos de data e hora formam uma datetime válida.

        Args:
            day: Data no formato YYYY-MM-DD.
            hour: Hora no formato HH:MM:SS.

        Returns:
            Tupla (is_valid, datetime_object).

        Examples:
            >>> validate_datetime_fields("2012-01-01", "21:30:00")
            (True, datetime(2012, 1, 1, 21, 30))
            >>> validate_datetime_fields("invalid", "time")
            (False, None)
    """
    try:
        datetime_str = f"{day} {hour}"
        datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        return True, datetime_obj
    except (ValueError, TypeError):
        return False, None


def validate_congestion_size(size: any) -> bool:
    """
        Valida se o tamanho do congestionamento é um valor numérico positivo.

        Args:
            size: Valor do tamanho do congestionamento.

        Returns:
            True se válido, False caso contrário.

        Examples:
            >>> validate_congestion_size(1300)
            True
            >>> validate_congestion_size(-100)
            False
            >>> validate_congestion_size("invalid")
            False
    """
    try:
        size_value = float(size)
        return size_value > 0
    except (ValueError, TypeError):
        return False


def create_ascii_bar_chart(
    counts: Dict[str, int],
    max_width: int = 40
) -> List[str]:
    """
        Cria um gráfico de barras horizontal em ASCII a partir de dados de contagem.

        Args:
            counts: Dicionário que mapeia rótulos para contagens.
            max_width: Largura máxima das barras em caracteres.

        Returns:
            Lista de strings formatadas representando o gráfico de barras.

        Examples:
            >>> create_ascii_bar_chart({"center": 1000, "north": 500}, 20)
            ['   center  : ███████████████████ 1000 (66.7%)',
            '   north   : ██████████           500 (33.3%)']
    """
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


def print_header(title: str) -> None:
    """
        Imprime um cabeçalho de seção formatado.

        Args:
            title: Texto do título do cabeçalho.
    """
    separator = "=" * 70
    print(f"\n{separator}")
    print(title.center(70))
    print(f"{separator}")


def print_step(step: int, total: int, description: str) -> None:
    """
        Imprime um indicador formatado da etapa de processamento.

        Args:
            step: Número da etapa atual.
            total: Número total de etapas.
            description: Descrição breve da etapa.
    """
    print(f"\n[{step}/{total}] {description}")


def print_success(message: str, value: Optional[str] = None) -> None:
    """
        Imprime mensagem de sucesso com marca de verificação.

        Args:
            message: Texto da mensagem de sucesso.
            value: Valor opcional a ser exibido após a mensagem.
    """
    if value:
        print(f"   ✓ {message}: {value}")
    else:
        print(f"   ✓ {message}")


def print_warning(message: str) -> None:
    """
        Imprime mensagem de aviso.

        Args:
            message: Texto da mensagem de aviso.
    """
    print(f"   ⚠ {message}")


def print_info(message: str) -> None:
    """
        Imprime mensagem informativa.

        Args:
            message: Texto da mensagem informativa.
    """
    print(f"   ⓘ {message}")


def save_invalid_records(
    records: List[Dict],
    filename: Path = INVALID_RECORDS_FILE
) -> None:
    """
        Salva registros inválidos em um arquivo JSON para auditoria.

        Args:
            records: Lista de dicionários contendo registros inválidos.
            filename: Caminho do arquivo JSON de saída.
    """
    data = {
        "invalid_records": records,
        "count": len(records)
    }

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def generate_statistical_report(dataframe: pd.DataFrame) -> None:
    """
        Gera e imprime um relatório estatístico simples e consistente.

        Args:
            dataframe: DataFrame pandas processado contendo dados de congestionamento.
    """
    print_header("RESUMO ESTATISTICO DOS DADOS PROCESSADOS")

    total_records = len(dataframe)
    print(f"\nTotal de registros: {total_records:,}")

    if "region" in dataframe.columns:
        region_counts = (dataframe["region"]
                        .value_counts()
                        .sort_index()
                        .to_dict())

        print(f"\nDistribuicao por regiao:")
        bar_chart = create_ascii_bar_chart(region_counts)
        for line in bar_chart:
            print(line)

    if "congestion_size" in dataframe.columns:
        congestion_stats = dataframe["congestion_size"].describe()
        print(f"\nEstatisticas de tamanho de congestionamento (metros):")
        print(f"   Media       : {congestion_stats['mean']:>10,.2f}")
        print(f"   Mediana     : {congestion_stats['50%']:>10,.2f}")
        print(f"   Minimo      : {congestion_stats['min']:>10,.2f}")
        print(f"   Maximo      : {congestion_stats['max']:>10,.2f}")
        print(f"   Desvio Pad. : {congestion_stats['std']:>10,.2f}")

    if "day" in dataframe.columns:
        dataframe_copy = dataframe.copy()
        dataframe_copy["day"] = pd.to_datetime(dataframe_copy["day"])
        min_date = dataframe_copy["day"].min()
        max_date = dataframe_copy["day"].max()
        date_range = (max_date - min_date).days
        print(f"\nPeriodo dos dados:")
        print(f"   Data inicial: {min_date.strftime('%Y-%m-%d')}")
        print(f"   Data final  : {max_date.strftime('%Y-%m-%d')}")
        print(f"   Duracao     : {date_range:,} dias")

    unique_roads = dataframe["road"].nunique()
    print(f"\nVias unicas no dataset: {unique_roads:,}")


def validate_data_quality(dataframe: pd.DataFrame) -> List[Dict]:
    """
        Valida a qualidade dos dados e identifica registros problemáticos.

        Args:
            dataframe: DataFrame para validação.

        Returns:
            Lista de registros inválidos com motivo da invalidação.
    """
    invalid_records = []

    for index, row in dataframe.iterrows():
        issues = []

        is_valid_datetime, _ = validate_datetime_fields(
            str(row["day"]),
            str(row["hour"])
        )
        if not is_valid_datetime:
            issues.append("invalid_datetime")

        if not validate_congestion_size(row["congestion_size"]):
            issues.append("invalid_congestion_size")

        if row["region"] not in REGIONS:
            issues.append("invalid_region")

        if issues:
            invalid_records.append({
                "index": int(index),
                "id": str(row["id"]),
                "issues": issues,
                "data": row.to_dict()
            })

    return invalid_records


def main() -> None:
    """
        Função principal para processar o conjunto de dados de congestionamentos.

        Orquestra o pipeline completo:
        1. Carregar dados do CSV
        2. Normalizar campos de texto
        3. Normalizar regiões
        4. Validar datas e horas
        5. Validar tamanhos de congestionamento
        6. Remover duplicatas
        7. Salvar dados processados em CSV
        8. Gerar relatório estatístico
    """
    print_header("TRAFFIC CONGESTION DATA PROCESSING - SAO PAULO")
    print(f"Arquivo de entrada: {INPUT_FILE}")

    print_step(1, 7, "Carregando dados...")
    try:
        dataframe = pd.read_csv(
            INPUT_FILE,
            sep=INPUT_CSV_SEPARATOR,
            encoding=INPUT_ENCODING
        )
        print_success("Registros carregados", f"{len(dataframe):,}")
    except FileNotFoundError:
        print_warning(f"Arquivo nao encontrado: {INPUT_FILE}")
        return
    except Exception as error:
        print_warning(f"Erro ao carregar arquivo: {error}")
        return

    print_step(2, 7, "Normalizando campos de texto...")
    text_columns = ["road", "via", "expressway", "region"]
    for column in text_columns:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].apply(normalize_text)
    print_success("Normalizacao concluida")

    print_step(3, 7, "Normalizando regioes...")
    if "region" in dataframe.columns:
        dataframe["region"] = dataframe["region"].apply(normalize_region)
        invalid_regions = dataframe[
            ~dataframe["region"].isin(REGIONS)
        ]["region"].unique()
        if len(invalid_regions) > 0:
            print_warning(f"Encontradas {len(invalid_regions)} regioes invalidas")
            dataframe = dataframe[dataframe["region"].isin(REGIONS)]
            print_success(f"Regioes invalidas removidas")
        else:
            print_success("Todas as regioes sao validas")

    print_step(4, 7, "Validando campos de data e hora...")
    initial_count = len(dataframe)
    valid_datetime_mask = dataframe.apply(
        lambda row: validate_datetime_fields(
            str(row["day"]),
            str(row["hour"])
        )[0],
        axis=1
    )
    dataframe = dataframe[valid_datetime_mask]
    removed_count = initial_count - len(dataframe)
    if removed_count > 0:
        print_warning(f"Removidos {removed_count:,} registros com data/hora invalida")
    else:
        print_success("Todos os registros possuem data/hora validas")

    print_step(5, 7, "Validando tamanhos de congestionamento...")
    initial_count = len(dataframe)
    valid_size_mask = dataframe["congestion_size"].apply(validate_congestion_size)
    dataframe = dataframe[valid_size_mask]
    removed_count = initial_count - len(dataframe)
    if removed_count > 0:
        print_warning(f"Removidos {removed_count:,} registros com tamanho invalido")
    else:
        print_success("Todos os tamanhos de congestionamento sao validos")

    print_step(6, 7, "Removendo registros duplicados...")
    initial_count = len(dataframe)
    dataframe = dataframe.drop_duplicates()
    removed_count = initial_count - len(dataframe)
    if removed_count > 0:
        print_success("Duplicatas removidas", f"{removed_count:,}")
    else:
        print_success("Nenhuma duplicata encontrada")

    print_step(7, 7, "Salvando arquivo processado...")
    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(
            OUTPUT_FILE,
            sep=OUTPUT_CSV_SEPARATOR,
            index=False,
            encoding=OUTPUT_ENCODING
        )
        print_success("Arquivo salvo", str(OUTPUT_FILE))
        print_success("Total de registros", f"{len(dataframe):,}")
    except Exception as error:
        print_warning(f"Erro ao salvar arquivo: {error}")
        return

    print_header("PROCESSAMENTO CONCLUIDO COM SUCESSO!")

    generate_statistical_report(dataframe)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()