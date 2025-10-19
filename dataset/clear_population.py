#!/usr/bin/env python3
"""
    Script de processamento de dados populacionais para os distritos de São Paulo.

    Este script realiza as seguintes operações:
    - Carrega dados de estimativa populacional por idade, sexo e distrito
    - Normaliza textos (remove acentos e espaços extras)
    - Filtra registros com idade >= 20 anos
    - Mapeia distritos para regiões geográficas
    - Gera relatório estatístico abrangente
    - Exporta os dados processados para CSV

    Autor: Alexandre Marques Tortoza Canoa
    Versão do Python: 3.13.7
"""

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

INPUT_FILE: Path = Path("./rawData/estimativa_pop_idade_sexo_msp.csv")
OUTPUT_FILE: Path = Path("./clean/clean_population.csv")
UNMAPPED_FILE: Path = Path("zDistritos_nao_mapeados.json")
CSV_SEPARATOR: str = ";"
INPUT_ENCODING: str = "iso-8859-1"
OUTPUT_ENCODING: str = "utf-8"
MINIMUM_AGE: int = 20

REGIONS: List[str] = ["Center", "North", "South", "East", "West"]


def remove_accents(text: str) -> str:
    """
        Remover acentos de strings usando normalização Unicode.

        Esta função converte caracteres acentuados para seus equivalentes ASCII.
        Por exemplo: 'São Paulo' -> 'Sao Paulo'

        Args:
            text: String que pode conter caracteres acentuados.

        Returns:
            String sem acentos, ou o valor original se não for uma string.

        Examples:
            >>> remove_accents("São Paulo")
            'Sao Paulo'
            >>> remove_accents("Água Rasa")
            'Agua Rasa'
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

        Exemplos:
            >>> normalize_text("São  Paulo  ")
            'Sao Paulo'
    """
    if isinstance(text, str):
        text = remove_accents(text)
        text = " ".join(text.split())
        return text.strip()
    return text


def validate_age_range(age_range: str) -> bool:
    """
        Verifica se o intervalo de idade inicia em 20 anos ou mais.

        Args:
            age_range: String no formato "XX a YY" ou "XX e +".

        Returns:
            True se a idade inicial for >= 20, False caso contrário.

        Exemplos:
            >>> validate_age_range("20 a 24")
            True
            >>> validate_age_range("15 a 19")
            False
            >>> validate_age_range("75 e +")
            True
    """
    try:
        age_initial = int(age_range.split()[0])
        return age_initial >= MINIMUM_AGE
    except (ValueError, IndexError, AttributeError):
        return False


def get_district_region_mapping() -> Dict[str, str]:
    """
        Obter o mapeamento completo de distritos para as regiões geográficas.

        Todos os nomes de distritos devem estar normalizados (sem acentos e com
        apenas espaços simples). As regiões são em inglês: Center, North, South,
        East, West.

        Retorna:
            Dicionário que mapeia nomes de distritos normalizados para nomes de
            regiões.

        Observação:
            Este mapeamento abrange os 96 distritos oficiais de São Paulo.
    """
    return {
        # Center Region
        "Bela Vista": "center",
        "Bom Retiro": "center",
        "Bras": "center",
        "Cambuci": "center",
        "Consolacao": "center",
        "Liberdade": "center",
        "Republica": "center",
        "Santa Cecilia": "center",
        "Se": "center",

        # North Region
        "Anhanguera": "north",
        "Brasilandia": "north",
        "Cachoeirinha": "north",
        "Casa Verde": "north",
        "Freguesia do O": "north",
        "Jacana": "north",
        "Jaragua": "north",
        "Limao": "north",
        "Mandaqui": "north",
        "Perus": "north",
        "Pirituba": "north",
        "Santana": "north",
        "Sao Domingos": "north",
        "Tremembe": "north",
        "Tucuruvi": "north",
        "Vila Guilherme": "north",
        "Vila Maria": "north",
        "Vila Medeiros": "north",

        # South Region
        "Campo Limpo": "south",
        "Capao Redondo": "south",
        "Cidade Ademar": "south",
        "Cidade Dutra": "south",
        "Campo Belo": "south",
        "Campo Grande": "south",
        "Cursino": "south",
        "Grajau": "south",
        "Ipiranga": "south",
        "Jabaquara": "south",
        "Jardim Angela": "south",
        "Jardim Sao Luis": "south",
        "Marsilac": "south",
        "Moema": "south",
        "Parelheiros": "south",
        "Pedreira": "south",
        "Santo Amaro": "south",
        "Sacoma": "south",
        "Saude": "south",
        "Socorro": "south",
        "Vila Andrade": "south",
        "Vila Mariana": "south",

        # East Region
        "Agua Rasa": "east",
        "Aricanduva": "east",
        "Artur Alvim": "east",
        "Belem": "east",
        "Cangaiba": "east",
        "Carrao": "east",
        "Cidade Lider": "east",
        "Cidade Tiradentes": "east",
        "Ermelino Matarazzo": "east",
        "Guaianases": "east",
        "Iguatemi": "east",
        "Itaim Paulista": "east",
        "Itaquera": "east",
        "Jardim Helena": "east",
        "Jose Bonifacio": "east",
        "Lajeado": "east",
        "Mooca": "east",
        "Pari": "east",
        "Parque do Carmo": "east",
        "Penha": "east",
        "Ponte Rasa": "east",
        "Sao Lucas": "east",
        "Sao Mateus": "east",
        "Sao Miguel": "east",
        "Sao Rafael": "east",
        "Sapopemba": "east",
        "Tatuape": "east",
        "Vila Curuca": "east",
        "Vila Formosa": "east",
        "Vila Jacui": "east",
        "Vila Matilde": "east",
        "Vila Prudente": "east",

        # West Region
        "Alto de Pinheiros": "west",
        "Barra Funda": "west",
        "Butanta": "west",
        "Itaim Bibi": "west",
        "Jardim Paulista": "west",
        "Jaguara": "west",
        "Jaguare": "west",
        "Lapa": "west",
        "Morumbi": "west",
        "Perdizes": "west",
        "Pinheiros": "west",
        "Raposo Tavares": "west",
        "Rio Pequeno": "west",
        "Vila Leopoldina": "west",
        "Vila Sonia": "west"
    }


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
            >>> create_ascii_bar_chart({"A": 100, "B": 50}, 20)
            ['   A       : ███████████████████ 100 (66.7%)',
            '   B       : ██████████           50 (33.3%)']
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
        print(f"   \uf05d{message}: {value}")
    else:
        print(f"   \uf05d{message}")


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


def save_unmapped_districts(
    districts: List[str],
    filename: Path = UNMAPPED_FILE
) -> None:
    """
        Salva uma lista dos distritos não mapeados em um arquivo JSON.

        Args:
            districts: Lista de nomes de distritos não mapeados.
            filename: Caminho do arquivo JSON de saída.
    """
    data = {
        "unmapped_districts": sorted(districts),
        "count": len(districts)
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_statistical_report(dataframe: pd.DataFrame) -> None:
    """
        Gera e imprime um relatório estatístico abrangente.

        Args:
            dataframe: DataFrame pandas processado contendo, no mínimo, a coluna
                    'region' (região) e outras colunas relevantes como 'distritos',
                    'sexo' e 'ano'.
    """
    print_header("RESUMO ESTATÍSTICO DOS DADOS PROCESSADOS")

    total_records = len(dataframe)
    print(f"\n\uf013 Total de registros: {total_records:,}")
    
    mapped_records = dataframe["region"].notna().sum()
    unmapped_records = dataframe["region"].isna().sum()

    print(f"\n🗺️  Status de mapeamento de regiões:")
    print(f"   \uf05dMapeado: {mapped_records:,} "
          f"({mapped_records/total_records*100:.1f}%)")

    if unmapped_records > 0:
        print(f"   ✗ Não mapeado: {unmapped_records:,} "
              f"({unmapped_records/total_records*100:.1f}%)")

    if mapped_records > 0:
        region_counts = (dataframe[dataframe["region"].notna()]["region"]
                        .value_counts()
                        .sort_index()
                        .to_dict())

        print(f"\n📍 Distribuição por região:")
        bar_chart = create_ascii_bar_chart(region_counts)
        for line in bar_chart:
            print(line)

    unique_districts = dataframe["distritos"].nunique()
    print(f"\n🏘️  Distritos únicos no conjunto de dados: {unique_districts}")

    if "ano" in dataframe.columns:
        min_year = dataframe["ano"].min()
        max_year = dataframe["ano"].max()
        print(f"\n📅 Intervalo de anos: {min_year} até {max_year}")

    if "sexo" in dataframe.columns:
        sex_counts = dataframe["sexo"].value_counts().to_dict()
        print(f"\n👥 Distribuição por sexo:")
        for sex, count in sorted(sex_counts.items()):
            percentage = count / total_records * 100
            print(f"   {sex:10s}: {count:>8,d} ({percentage:5.1f}%)")


def validate_mapping_coverage(
    dataframe: pd.DataFrame,
    distrito_regiao: Dict[str, str]
) -> Tuple[List[str], int]:
    """
        Valida que todos os distritos no dataset estejam mapeados para regiões.

        Args:
            dataframe: DataFrame contendo dados dos distritos.
            distrito_regiao: Dicionário que mapeia distritos para regiões.

        Returns:
            Tupla contendo (lista de distritos não mapeados, contador de distritos mapeados).
    """
    unique_districts = dataframe["distritos"].unique()
    unmapped = []

    for district in unique_districts:
        if district not in distrito_regiao:
            unmapped.append(district)

    mapped_count = len(unique_districts) - len(unmapped)
    return unmapped, mapped_count


def main() -> None:
    """
        Função principal para processar o conjunto de dados populacionais.

        Orquestra o pipeline completo:
        1. Carregar dados do CSV
        2. Normalizar campos de texto
        3. Filtrar por faixa etária (>= MINIMUM_AGE)
        4. Adicionar coluna 'region' com mapeamento de distritos
        5. Validar cobertura do mapeamento e salvar distritos não mapeados
        6. Salvar dados processados em CSV
        7. Gerar relatório estatístico resumido
    """
    print_header("POPULATION DATA PROCESSING - SAO PAULO")
    print(f"Arquivo de entrada: {INPUT_FILE}")

    distrito_regiao = get_district_region_mapping()

    print_step(1, 6, "Loading data...")
    try:
        dataframe = pd.read_csv(
            INPUT_FILE,
            sep=CSV_SEPARATOR,
            encoding=INPUT_ENCODING
        )
        print_success("Registros carregados", f"{len(dataframe):,}")
    except FileNotFoundError:
        print_warning(f"Arquivo não encontrado: {INPUT_FILE}")
        return
    except Exception as e:
        print_warning(f"Erro ao carregar arquivo: {e}")
        return

    print_step(2, 6, "Normalizando campos de texto...")
    string_columns = dataframe.select_dtypes(include=["object"]).columns
    for column in string_columns:
        dataframe[column] = dataframe[column].apply(normalize_text)
    print_success("Normalização concluída")

    print_step(3, 6, f"Filtrando registros com idade >= {MINIMUM_AGE} anos...")
    initial_count = len(dataframe)
    dataframe = dataframe[dataframe["idade"].apply(validate_age_range)]
    removed_count = initial_count - len(dataframe)
    print_success("Registros removidos", f"{removed_count:,}")
    print_success("Registros mantidos", f"{len(dataframe):,}")

    print_step(4, 6, "Verificando mapeamentos manuais...")
    if UNMAPPED_FILE.exists():
        with open(UNMAPPED_FILE, "r", encoding="utf-8") as f:
            manual_mappings = json.load(f)
        print_info(f"Encontrados {len(manual_mappings)} mapeamentos manuais")
    else:
        print_info("Nenhum mapeamento manual encontrado")

    print_step(5, 6, "Adicionando coluna de região...")
    dataframe["region"] = dataframe["distritos"].map(distrito_regiao)

    unmapped_districts, mapped_count = validate_mapping_coverage(
        dataframe,
        distrito_regiao
    )

    print_success("Registros mapeados", f"{len(dataframe):,}")

    if unmapped_districts:
        print_warning(
            f"Encontrados {len(unmapped_districts)} distritos não mapeados:"
        )
        for district in sorted(unmapped_districts):
            print(f"      - {district}")
        save_unmapped_districts(unmapped_districts)
        print_info(f"Distritos não mapeados salvos em: {UNMAPPED_FILE}")
    else:
        print_success("Todos os distritos foram mapeados com sucesso!")

    print_step(6, 6, "Salvando arquivo processado...")
    try:
        dataframe.to_csv(
            OUTPUT_FILE,
            sep=CSV_SEPARATOR,
            index=False,
            encoding=OUTPUT_ENCODING
        )
        print_success("Arquivo salvo", str(OUTPUT_FILE))
        print_success("Total de registros", f"{len(dataframe):,}")
    except Exception as e:
        print_warning(f"Erro ao salvar arquivo: {e}")
        return

    print_header("\uf05dPROCESSAMENTO CONCLUÍDO COM SUCESSO!")

    generate_statistical_report(dataframe)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
