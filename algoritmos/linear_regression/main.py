#!/usr/bin/env python3
"""
    Previsão de Congestionamento de Trânsito usando Linear Regression
    
    Autor: Alexandre Marques Tortoza Canoa
    Versão do Python: 3.13.7
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import json
from datetime import datetime

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("\uea6c  Para usar SHAP: pip install shap")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES GLOBAIS - ALTERE CONFORME NECESSÁRIO
# ============================================================================

class ConfigLinearRegression:
    """Centraliza todas as configurações do experimento."""
    
    DATASET_PATH = "../../dataset/preparedData/dataset.csv"
    DELIMITER = ";"
    
    # Estratégia de Validação
    # Opções: 'holdout' ou 'kfold'
    VALIDATION_STRATEGY = 'holdout'
    HOLDOUT_TEST_SIZE = 0.2  # Proporção teste (0.2 = 80/20)
    KFOLD_N_SPLITS = 5  # Número de folds para validação cruzada
    RANDOM_STATE = 42
    
    # Hiperparâmetros da Linear Regression
    FIT_INTERCEPT = True  # Incluir intercepto (bias)
    NORMALIZE = False  # Deprecated, usar StandardScaler ao invés
    N_JOBS = -1  # Usar todos os cores disponíveis
    
    # Features e Target
    FEATURES = [
        'pop_total',            # População total (principal feature)
        'hora_numeric',         # Hora do dia (0-23)
        'via_expressa_encoded', # Via expressa (E, N, S, W)
        'regiao_encoded',       # Região (east, center, etc)
        'sexo_encoded',         # Sexo (Homens, Mulheres)
        'dia_semana',           # Dia da semana (0-6)
        'mes',                  # Mês (1-12)
    ]
    TARGET = 'tamanho_congestionamento'
    
    # SHAP
    SHAP_ENABLED = True
    SHAP_N_SAMPLES = 100  # Amostras para explicabilidade
    
    # Visualizações
    PLOT_RESULTS = True
    PLOT_SHAP = True
    PLOT_COEFFICIENTS = True  # Visualizar coeficientes da regressão
    PLOT_FEATURE_IMPORTANCE = True  # Importância baseada em coeficientes
    
    # Análise Estatística
    CALCULATE_STATISTICS = True  # P-values, confidence intervals, etc
    
    # Logs
    VERBOSE = True
    SAVE_RESULTS = True
    RESULTS_FILE = f"resultados_linear_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


# ============================================================================
# FUNÇÕES DE CARREGAMENTO E PREPARAÇÃO DE DADOS
# ============================================================================

def load_data(filepath: str, delimiter: str = ";") -> pd.DataFrame:
    """
        Carrega o dataset do arquivo CSV.
        
        Args:
            filepath: Caminho do arquivo
            delimiter: Delimitador do CSV
            
        Returns:
            DataFrame com os dados carregados
    """
    try:
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                dataframe = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
                if ConfigLinearRegression.VERBOSE:
                    print(f"\uf05d Dataset carregado: {dataframe.shape}")
                    print(f"  Encoding: {encoding}")
                    print(f"  Colunas: {list(dataframe.columns)}\n")
                return dataframe
            except UnicodeDecodeError:
                continue
    except FileNotFoundError:
        print(f"\uea87 Arquivo não encontrado: {filepath}")
        raise
    
    print(f"\uea87 Não foi possível carregar o arquivo com nenhum encoding")
    raise ValueError("Erro ao carregar arquivo")


def extract_datetime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
        Extrai features temporais das colunas 'data' e 'hora'.
        
        Args:
            dataframe: DataFrame com colunas 'data' e 'hora'
            
        Returns:
            DataFrame com novas colunas de features temporais
    """
    if ConfigLinearRegression.VERBOSE:
        print("\uf017 Extraindo features temporais...")
    
    dataframe['datetime'] = pd.to_datetime(
        dataframe['data'] + ' ' + dataframe['hora'],
        format='%Y-%m-%d %H:%M:%S'
    )
    
    dataframe['hora_numeric'] = dataframe['datetime'].dt.hour
    dataframe['dia_semana'] = dataframe['datetime'].dt.dayofweek  # 0=segunda, 6=domingo
    dataframe['mes'] = dataframe['datetime'].dt.month
    dataframe['dia_mes'] = dataframe['datetime'].dt.day
    
    if ConfigLinearRegression.VERBOSE:
        print("  \uf05d Features: hora_numeric, dia_semana, mes, dia_mes\n")
    
    return dataframe


def encode_categorical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
        Codifica features categóricas em numéricas.
        
        Args:
            dataframe: DataFrame com features categóricas
            
        Returns:
            DataFrame com features codificadas
    """
    if ConfigLinearRegression.VERBOSE:
        print("\uf15c Codificando features categóricas...")
    
    encoders = {}
    categorical_cols = ['via_expressa', 'regiao', 'sexo']
    
    for column in categorical_cols:
        if column in dataframe.columns:
            label_encoder = LabelEncoder()
            dataframe[f'{column}_encoded'] = label_encoder.fit_transform(
                dataframe[column].fillna('Unknown')
            )
            encoders[column] = label_encoder
            
            if ConfigLinearRegression.VERBOSE:
                classes_dict = dict(zip(
                    label_encoder.classes_, 
                    label_encoder.transform(label_encoder.classes_)
                ))
                print(f"  \uf05d {column}: {classes_dict}")
    
    if ConfigLinearRegression.VERBOSE:
        print()
    
    return dataframe


def prepare_data(filepath: str) -> pd.DataFrame:
    """
        Pipeline completo de preparação de dados.
        
        Args:
            filepath: Caminho do dataset
            
        Returns:
            DataFrame preparado
    """
    print("="*80)
    print("  \ueb03 CARREGAMENTO E PREPARAÇÃO DE DADOS")
    print("="*80)
    
    dataframe = load_data(filepath, delimiter=ConfigLinearRegression.DELIMITER)
    
    print(f"\ueb2b Verificando valores ausentes iniciais:")
    missing = dataframe.isnull().sum()
    if missing.sum() > 0:
        print(f"  {missing[missing > 0].to_dict()}\n")
    else:
        print("  \uf05d Nenhum valor ausente\n")
    
    dataframe = extract_datetime_features(dataframe)
    
    dataframe = encode_categorical_features(dataframe)
    
    print("\uebab Agregando dados por data/hora/via/região...")
    dataframe_aggregated = dataframe.groupby(
        ['data', 'hora', 'via_expressa', 'regiao']
    ).agg({
        'pop_total': 'sum', 
        'tamanho_congestionamento': 'first',  
        'hora_numeric': 'first',
        'dia_semana': 'first',
        'mes': 'first',
        'via_expressa_encoded': 'first',
        'regiao_encoded': 'first',
        'sexo_encoded': 'mean',
    }).reset_index()
    
    print(f"  \uf05d {len(dataframe)} → {len(dataframe_aggregated)} linhas\n")
    
    print(f"\ueb2b Valores ausentes após preparação:")
    missing = dataframe_aggregated.isnull().sum()
    if missing.sum() > 0:
        print(f"  {missing[missing > 0].to_dict()}\n")
    else:
        print("  \uf05d Nenhum valor ausente\n")
    
    print(f"\ueb03 Estatísticas do target (tamanho_congestionamento):")
    print(f"  Mínimo: {dataframe_aggregated['tamanho_congestionamento'].min()}")
    print(f"  Máximo: {dataframe_aggregated['tamanho_congestionamento'].max()}")
    print(f"  Média: {dataframe_aggregated['tamanho_congestionamento'].mean():.2f}")
    print(f"  Mediana: {dataframe_aggregated['tamanho_congestionamento'].median():.2f}\n")
    
    return dataframe_aggregated


# ============================================================================
# FUNÇÕES DE ANÁLISE ESTATÍSTICA
# ============================================================================

def calculate_statistics(
    model: LinearRegression,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list
) -> dict:
    """
        Calcula estatísticas detalhadas do modelo linear.
        
        Args:
            model: Modelo treinado
            X: Features
            y: Target
            feature_names: Nomes das features
            
        Returns:
            Dicionário com estatísticas
    """
    if not ConfigLinearRegression.CALCULATE_STATISTICS:
        return {}
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Predições
    y_pred = model.predict(X)
    
    # Resíduos
    residuals = y - y_pred
    
    # MSE e variância dos resíduos
    mse = mean_squared_error(y, y_pred)
    residual_variance = np.var(residuals)
    
    # Erro padrão dos coeficientes (simplificado)
    # Para cálculo completo precisaria de statsmodels
    try:
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        variance_matrix = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * mse
        standard_errors = np.sqrt(np.diag(variance_matrix))
        
        # T-statistics (simplificado)
        coefficients_with_intercept = np.concatenate([[model.intercept_], model.coef_])
        t_stats = coefficients_with_intercept / standard_errors
        
        # P-values (aproximado usando distribuição normal)
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        stats_dict = {
            'coefficients': dict(zip(['intercept'] + feature_names, coefficients_with_intercept)),
            'standard_errors': dict(zip(['intercept'] + feature_names, standard_errors)),
            't_statistics': dict(zip(['intercept'] + feature_names, t_stats)),
            'p_values': dict(zip(['intercept'] + feature_names, p_values)),
            'residual_variance': float(residual_variance),
            'mse': float(mse),
        }
        
        return stats_dict
        
    except Exception as error:
        print(f"\uea87 Erro ao calcular estatísticas: {error}")
        return {
            'coefficients': dict(zip(['intercept'] + feature_names, 
                                   np.concatenate([[model.intercept_], model.coef_]))),
        }


# ============================================================================
# FUNÇÕES DE TREINAMENTO
# ============================================================================

def train_linear_holdout(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    feature_names: list
) -> dict:
    """
        Treina Linear Regression com validação holdout (80/20).
        
        Args:
            X_train, X_test: Features de treino/teste
            y_train, y_test: Target de treino/teste
            feature_names: Nomes das features
            
        Returns:
            Dicionário com modelo, métricas e dados
    """
    print(f" Treinando Linear Regression...")
    
    model = LinearRegression(
        fit_intercept=ConfigLinearRegression.FIT_INTERCEPT,
        n_jobs=ConfigLinearRegression.N_JOBS
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'intercept': float(model.intercept_),
        'coefficients': dict(zip(feature_names, model.coef_)),
    }
    
    # Calcular estatísticas detalhadas
    statistics = calculate_statistics(model, X_train, y_train, feature_names)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'metrics': metrics,
        'statistics': statistics,
    }


def train_linear_kfold(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """
        Treina Linear Regression com validação cruzada K-Fold.
        
        Args:
            X: Features
            y: Target
            feature_names: Nomes das features
            
        Returns:
            Dicionário com scores e estatísticas
    """
    print(f" Treinando Linear Regression com {ConfigLinearRegression.KFOLD_N_SPLITS}-Fold CV...")
    
    model = LinearRegression(
        fit_intercept=ConfigLinearRegression.FIT_INTERCEPT,
        n_jobs=ConfigLinearRegression.N_JOBS
    )
    
    kfold = KFold(
        n_splits=ConfigLinearRegression.KFOLD_N_SPLITS, 
        shuffle=True, 
        random_state=ConfigLinearRegression.RANDOM_STATE
    )
    
    scores_r2 = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    scores_mae = cross_val_score(
        model, X, y, cv=kfold, 
        scoring='neg_mean_absolute_error'
    )
    
    # Treinar modelo final com todos os dados para análise posterior
    model.fit(X, y)
    
    # Calcular estatísticas detalhadas
    statistics = calculate_statistics(model, X, y, feature_names)
    
    return {
        'model': model,
        'X': X,
        'y': y,
        'cv_r2_scores': scores_r2,
        'cv_mae_scores': -scores_mae,
        'cv_r2_mean': scores_r2.mean(),
        'cv_r2_std': scores_r2.std(),
        'cv_mae_mean': -scores_mae.mean(),
        'cv_mae_std': scores_mae.std(),
        'statistics': statistics,
    }


def train_model(dataframe: pd.DataFrame) -> dict:
    """
        Pipeline de treinamento adaptado à estratégia de validação.
        
        Args:
            dataframe: DataFrame preparado
            
        Returns:
            Resultados do treinamento
    """
    print("="*80)
    print("   TREINAMENTO DO MODELO")
    print("="*80)
    
    missing_features = [
        feature for feature in ConfigLinearRegression.FEATURES 
        if feature not in dataframe.columns
    ]
    if missing_features:
        print(f"\uea87 Features não encontradas: {missing_features}")
        print(f"   Colunas disponíveis: {list(dataframe.columns)}")
        raise ValueError("Features faltando no dataset")
    
    dataframe_clean = dataframe.dropna(
        subset=ConfigLinearRegression.FEATURES + [ConfigLinearRegression.TARGET]
    )
    print(f"\uf05d Dados limpos: {len(dataframe)} → {len(dataframe_clean)} linhas\n")
    
    X = dataframe_clean[ConfigLinearRegression.FEATURES].values
    y = dataframe_clean[ConfigLinearRegression.TARGET].values
    
    # Normalização (opcional para Linear Regression, mas melhora interpretação)
    print(" Normalizando features (opcional para Linear Regression)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  \uf05d Média: {X_scaled.mean(axis=0)}")
    print(f"  \uf05d Std: {X_scaled.std(axis=0)}\n")
    
    if ConfigLinearRegression.VALIDATION_STRATEGY == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=ConfigLinearRegression.HOLDOUT_TEST_SIZE,
            random_state=ConfigLinearRegression.RANDOM_STATE
        )
        print(f"\uf05d Split: {len(X_train)} treino | {len(X_test)} teste\n")
        
        results = train_linear_holdout(
            X_train, X_test, y_train, y_test, 
            ConfigLinearRegression.FEATURES
        )
        
    elif ConfigLinearRegression.VALIDATION_STRATEGY == 'kfold':
        results = train_linear_kfold(X_scaled, y, ConfigLinearRegression.FEATURES)
    
    else:
        raise ValueError(
            f"Estratégia desconhecida: {ConfigLinearRegression.VALIDATION_STRATEGY}"
        )
    
    results['scaler'] = scaler
    results['features'] = ConfigLinearRegression.FEATURES
    results['dataframe'] = dataframe_clean
    
    return results


def print_metrics(results: dict) -> None:
    """Imprime métricas de forma formatada."""
    print("\n" + "="*80)
    print("  \ueb03 MÉTRICAS DO MODELO")
    print("="*80)
    
    if ConfigLinearRegression.VALIDATION_STRATEGY == 'holdout':
        metrics = results['metrics']
        print(f"\n{'Métrica':<25} {'Treino':>12} {'Teste':>12}")
        print("-" * 50)
        print(f"{'MAE':<25} {metrics['train_mae']:>12.4f} {metrics['test_mae']:>12.4f}")
        print(f"{'RMSE':<25} {metrics['train_rmse']:>12.4f} {metrics['test_rmse']:>12.4f}")
        print(f"{'R² Score':<25} {metrics['train_r2']:>12.4f} {metrics['test_r2']:>12.4f}")
        
        print(f"\n{'Intercepto':<25} {metrics['intercept']:>12.4f}")
        
    elif ConfigLinearRegression.VALIDATION_STRATEGY == 'kfold':
        print(f"\n{ConfigLinearRegression.KFOLD_N_SPLITS}-Fold Cross Validation:")
        print("-" * 50)
        print(f"R² Scores: {results['cv_r2_scores']}")
        print(f"R² Média: {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")
        print(f"MAE Média: {results['cv_mae_mean']:.4f} (+/- {results['cv_mae_std']:.4f})")
    
    # Imprimir estatísticas se disponíveis
    if results.get('statistics') and ConfigLinearRegression.CALCULATE_STATISTICS:
        print("\n" + "="*80)
        print("  ANÁLISE ESTATÍSTICA DOS COEFICIENTES")
        print("="*80)
        
        stats = results['statistics']
        
        if 'p_values' in stats:
            print(f"\n{'Feature':<25} {'Coef':>12} {'Std Err':>12} {'t-stat':>10} {'p-value':>10}")
            print("-" * 70)
            
            for feature in ['intercept'] + results['features']:
                coef = stats['coefficients'][feature]
                std_err = stats['standard_errors'][feature]
                t_stat = stats['t_statistics'][feature]
                p_val = stats['p_values'][feature]
                
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                
                print(f"{feature:<25} {coef:>12.4f} {std_err:>12.4f} {t_stat:>10.4f} {p_val:>10.4f} {significance}")
            
            print("\nSignificância: *** p<0.001, ** p<0.01, * p<0.05")
    
    print()


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================================

def plot_results_holdout(results: dict) -> None:
    """Plota resultados para validação holdout."""
    if not ConfigLinearRegression.PLOT_RESULTS:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Real vs Previsto (Teste)
    ax = axes[0, 0]
    ax.scatter(
        results['y_test'], 
        results['y_test_pred'], 
        alpha=0.5, 
        s=20, 
        edgecolors='k', 
        linewidth=0.5
    )
    min_val = min(results['y_test'].min(), results['y_test_pred'].min())
    max_val = max(results['y_test'].max(), results['y_test_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Previstos')
    ax.set_title('Teste: Real vs Previsto')
    ax.grid(True, alpha=0.3)
    textstr = (
        f"R² = {results['metrics']['test_r2']:.4f}\n"
        f"MAE = {results['metrics']['test_mae']:.2f}"
    )
    ax.text(
        0.05, 0.95, textstr, 
        transform=ax.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Real vs Previsto (Treino)
    ax = axes[0, 1]
    ax.scatter(
        results['y_train'], 
        results['y_train_pred'], 
        alpha=0.5, 
        s=20, 
        edgecolors='k', 
        linewidth=0.5
    )
    min_val = min(results['y_train'].min(), results['y_train_pred'].min())
    max_val = max(results['y_train'].max(), results['y_train_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Previstos')
    ax.set_title('Treino: Real vs Previsto')
    ax.grid(True, alpha=0.3)
    textstr = (
        f"R² = {results['metrics']['train_r2']:.4f}\n"
        f"MAE = {results['metrics']['train_mae']:.2f}"
    )
    ax.text(
        0.05, 0.95, textstr, 
        transform=ax.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )
    
    # Distribuição de Erros
    ax = axes[1, 0]
    errors = results['y_test'] - results['y_test_pred']
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Erro de Previsão')
    ax.set_ylabel('Frequência')
    ax.set_title(f'Distribuição de Erros (Teste) - Média: {errors.mean():.2f}')
    ax.grid(True, alpha=0.3)
    
    # Análise de Resíduos
    ax = axes[1, 1]
    residuals = results['y_test'] - results['y_test_pred']
    ax.scatter(
        results['y_test_pred'], 
        residuals, 
        alpha=0.5, 
        s=20, 
        edgecolors='k', 
        linewidth=0.5
    )
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Valores Previstos')
    ax.set_ylabel('Resíduos')
    ax.set_title('Análise de Resíduos (Teste)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_results_holdout.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: linear_results_holdout.png")
    plt.show()


def plot_results_kfold(results: dict) -> None:
    """Plota resultados para validação K-Fold."""
    if not ConfigLinearRegression.PLOT_RESULTS:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² por Fold
    ax = axes[0]
    folds = np.arange(1, len(results['cv_r2_scores']) + 1)
    ax.bar(
        folds, 
        results['cv_r2_scores'], 
        alpha=0.7, 
        color='skyblue', 
        edgecolor='black'
    )
    ax.axhline(
        y=results['cv_r2_mean'], 
        color='r', 
        linestyle='--', 
        linewidth=2, 
        label='Média'
    )
    ax.fill_between(
        folds, 
        results['cv_r2_mean'] - results['cv_r2_std'],
        results['cv_r2_mean'] + results['cv_r2_std'],
        alpha=0.2, 
        color='r'
    )
    ax.set_xlabel('Fold')
    ax.set_ylabel('R² Score')
    ax.set_title('Scores R² por Fold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # MAE por Fold
    ax = axes[1]
    ax.bar(
        folds, 
        results['cv_mae_scores'], 
        alpha=0.7, 
        color='lightcoral', 
        edgecolor='black'
    )
    ax.axhline(
        y=results['cv_mae_mean'], 
        color='r', 
        linestyle='--', 
        linewidth=2, 
        label='Média'
    )
    ax.fill_between(
        folds,
        results['cv_mae_mean'] - results['cv_mae_std'],
        results['cv_mae_mean'] + results['cv_mae_std'],
        alpha=0.2, 
        color='r'
    )
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE')
    ax.set_title('Scores MAE por Fold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('linear_results_kfold.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: linear_results_kfold.png")
    plt.show()


def plot_coefficients(results: dict) -> None:
    """Plota os coeficientes da regressão linear."""
    if not ConfigLinearRegression.PLOT_COEFFICIENTS:
        return
    
    print("\n" + "="*80)
    print("   COEFICIENTES DA REGRESSÃO LINEAR")
    print("="*80)
    
    model = results['model']
    features = results['features']
    
    # Criar DataFrame com coeficientes
    coef_dataframe = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico de barras dos coeficientes
    ax = axes[0]
    colors = ['green' if coef > 0 else 'red' for coef in coef_dataframe['Coefficient']]
    ax.barh(coef_dataframe['Feature'], coef_dataframe['Coefficient'], 
            color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Coeficiente')
    ax.set_title('Coeficientes da Regressão Linear')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Gráfico de importância absoluta
    ax = axes[1]
    abs_coef = coef_dataframe.copy()
    abs_coef['Coefficient'] = abs_coef['Coefficient'].abs()
    abs_coef = abs_coef.sort_values('Coefficient', ascending=True)
    
    ax.barh(abs_coef['Feature'], abs_coef['Coefficient'], 
            alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('|Coeficiente|')
    ax.set_title('Importância Absoluta das Features')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('linear_coefficients.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: linear_coefficients.png\n")
    plt.show()
    
    # Imprimir tabela de coeficientes
    print("\nRanking de Coeficientes (ordem decrescente de impacto absoluto):")
    print("-" * 50)
    for index, row in coef_dataframe.iterrows():
        direction = "+" if row['Coefficient'] > 0 else "-"
        print(f"{row['Feature']:<25} {direction} {abs(row['Coefficient']):>10.4f}")


# ============================================================================
# FUNÇÕES SHAP
# ============================================================================

def explain_with_shap(results: dict) -> None:
    """Análise de explicabilidade com SHAP."""
    if not ConfigLinearRegression.SHAP_ENABLED or not HAS_SHAP:
        if ConfigLinearRegression.VERBOSE:
            print("\uea6c  SHAP desabilitado ou não disponível")
        return
    
    if ConfigLinearRegression.VALIDATION_STRATEGY != 'holdout':
        print("\uea6c  SHAP disponível apenas para validação holdout")
        return
    
    print("\n" + "="*80)
    print("   ANÁLISE SHAP")
    print("="*80)
    print(f"Executando SHAP com LinearExplainer (nativo para regressão linear)...")
    
    model = results['model']
    X_sample = results['X_test'][:ConfigLinearRegression.SHAP_N_SAMPLES]
    
    # LinearExplainer é exato e muito rápido para modelos lineares
    explainer = shap.LinearExplainer(model, results['X_train'])
    shap_values = explainer.shap_values(X_sample)
    
    print("\uf05d SHAP values calculados (exatos para modelo linear)\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Summary Plot (beeswarm)
    plt.sca(axes[0, 0])
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=results['features'], 
        show=False
    )
    axes[0, 0].set_title('Importância Global (SHAP)', fontweight='bold', fontsize=12)
    
    # Bar Plot (média absoluta)
    plt.sca(axes[0, 1])
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=results['features'], 
        plot_type="bar", 
        show=False
    )
    axes[0, 1].set_title('Importância Média Absoluta', fontweight='bold', fontsize=12)
    
    # Dependence Plot (feature mais importante)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_index = np.argmax(mean_abs_shap)
    
    plt.sca(axes[1, 0])
    shap.dependence_plot(
        top_index, 
        shap_values, 
        X_sample,
        feature_names=results['features'], 
        ax=axes[1, 0], 
        show=False
    )
    
    # Tabela de Importâncias
    axes[1, 1].axis('off')
    importance_dataframe = pd.DataFrame({
        'Feature': results['features'],
        'SHAP Mean |Impact|': mean_abs_shap
    }).sort_values('SHAP Mean |Impact|', ascending=False)
    
    table_text = "RANKING SHAP\n" + "="*45 + "\n\n"
    for index, row in importance_dataframe.iterrows():
        table_text += (
            f"{row['Feature']:<25} {row['SHAP Mean |Impact|']:>10.4f}\n"
        )
    
    # Comparar com coeficientes
    table_text += "\n" + "="*45 + "\n"
    table_text += "COEFICIENTES\n" + "="*45 + "\n\n"
    for feature, coef in zip(results['features'], model.coef_):
        table_text += f"{feature:<25} {coef:>10.4f}\n"
    
    axes[1, 1].text(
        0.1, 0.9, table_text, 
        transform=axes[1, 1].transAxes,
        fontsize=9, 
        verticalalignment='top', 
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    )
    
    plt.tight_layout()
    plt.savefig('linear_shap_analysis.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: linear_shap_analysis.png\n")
    plt.show()


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def save_results(results: dict, filename: str) -> None:
    """Salva resultados em JSON."""
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'strategy': ConfigLinearRegression.VALIDATION_STRATEGY,
            'fit_intercept': ConfigLinearRegression.FIT_INTERCEPT,
            'features': ConfigLinearRegression.FEATURES,
        }
    }
    
    if ConfigLinearRegression.VALIDATION_STRATEGY == 'holdout':
        results_to_save['metrics'] = {
            key: float(value) if isinstance(value, (int, float, np.number)) else value
            for key, value in results['metrics'].items()
            if key != 'coefficients'
        }
        results_to_save['coefficients'] = {
            key: float(value) 
            for key, value in results['metrics']['coefficients'].items()
        }
    else:
        results_to_save['cv_scores'] = {
            'r2_mean': float(results['cv_r2_mean']),
            'r2_std': float(results['cv_r2_std']),
            'mae_mean': float(results['cv_mae_mean']),
            'mae_std': float(results['cv_mae_std']),
        }
    
    with open(filename, 'w') as file:
        json.dump(results_to_save, file, indent=2)
    
    print(f"\uf05d Resultados salvos: {filename}")


def main():
    """Pipeline principal."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  PREVISÃO DE CONGESTIONAMENTO COM LINEAR REGRESSION - PIPELINE MODULAR".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print(f"\n\ued7b CONFIGURAÇÕES:")
    print(f"   Dataset: {ConfigLinearRegression.DATASET_PATH}")
    print(f"   Features: {ConfigLinearRegression.FEATURES}")
    print(f"   Target: {ConfigLinearRegression.TARGET}")
    print(f"   Validação: {ConfigLinearRegression.VALIDATION_STRATEGY.upper()}")
    if ConfigLinearRegression.VALIDATION_STRATEGY == 'holdout':
        print(f"   Test Size: {ConfigLinearRegression.HOLDOUT_TEST_SIZE}")
    else:
        print(f"   N-Folds: {ConfigLinearRegression.KFOLD_N_SPLITS}")
    print(f"   Fit Intercept: {ConfigLinearRegression.FIT_INTERCEPT}")
    print(f"   SHAP: {'Habilitado' if ConfigLinearRegression.SHAP_ENABLED else 'Desabilitado'}")
    print(f"   Análise Estatística: {'Habilitada' if ConfigLinearRegression.CALCULATE_STATISTICS else 'Desabilitada'}\n")
    
    try:
        dataframe = prepare_data(ConfigLinearRegression.DATASET_PATH)
        
        results = train_model(dataframe)
        
        print_metrics(results)
        
        if ConfigLinearRegression.VALIDATION_STRATEGY == 'holdout':
            plot_results_holdout(results)
        else:
            plot_results_kfold(results)
        
        plot_coefficients(results)
        
        explain_with_shap(results)
        
        if ConfigLinearRegression.SAVE_RESULTS:
            save_results(results, ConfigLinearRegression.RESULTS_FILE)
        
        print("\n" + "="*80)
        print("  \uf058 PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*80 + "\n")
        
    except Exception as error:
        print(f"\n\uea87 ERRO: {str(error)}")
        raise


if __name__ == '__main__':
    main()