#!/usr/bin/env python3
"""
    Previsão de Congestionamento de Trânsito usando XGBoost
    
    Autor: Alexandre Marques Tortoza Canoa
    Versão do Python: 3.13.7
    
    DEPENDÊNCIAS:
    ------------
    pip install xgboost scikit-learn pandas numpy matplotlib shap
    
    VERSÕES RECOMENDADAS (para evitar conflitos com SHAP):
    ------------------------------------------------------
    xgboost==1.7.6
    shap==0.42.1
    scikit-learn>=1.0.0
    
    OU (versões mais recentes, mas pode ter problemas com SHAP):
    xgboost>=2.0.0
    shap>=0.43.0
    
    Se encontrar erro com SHAP:
    - Opção 1: pip install xgboost==1.7.6 shap==0.42.1
    - Opção 2: Desabilitar SHAP (ConfigXGBoost.SHAP_ENABLED = False)
    - Opção 3: O código tem fallback automático para métodos alternativos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
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

# Verificação de versões (útil para debugging)
def check_versions():
    """Verifica versões das bibliotecas principais."""
    import sys
    print("\n" + "="*80)
    print("   VERSÕES DAS BIBLIOTECAS")
    print("="*80)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"XGBoost: {xgb.__version__}")
    if HAS_SHAP:
        print(f"SHAP: {shap.__version__}")
    print("="*80 + "\n")
    
    # Aviso sobre compatibilidade
    xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
    if HAS_SHAP and xgb_version >= (2, 0):
        print("\uea6c  AVISO: XGBoost 2.0+ pode ter problemas com SHAP.")
        print("   Se encontrar erros, recomenda-se:")
        print("   - pip install xgboost==1.7.6 shap==0.42.1")
        print("   OU")
        print("   - Desabilitar SHAP: ConfigXGBoost.SHAP_ENABLED = False\n")

# ============================================================================
# CONFIGURAÇÕES GLOBAIS - ALTERE CONFORME NECESSÁRIO
# ============================================================================

class ConfigXGBoost:
    """Centraliza todas as configurações do experimento."""
    
    DATASET_PATH = "../../dataset/preparedData/dataset.csv"
    DELIMITER = ";"
    
    # Estratégia de Validação
    # Opções: 'holdout' ou 'kfold'
    VALIDATION_STRATEGY = 'holdout'
    HOLDOUT_TEST_SIZE = 0.2  # Proporção teste (0.2 = 80/20)
    KFOLD_N_SPLITS = 5  # Número de folds para validação cruzada
    RANDOM_STATE = 42
    
    # Hiperparâmetros do XGBoost
    N_ESTIMATORS = 100  # Número de árvores
    MAX_DEPTH = 6  # Profundidade máxima das árvores
    LEARNING_RATE = 0.1  # Taxa de aprendizado (eta)
    SUBSAMPLE = 0.8  # Proporção de amostras para cada árvore
    COLSAMPLE_BYTREE = 0.8  # Proporção de features para cada árvore
    MIN_CHILD_WEIGHT = 1  # Peso mínimo em cada nó filho
    GAMMA = 0  # Redução mínima de perda para split
    REG_ALPHA = 0  # Regularização L1
    REG_LAMBDA = 1  # Regularização L2
    
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
    PLOT_TREES = True  # Visualizar árvores de decisão
    PLOT_FEATURE_IMPORTANCE = True  # Importância de features do XGBoost
    
    # Logs
    VERBOSE = True
    SAVE_RESULTS = True
    RESULTS_FILE = f"resultados_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


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
                if ConfigXGBoost.VERBOSE:
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
    if ConfigXGBoost.VERBOSE:
        print("\uf017 Extraindo features temporais...")
    
    dataframe['datetime'] = pd.to_datetime(
        dataframe['data'] + ' ' + dataframe['hora'],
        format='%Y-%m-%d %H:%M:%S'
    )
    
    dataframe['hora_numeric'] = dataframe['datetime'].dt.hour
    dataframe['dia_semana'] = dataframe['datetime'].dt.dayofweek  # 0=segunda, 6=domingo
    dataframe['mes'] = dataframe['datetime'].dt.month
    dataframe['dia_mes'] = dataframe['datetime'].dt.day
    
    if ConfigXGBoost.VERBOSE:
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
    if ConfigXGBoost.VERBOSE:
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
            
            if ConfigXGBoost.VERBOSE:
                classes_dict = dict(zip(
                    label_encoder.classes_, 
                    label_encoder.transform(label_encoder.classes_)
                ))
                print(f"  \uf05d {column}: {classes_dict}")
    
    if ConfigXGBoost.VERBOSE:
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
    
    dataframe = load_data(filepath, delimiter=ConfigXGBoost.DELIMITER)
    
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
# FUNÇÕES DE TREINAMENTO
# ============================================================================

def train_xgboost_holdout(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray
) -> dict:
    """
        Treina XGBoost com validação holdout (80/20).
        
        Args:
            X_train, X_test: Features de treino/teste
            y_train, y_test: Target de treino/teste
            
        Returns:
            Dicionário com modelo, métricas e dados
    """
    print(f" Treinando XGBoost (n_estimators={ConfigXGBoost.N_ESTIMATORS})...")
    
    model = xgb.XGBRegressor(
        n_estimators=ConfigXGBoost.N_ESTIMATORS,
        max_depth=ConfigXGBoost.MAX_DEPTH,
        learning_rate=ConfigXGBoost.LEARNING_RATE,
        subsample=ConfigXGBoost.SUBSAMPLE,
        colsample_bytree=ConfigXGBoost.COLSAMPLE_BYTREE,
        min_child_weight=ConfigXGBoost.MIN_CHILD_WEIGHT,
        gamma=ConfigXGBoost.GAMMA,
        reg_alpha=ConfigXGBoost.REG_ALPHA,
        reg_lambda=ConfigXGBoost.REG_LAMBDA,
        random_state=ConfigXGBoost.RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    
    # Treinar com early stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, 
        y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
    }
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'metrics': metrics,
    }


def train_xgboost_kfold(X: np.ndarray, y: np.ndarray) -> dict:
    """
        Treina XGBoost com validação cruzada K-Fold.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dicionário com scores e estatísticas
    """
    print(f" Treinando XGBoost com {ConfigXGBoost.KFOLD_N_SPLITS}-Fold CV...")
    
    model = xgb.XGBRegressor(
        n_estimators=ConfigXGBoost.N_ESTIMATORS,
        max_depth=ConfigXGBoost.MAX_DEPTH,
        learning_rate=ConfigXGBoost.LEARNING_RATE,
        subsample=ConfigXGBoost.SUBSAMPLE,
        colsample_bytree=ConfigXGBoost.COLSAMPLE_BYTREE,
        min_child_weight=ConfigXGBoost.MIN_CHILD_WEIGHT,
        gamma=ConfigXGBoost.GAMMA,
        reg_alpha=ConfigXGBoost.REG_ALPHA,
        reg_lambda=ConfigXGBoost.REG_LAMBDA,
        random_state=ConfigXGBoost.RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    
    kfold = KFold(
        n_splits=ConfigXGBoost.KFOLD_N_SPLITS, 
        shuffle=True, 
        random_state=ConfigXGBoost.RANDOM_STATE
    )
    
    scores_r2 = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    scores_mae = cross_val_score(
        model, X, y, cv=kfold, 
        scoring='neg_mean_absolute_error'
    )
    
    # Treinar modelo final com todos os dados para análise posterior
    model.fit(X, y, verbose=False)
    
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
        feature for feature in ConfigXGBoost.FEATURES 
        if feature not in dataframe.columns
    ]
    if missing_features:
        print(f"\uea87 Features não encontradas: {missing_features}")
        print(f"   Colunas disponíveis: {list(dataframe.columns)}")
        raise ValueError("Features faltando no dataset")
    
    dataframe_clean = dataframe.dropna(
        subset=ConfigXGBoost.FEATURES + [ConfigXGBoost.TARGET]
    )
    print(f"\uf05d Dados limpos: {len(dataframe)} → {len(dataframe_clean)} linhas\n")
    
    X = dataframe_clean[ConfigXGBoost.FEATURES].values
    y = dataframe_clean[ConfigXGBoost.TARGET].values
    
    # XGBoost não requer normalização, mas vamos manter para consistência
    print(" Normalizando features (opcional para XGBoost)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  \uf05d Média: {X_scaled.mean(axis=0)}")
    print(f"  \uf05d Std: {X_scaled.std(axis=0)}\n")
    
    if ConfigXGBoost.VALIDATION_STRATEGY == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=ConfigXGBoost.HOLDOUT_TEST_SIZE,
            random_state=ConfigXGBoost.RANDOM_STATE
        )
        print(f"\uf05d Split: {len(X_train)} treino | {len(X_test)} teste\n")
        
        results = train_xgboost_holdout(X_train, X_test, y_train, y_test)
        
    elif ConfigXGBoost.VALIDATION_STRATEGY == 'kfold':
        results = train_xgboost_kfold(X_scaled, y)
    
    else:
        raise ValueError(
            f"Estratégia desconhecida: {ConfigXGBoost.VALIDATION_STRATEGY}"
        )
    
    results['scaler'] = scaler
    results['features'] = ConfigXGBoost.FEATURES
    results['dataframe'] = dataframe_clean
    
    return results


def print_metrics(results: dict) -> None:
    """Imprime métricas de forma formatada."""
    print("\n" + "="*80)
    print("  \ueb03 MÉTRICAS DO MODELO")
    print("="*80)
    
    if ConfigXGBoost.VALIDATION_STRATEGY == 'holdout':
        metrics = results['metrics']
        print(f"\n{'Métrica':<25} {'Treino':>12} {'Teste':>12}")
        print("-" * 50)
        print(f"{'MAE':<25} {metrics['train_mae']:>12.4f} {metrics['test_mae']:>12.4f}")
        print(f"{'RMSE':<25} {metrics['train_rmse']:>12.4f} {metrics['test_rmse']:>12.4f}")
        print(f"{'R² Score':<25} {metrics['train_r2']:>12.4f} {metrics['test_r2']:>12.4f}")
        
    elif ConfigXGBoost.VALIDATION_STRATEGY == 'kfold':
        print(f"\n{ConfigXGBoost.KFOLD_N_SPLITS}-Fold Cross Validation:")
        print("-" * 50)
        print(f"R² Scores: {results['cv_r2_scores']}")
        print(f"R² Média: {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")
        print(f"MAE Média: {results['cv_mae_mean']:.4f} (+/- {results['cv_mae_std']:.4f})")
    
    print()


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================================

def plot_results_holdout(results: dict) -> None:
    """Plota resultados para validação holdout."""
    if not ConfigXGBoost.PLOT_RESULTS:
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
    plt.savefig('xgboost_results_holdout.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: xgboost_results_holdout.png")
    plt.show()


def plot_results_kfold(results: dict) -> None:
    """Plota resultados para validação K-Fold."""
    if not ConfigXGBoost.PLOT_RESULTS:
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
    plt.savefig('xgboost_results_kfold.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: xgboost_results_kfold.png")
    plt.show()


def plot_feature_importance(results: dict) -> None:
    """Plota importância de features do XGBoost."""
    if not ConfigXGBoost.PLOT_FEATURE_IMPORTANCE:
        return
    
    print("\n" + "="*80)
    print("   IMPORTÂNCIA DE FEATURES (XGBoost)")
    print("="*80)
    
    model = results['model']
    features = results['features']
    
    # Diferentes métricas de importância
    importance_types = ['weight', 'gain', 'cover']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for index, importance_type in enumerate(importance_types):
        importance_dict = model.get_booster().get_score(
            importance_type=importance_type
        )
        
        # Mapear de f0, f1, etc para nomes reais
        importance_values = []
        for feature_index, feature_name in enumerate(features):
            feature_key = f'f{feature_index}'
            importance_values.append(importance_dict.get(feature_key, 0))
        
        # Ordenar
        sorted_indices = np.argsort(importance_values)
        sorted_features = [features[index] for index in sorted_indices]
        sorted_values = [importance_values[index] for index in sorted_indices]
        
        ax = axes[index]
        ax.barh(sorted_features, sorted_values, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'Importância ({importance_type})')
        ax.set_title(f'Feature Importance - {importance_type.upper()}')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: xgboost_feature_importance.png\n")
    plt.show()


def plot_tree(results: dict, tree_index: int = 0) -> None:
    """Plota uma árvore de decisão do XGBoost."""
    if not ConfigXGBoost.PLOT_TREES:
        return
    
    try:
        fig, ax = plt.subplots(figsize=(20, 10))
        xgb.plot_tree(
            results['model'], 
            num_trees=tree_index, 
            ax=ax,
            rankdir='LR'
        )
        plt.title(f'Árvore de Decisão #{tree_index}')
        plt.tight_layout()
        plt.savefig(
            f'xgboost_tree_{tree_index}.png', 
            dpi=150, 
            bbox_inches='tight'
        )
        print(f"\uf05d Gráfico salvo: xgboost_tree_{tree_index}.png")
        plt.show()
    except Exception as error:
        print(f"\uea87 Erro ao plotar árvore: {error}")


# ============================================================================
# FUNÇÕES SHAP
# ============================================================================

def explain_with_shap(results: dict) -> None:
    """Análise de explicabilidade com SHAP."""
    if not ConfigXGBoost.SHAP_ENABLED or not HAS_SHAP:
        if ConfigXGBoost.VERBOSE:
            print("\uea6c  SHAP desabilitado ou não disponível")
        return
    
    print("\n" + "="*80)
    print("   ANÁLISE SHAP")
    print("="*80)
    print(f"Executando SHAP com TreeExplainer (nativo para XGBoost)...")
    
    model = results['model']
    
    if ConfigXGBoost.VALIDATION_STRATEGY == 'holdout':
        X_sample = results['X_test'][:ConfigXGBoost.SHAP_N_SAMPLES]
    else:
        X_sample = results['X'][:ConfigXGBoost.SHAP_N_SAMPLES]
    
    # TreeExplainer é muito mais rápido para modelos baseados em árvores
    try:
        # Tenta criar explainer com model_output='raw' para evitar problemas
        explainer = shap.TreeExplainer(model, model_output='raw')
        shap_values = explainer.shap_values(X_sample)
    except (ValueError, AttributeError) as error:
        print(f"\uea87 Erro com TreeExplainer: {error}")
        print("   Tentando método alternativo (mais lento)...")
        
        # Fallback: usar Explainer genérico que escolhe automaticamente
        try:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample).values
        except Exception as error2:
            print(f"\uea87 Erro com Explainer genérico: {error2}")
            print("   SHAP não disponível para este modelo. Pulando análise SHAP.")
            return
    
    print("\uf05d SHAP values calculados\n")
    
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
    
    axes[1, 1].text(
        0.1, 0.9, table_text, 
        transform=axes[1, 1].transAxes,
        fontsize=10, 
        verticalalignment='top', 
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    )
    
    plt.tight_layout()
    plt.savefig('xgboost_shap_analysis.png', dpi=150, bbox_inches='tight')
    print("\uf05d Gráfico salvo: xgboost_shap_analysis.png\n")
    plt.show()
    
    # Waterfall plot para uma predição individual
    if ConfigXGBoost.PLOT_SHAP:
        print("Gerando Waterfall Plot para uma predição exemplo...")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_sample[0],
                feature_names=results['features']
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig('xgboost_shap_waterfall.png', dpi=150, bbox_inches='tight')
        print("\uf05d Gráfico salvo: xgboost_shap_waterfall.png\n")
        plt.show()


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def save_results(results: dict, filename: str) -> None:
    """Salva resultados em JSON."""
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'strategy': ConfigXGBoost.VALIDATION_STRATEGY,
            'n_estimators': ConfigXGBoost.N_ESTIMATORS,
            'max_depth': ConfigXGBoost.MAX_DEPTH,
            'learning_rate': ConfigXGBoost.LEARNING_RATE,
            'features': ConfigXGBoost.FEATURES,
        }
    }
    
    if ConfigXGBoost.VALIDATION_STRATEGY == 'holdout':
        results_to_save['metrics'] = {
            key: float(value) 
            for key, value in results['metrics'].items()
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
    print("║" + "  PREVISÃO DE CONGESTIONAMENTO COM XGBOOST - PIPELINE MODULAR".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Verificar versões das bibliotecas
    check_versions()
    
    print(f"\n\ued7b CONFIGURAÇÕES:")
    print(f"   Dataset: {ConfigXGBoost.DATASET_PATH}")
    print(f"   Features: {ConfigXGBoost.FEATURES}")
    print(f"   Target: {ConfigXGBoost.TARGET}")
    print(f"   Validação: {ConfigXGBoost.VALIDATION_STRATEGY.upper()}")
    if ConfigXGBoost.VALIDATION_STRATEGY == 'holdout':
        print(f"   Test Size: {ConfigXGBoost.HOLDOUT_TEST_SIZE}")
    else:
        print(f"   N-Folds: {ConfigXGBoost.KFOLD_N_SPLITS}")
    print(f"   N-Estimators: {ConfigXGBoost.N_ESTIMATORS}")
    print(f"   Max Depth: {ConfigXGBoost.MAX_DEPTH}")
    print(f"   Learning Rate: {ConfigXGBoost.LEARNING_RATE}")
    print(f"   SHAP: {'Habilitado' if ConfigXGBoost.SHAP_ENABLED else 'Desabilitado'}\n")
    
    try:
        dataframe = prepare_data(ConfigXGBoost.DATASET_PATH)
        
        results = train_model(dataframe)
        
        print_metrics(results)
        
        if ConfigXGBoost.VALIDATION_STRATEGY == 'holdout':
            plot_results_holdout(results)
        else:
            plot_results_kfold(results)
        
        plot_feature_importance(results)
        
        if ConfigXGBoost.PLOT_TREES:
            plot_tree(results, tree_index=0)
        
        explain_with_shap(results)
        
        if ConfigXGBoost.SAVE_RESULTS:
            save_results(results, ConfigXGBoost.RESULTS_FILE)
        
        print("\n" + "="*80)
        print("  \uf058 PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*80 + "\n")
        
    except Exception as error:
        print(f"\n\uea87 ERRO: {str(error)}")
        raise


if __name__ == '__main__':
    main()