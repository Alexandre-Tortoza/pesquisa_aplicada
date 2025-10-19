#!/usr/bin/env python3
"""
Previsão de Congestionamento de Trânsito usando Least Trimmed Squares (LTS)

Pipeline modular para prever tamanho_congestionamento baseado em população
e outras features. Suporta validação cruzada, holdout e análise SHAP.

LTS é um método robusto que minimiza a soma dos resíduos ordenados,
tornando-o resistente a outliers.

Dataset: ../../dataset/preparedData/dataset.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import HuberRegressor
import warnings
import json
from datetime import datetime
from scipy import stats

try:
    from sklearn.linear_model import LassoCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  Para usar SHAP: pip install shap")

try:
    from sklearn.covariance import EllipticEnvelope
    HAS_ROBUST = True
except ImportError:
    HAS_ROBUST = False

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES GLOBAIS - ALTERE CONFORME NECESSÁRIO
# ============================================================================

class ConfigLTS:
    """Centraliza todas as configurações do experimento."""
    
    # 📁 Caminhos
    DATASET_PATH = "../../dataset/preparedData/dataset.csv"
    DELIMITER = ";"  # Delimitador do CSV
    
    # 🔧 Estratégia de Validação
    # Opções: 'holdout' ou 'kfold'
    VALIDATION_STRATEGY = 'holdout'
    HOLDOUT_TEST_SIZE = 0.2  # Proporção teste (0.2 = 80/20)
    KFOLD_N_SPLITS = 5  # Número de folds para validação cruzada
    RANDOM_STATE = 42
    
    # 🎯 Hiperparâmetros do LTS (via HuberRegressor como aproximação)
    # Nota: sklearn não tem LTS nativo. Usamos HuberRegressor como alternativa robusta
    # LTS = Least Trimmed Squares (robust regression que ignora outliers)
    # HuberRegressor = similar, mas mais eficiente computacionalmente
    
    EPSILON = 1.35  # Threshold de robustez (quanto maior = menos robusto)
    MAX_ITER = 1000  # Máximo de iterações
    ALPHA = 0.0001  # Regularização L2 (Tikhonov)
    
    # 📊 Features e Target
    FEATURES = [
        'pop_total',          # População total (principal feature)
        'hora_numeric',       # Hora do dia (0-23)
        'via_expressa_encoded',  # Via expressa (E, N, S, W)
        'regiao_encoded',     # Região (east, center, etc)
        'sexo_encoded',       # Sexo (Homens, Mulheres)
        'dia_semana',         # Dia da semana (0-6)
        'mes',                # Mês (1-12)
    ]
    TARGET = 'tamanho_congestionamento'
    
    # 🎯 SHAP
    SHAP_ENABLED = True
    SHAP_N_SAMPLES = 100  # Amostras para explicabilidade
    
    # 📈 Visualizações
    PLOT_RESULTS = True
    PLOT_SHAP = True
    PLOT_OUTLIERS = True  # Plot detecção de outliers
    
    # 📝 Logs
    VERBOSE = True
    SAVE_RESULTS = True
    RESULTS_FILE = f"resultados_lts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 🔍 Detecção de Outliers
    OUTLIER_DETECTION = True  # Detectar outliers
    OUTLIER_THRESHOLD = 2.5  # Desvios padrão para considerar outlier


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
        # Tenta diferentes encodings
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
                if ConfigLTS.VERBOSE:
                    print(f"✓ Dataset carregado: {df.shape}")
                    print(f"  Encoding: {encoding}")
                    print(f"  Colunas: {list(df.columns)}\n")
                return df
            except UnicodeDecodeError:
                continue
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {filepath}")
        raise
    
    print(f"❌ Não foi possível carregar o arquivo com nenhum encoding")
    raise ValueError("Erro ao carregar arquivo")


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai features temporais das colunas 'data' e 'hora'.
    
    Args:
        df: DataFrame com colunas 'data' e 'hora'
        
    Returns:
        DataFrame com novas colunas de features temporais
    """
    if ConfigLTS.VERBOSE:
        print("🕐 Extraindo features temporais...")
    
    # Combina data e hora
    df['datetime'] = pd.to_datetime(
        df['data'] + ' ' + df['hora'],
        format='%Y-%m-%d %H:%M:%S'
    )
    
    # Extrai features
    df['hora_numeric'] = df['datetime'].dt.hour
    df['dia_semana'] = df['datetime'].dt.dayofweek  # 0=segunda, 6=domingo
    df['mes'] = df['datetime'].dt.month
    df['dia_mes'] = df['datetime'].dt.day
    
    if ConfigLTS.VERBOSE:
        print("  ✓ Features: hora_numeric, dia_semana, mes, dia_mes\n")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica features categóricas em numéricas.
    
    Args:
        df: DataFrame com features categóricas
        
    Returns:
        DataFrame com features codificadas
    """
    if ConfigLTS.VERBOSE:
        print("🔤 Codificando features categóricas...")
    
    encoders = {}
    categorical_cols = ['via_expressa', 'regiao', 'sexo']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
            encoders[col] = le
            
            if ConfigLTS.VERBOSE:
                print(f"  ✓ {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    if ConfigLTS.VERBOSE:
        print()
    
    return df


def prepare_data(filepath: str) -> pd.DataFrame:
    """
    Pipeline completo de preparação de dados.
    
    Args:
        filepath: Caminho do dataset
        
    Returns:
        DataFrame preparado
    """
    print("="*80)
    print("  📊 CARREGAMENTO E PREPARAÇÃO DE DADOS")
    print("="*80)
    
    # 1. Carrega dados
    df = load_data(filepath, delimiter=ConfigLTS.DELIMITER)
    
    # 2. Verifica valores ausentes iniciais
    print(f"📌 Verificando valores ausentes iniciais:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  {missing[missing > 0].to_dict()}\n")
    else:
        print("  ✓ Nenhum valor ausente\n")
    
    # 3. Extrai features temporais
    df = extract_datetime_features(df)
    
    # 4. Codifica features categóricas
    df = encode_categorical_features(df)
    
    # 5. Agregação de dados
    print("🔀 Agregando dados por data/hora/via/região...")
    df_agg = df.groupby(['data', 'hora', 'via_expressa', 'regiao']).agg({
        'pop_total': 'sum',
        'tamanho_congestionamento': 'first',
        'hora_numeric': 'first',
        'dia_semana': 'first',
        'mes': 'first',
        'via_expressa_encoded': 'first',
        'regiao_encoded': 'first',
        'sexo_encoded': 'mean',
    }).reset_index()
    
    print(f"  ✓ {len(df)} → {len(df_agg)} linhas\n")
    
    # 6. Verifica valores ausentes finais
    print(f"📌 Valores ausentes após preparação:")
    missing = df_agg.isnull().sum()
    if missing.sum() > 0:
        print(f"  {missing[missing > 0].to_dict()}\n")
    else:
        print("  ✓ Nenhum valor ausente\n")
    
    # 7. Estatísticas básicas
    print(f"📊 Estatísticas do target (tamanho_congestionamento):")
    print(f"  Mínimo: {df_agg['tamanho_congestionamento'].min()}")
    print(f"  Máximo: {df_agg['tamanho_congestionamento'].max()}")
    print(f"  Média: {df_agg['tamanho_congestionamento'].mean():.2f}")
    print(f"  Mediana: {df_agg['tamanho_congestionamento'].median():.2f}")
    print(f"  Std Dev: {df_agg['tamanho_congestionamento'].std():.2f}\n")
    
    return df_agg


# ============================================================================
# FUNÇÕES DE DETECÇÃO DE OUTLIERS
# ============================================================================

def detect_outliers(y: np.ndarray, threshold: float = 2.5) -> np.ndarray:
    """
    Detecta outliers usando z-score.
    
    Args:
        y: Array do target
        threshold: Número de desvios padrão (padrão: 2.5)
        
    Returns:
        Boolean array indicando outliers
    """
    z_scores = np.abs(stats.zscore(y))
    return z_scores > threshold


def print_outlier_summary(y: np.ndarray, outlier_mask: np.ndarray) -> None:
    """Imprime resumo de outliers detectados."""
    n_outliers = outlier_mask.sum()
    pct_outliers = (n_outliers / len(y)) * 100
    
    print(f"\n🔍 ANÁLISE DE OUTLIERS")
    print(f"  Total de outliers: {n_outliers} ({pct_outliers:.2f}%)")
    print(f"  Valores outliers: {y[outlier_mask].min():.2f} - {y[outlier_mask].max():.2f}")
    print(f"  Valores normais: {y[~outlier_mask].min():.2f} - {y[~outlier_mask].max():.2f}\n")


# ============================================================================
# FUNÇÕES DE TREINAMENTO
# ============================================================================

def train_lts_holdout(X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray,
                      feature_names: list) -> dict:
    """
    Treina modelo LTS (via HuberRegressor) com validação holdout.
    
    Nota: HuberRegressor é uma alternativa robusta ao LTS.
    Ambos minimizam o impacto de outliers no treinamento.
    
    Args:
        X_train, X_test: Features de treino/teste (escaladas)
        y_train, y_test: Target de treino/teste
        feature_names: Nomes das features
        
    Returns:
        Dicionário com modelo, métricas e dados
    """
    print(f"📊 Treinando LTS (HuberRegressor com epsilon={ConfigLTS.EPSILON})...")
    
    # Treina modelo robusto
    model = HuberRegressor(
        epsilon=ConfigLTS.EPSILON,
        max_iter=ConfigLTS.MAX_ITER,
        alpha=ConfigLTS.ALPHA,
        random_state=ConfigLTS.RANDOM_STATE,
        verbose=0
    )
    model.fit(X_train, y_train)
    
    # Predições
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
    }
    
    # Importância dos coeficientes
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': np.abs(model.coef_)
    }).sort_values('Coefficient', ascending=False)
    
    # Detecção de outliers
    outlier_mask_train = detect_outliers(y_train, ConfigLTS.OUTLIER_THRESHOLD)
    outlier_mask_test = detect_outliers(y_test, ConfigLTS.OUTLIER_THRESHOLD)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'outlier_mask_train': outlier_mask_train,
        'outlier_mask_test': outlier_mask_test,
    }


def train_lts_kfold(X: np.ndarray, y: np.ndarray, 
                    feature_names: list) -> dict:
    """
    Treina LTS com validação cruzada K-Fold.
    
    Args:
        X: Features (escaladas)
        y: Target
        feature_names: Nomes das features
        
    Returns:
        Dicionário com scores e estatísticas
    """
    print(f"📊 Treinando LTS com {ConfigLTS.KFOLD_N_SPLITS}-Fold CV...")
    
    from sklearn.metrics import mean_absolute_error, r2_score
    
    model = HuberRegressor(
        epsilon=ConfigLTS.EPSILON,
        max_iter=ConfigLTS.MAX_ITER,
        alpha=ConfigLTS.ALPHA,
        random_state=ConfigLTS.RANDOM_STATE,
        verbose=0
    )
    
    kfold = KFold(n_splits=ConfigLTS.KFOLD_N_SPLITS, 
                  shuffle=True, 
                  random_state=ConfigLTS.RANDOM_STATE)
    
    # Calcula scores
    scores_r2 = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    scores_mae = cross_val_score(model, X, y, cv=kfold, 
                                 scoring='neg_mean_absolute_error')
    
    # Treina modelo final
    model.fit(X, y)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': np.abs(model.coef_)
    }).sort_values('Coefficient', ascending=False)
    
    # Detecção de outliers
    outlier_mask = detect_outliers(y, ConfigLTS.OUTLIER_THRESHOLD)
    
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
        'feature_importance': feature_importance,
        'outlier_mask': outlier_mask,
    }


def train_model(df: pd.DataFrame) -> dict:
    """
    Pipeline de treinamento adaptado à estratégia de validação.
    
    Args:
        df: DataFrame preparado
        
    Returns:
        Resultados do treinamento
    """
    print("="*80)
    print("  📊 TREINAMENTO DO MODELO - LEAST TRIMMED SQUARES")
    print("="*80)
    
    # Verifica features disponíveis
    missing_features = [f for f in ConfigLTS.FEATURES if f not in df.columns]
    if missing_features:
        print(f"❌ Features não encontradas: {missing_features}")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        raise ValueError("Features faltando no dataset")
    
    # Remove NaN
    df_clean = df.dropna(subset=ConfigLTS.FEATURES + [ConfigLTS.TARGET])
    print(f"✓ Dados limpos: {len(df)} → {len(df_clean)} linhas")
    print(f"  Proporção mantida: {len(df_clean)/len(df)*100:.2f}%\n")
    
    # Separa features e target
    X = df_clean[ConfigLTS.FEATURES].values
    y = df_clean[ConfigLTS.TARGET].values
    
    # Normaliza features (importante para regressão linear robusta)
    print("📏 Normalizando features (necessário para LTS)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  ✓ Média: {X_scaled.mean(axis=0)}")
    print(f"  ✓ Std: {X_scaled.std(axis=0)}\n")
    
    # Treina conforme estratégia
    if ConfigLTS.VALIDATION_STRATEGY == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=ConfigLTS.HOLDOUT_TEST_SIZE,
            random_state=ConfigLTS.RANDOM_STATE
        )
        print(f"✓ Split: {len(X_train)} treino | {len(X_test)} teste\n")
        
        results = train_lts_holdout(X_train, X_test, y_train, y_test, 
                                    ConfigLTS.FEATURES)
        
    elif ConfigLTS.VALIDATION_STRATEGY == 'kfold':
        results = train_lts_kfold(X_scaled, y, ConfigLTS.FEATURES)
    
    else:
        raise ValueError(f"Estratégia desconhecida: {ConfigLTS.VALIDATION_STRATEGY}")
    
    # Adiciona informações adicionais
    results['scaler'] = scaler
    results['features'] = ConfigLTS.FEATURES
    results['df'] = df_clean
    
    return results


def print_metrics(results: dict) -> None:
    """Imprime métricas de forma formatada."""
    print("\n" + "="*80)
    print("  📊 MÉTRICAS DO MODELO - LEAST TRIMMED SQUARES")
    print("="*80)
    
    if ConfigLTS.VALIDATION_STRATEGY == 'holdout':
        metrics = results['metrics']
        print(f"\n{'Métrica':<25} {'Treino':>12} {'Teste':>12}")
        print("-" * 50)
        print(f"{'MAE':<25} {metrics['train_mae']:>12.4f} {metrics['test_mae']:>12.4f}")
        print(f"{'RMSE':<25} {metrics['train_rmse']:>12.4f} {metrics['test_rmse']:>12.4f}")
        print(f"{'R² Score':<25} {metrics['train_r2']:>12.4f} {metrics['test_r2']:>12.4f}")
        
        # Resumo de outliers
        print_outlier_summary(results['y_train'], results['outlier_mask_train'])
        print_outlier_summary(results['y_test'], results['outlier_mask_test'])
        
    elif ConfigLTS.VALIDATION_STRATEGY == 'kfold':
        print(f"\n{ConfigLTS.KFOLD_N_SPLITS}-Fold Cross Validation:")
        print("-" * 50)
        print(f"R² Scores: {results['cv_r2_scores']}")
        print(f"R² Média: {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")
        print(f"MAE Média: {results['cv_mae_mean']:.4f} (+/- {results['cv_mae_std']:.4f})")
        
        # Resumo de outliers
        print_outlier_summary(results['y'], results['outlier_mask'])
    
    print(f"\n{'Coeficientes (Importância)':<25}")
    print("-" * 50)
    for idx, row in results['feature_importance'].head(7).iterrows():
        bar = "█" * int(row['Coefficient'] * 50)
        print(f"{row['Feature']:<25} {bar} {row['Coefficient']:.6f}")
    
    print()


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================================

def plot_results_holdout(results: dict) -> None:
    """Plota resultados para validação holdout."""
    if not ConfigLTS.PLOT_RESULTS:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cores para outliers
    colors_train = np.where(results['outlier_mask_train'], 'red', 'blue')
    colors_test = np.where(results['outlier_mask_test'], 'red', 'blue')
    
    # Gráfico 1: Real vs Previsto (Teste)
    ax = axes[0, 0]
    ax.scatter(results['y_test'][~results['outlier_mask_test']], 
               results['y_test_pred'][~results['outlier_mask_test']], 
               alpha=0.5, s=20, edgecolors='k', linewidth=0.5, label='Normal', color='blue')
    ax.scatter(results['y_test'][results['outlier_mask_test']], 
               results['y_test_pred'][results['outlier_mask_test']], 
               alpha=0.7, s=30, edgecolors='k', linewidth=1, label='Outlier', color='red', marker='x')
    min_val = min(results['y_test'].min(), results['y_test_pred'].min())
    max_val = max(results['y_test'].max(), results['y_test_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Previstos')
    ax.set_title('Teste: Real vs Previsto')
    ax.grid(True, alpha=0.3)
    ax.legend()
    textstr = f"R² = {results['metrics']['test_r2']:.4f}\nMAE = {results['metrics']['test_mae']:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Gráfico 2: Real vs Previsto (Treino)
    ax = axes[0, 1]
    ax.scatter(results['y_train'][~results['outlier_mask_train']], 
               results['y_train_pred'][~results['outlier_mask_train']], 
               alpha=0.5, s=20, edgecolors='k', linewidth=0.5, label='Normal', color='blue')
    ax.scatter(results['y_train'][results['outlier_mask_train']], 
               results['y_train_pred'][results['outlier_mask_train']], 
               alpha=0.7, s=30, edgecolors='k', linewidth=1, label='Outlier', color='red', marker='x')
    min_val = min(results['y_train'].min(), results['y_train_pred'].min())
    max_val = max(results['y_train'].max(), results['y_train_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Previstos')
    ax.set_title('Treino: Real vs Previsto')
    ax.grid(True, alpha=0.3)
    ax.legend()
    textstr = f"R² = {results['metrics']['train_r2']:.4f}\nMAE = {results['metrics']['train_mae']:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Gráfico 3: Distribuição de erros (Teste)
    ax = axes[1, 0]
    errors = results['y_test'] - results['y_test_pred']
    ax.hist(errors[~results['outlier_mask_test']], bins=50, edgecolor='black', alpha=0.7, 
            color='skyblue', label='Normal')
    ax.hist(errors[results['outlier_mask_test']], bins=20, edgecolor='black', alpha=0.7, 
            color='red', label='Outlier')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Erro de Previsão')
    ax.set_ylabel('Frequência')
    ax.set_title(f'Distribuição de Erros (Teste) - Média: {errors.mean():.2f}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Gráfico 4: Resíduos vs Previstos
    ax = axes[1, 1]
    residuals = results['y_test'] - results['y_test_pred']
    ax.scatter(results['y_test_pred'][~results['outlier_mask_test']], 
               residuals[~results['outlier_mask_test']], 
               alpha=0.5, s=20, edgecolors='k', linewidth=0.5, label='Normal', color='blue')
    ax.scatter(results['y_test_pred'][results['outlier_mask_test']], 
               residuals[results['outlier_mask_test']], 
               alpha=0.7, s=30, edgecolors='k', linewidth=1, label='Outlier', color='red', marker='x')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Valores Previstos')
    ax.set_ylabel('Residuais')
    ax.set_title('Análise de Resíduos (Teste)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('lts_results_holdout.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo: lts_results_holdout.png")
    plt.show()


def plot_results_kfold(results: dict) -> None:
    """Plota resultados para validação K-Fold."""
    if not ConfigLTS.PLOT_RESULTS:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Scores R²
    ax = axes[0]
    folds = np.arange(1, len(results['cv_r2_scores']) + 1)
    ax.bar(folds, results['cv_r2_scores'], alpha=0.7, color='skyblue', edgecolor='black')
    ax.axhline(y=results['cv_r2_mean'], color='r', linestyle='--', linewidth=2, label='Média')
    ax.fill_between(folds, 
                     results['cv_r2_mean'] - results['cv_r2_std'],
                     results['cv_r2_mean'] + results['cv_r2_std'],
                     alpha=0.2, color='r')
    ax.set_xlabel('Fold')
    ax.set_ylabel('R² Score')
    ax.set_title('Scores R² por Fold - LTS')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Gráfico 2: Scores MAE
    ax = axes[1]
    ax.bar(folds, results['cv_mae_scores'], alpha=0.7, color='lightcoral', edgecolor='black')
    ax.axhline(y=results['cv_mae_mean'], color='r', linestyle='--', linewidth=2, label='Média')
    ax.fill_between(folds,
                     results['cv_mae_mean'] - results['cv_mae_std'],
                     results['cv_mae_mean'] + results['cv_mae_std'],
                     alpha=0.2, color='r')
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE')
    ax.set_title('Scores MAE por Fold - LTS')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('lts_results_kfold.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo: lts_results_kfold.png")
    plt.show()


def plot_coefficients(results: dict) -> None:
    """Plota coeficientes do modelo."""
    if not ConfigLTS.PLOT_RESULTS:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coef_df = results['feature_importance'].head(10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(coef_df)))
    
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
    ax.set_xlabel('Coeficiente Absoluto', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Coeficientes do Modelo LTS', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Adiciona valores nas barras
    for i, (feature, coef) in enumerate(zip(coef_df['Feature'], coef_df['Coefficient'])):
        ax.text(coef, i, f' {coef:.6f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('lts_coefficients.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo: lts_coefficients.png")
    plt.show()


# ============================================================================
# FUNÇÕES SHAP
# ============================================================================

def explain_with_shap(results: dict) -> None:
    """Análise de explicabilidade com SHAP (holdout only)."""
    if not ConfigLTS.SHAP_ENABLED or not HAS_SHAP:
        if ConfigLTS.VERBOSE:
            print("⚠️  SHAP desabilitado ou não disponível")
        return
    
    if ConfigLTS.VALIDATION_STRATEGY != 'holdout':
        print("⚠️  SHAP disponível apenas para validação holdout")
        return
    
    print("\n" + "="*80)
    print("  🔍 ANÁLISE SHAP - LEAST TRIMMED SQUARES")
    print("="*80)
    print(f"Executando SHAP com {ConfigLTS.SHAP_N_SAMPLES} amostras...")
    
    # Usa KernelExplainer para modelos lineares
    explainer = shap.KernelExplainer(results['model'].predict, results['X_train'])
    
    # Calcula SHAP values para amostra do teste
    n_samples = min(ConfigLTS.SHAP_N_SAMPLES, len(results['X_test']))
    X_test_sample = results['X_test'][:n_samples]
    shap_values = explainer.shap_values(X_test_sample)
    
    print("✓ SHAP values calculados\n")
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Summary plot (scatter)
    plt.sca(axes[0, 0])
    shap.summary_plot(shap_values, X_test_sample, 
                      feature_names=results['features'], show=False)
    axes[0, 0].set_title('Importância Global (SHAP)', fontweight='bold', fontsize=12)
    
    # 2. Summary plot (bar)
    plt.sca(axes[0, 1])
    shap.summary_plot(shap_values, X_test_sample, 
                      feature_names=results['features'], 
                      plot_type="bar", show=False)
    axes[0, 1].set_title('Importância Média Absoluta', fontweight='bold', fontsize=12)
    
    # 3. Dependence plot
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argmax(mean_abs_shap)
    
    plt.sca(axes[1, 0])
    shap.dependence_plot(top_idx, shap_values, X_test_sample,
                         feature_names=results['features'], ax=axes[1, 0], show=False)
    
    # 4. Ranking
    axes[1, 1].axis('off')
    importance_df = pd.DataFrame({
        'Feature': results['features'],
        'SHAP Mean |Impact|': mean_abs_shap
    }).sort_values('SHAP Mean |Impact|', ascending=False)
    
    table_text = "RANKING SHAP\n" + "="*45 + "\n\n"
    for idx, row in importance_df.iterrows():
        table_text += f"{row['Feature']:<25} {row['SHAP Mean |Impact|']:>10.4f}\n"
    
    axes[1, 1].text(0.1, 0.9, table_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('lts_shap_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo: lts_shap_analysis.png\n")
    plt.show()


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Pipeline principal."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  PREVISÃO DE CONGESTIONAMENTO COM LTS - PIPELINE MODULAR".center(78) + "║")
    print("║" + " "*78 + "║")
    print("║" + "  (Least Trimmed Squares - Regressão Robusta)".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print(f"\n📋 CONFIGURAÇÕES:")
    print(f"   Dataset: {ConfigLTS.DATASET_PATH}")
    print(f"   Features: {ConfigLTS.FEATURES}")
    print(f"   Target: {ConfigLTS.TARGET}")
    print(f"   Validação: {ConfigLTS.VALIDATION_STRATEGY.upper()}")
    if ConfigLTS.VALIDATION_STRATEGY == 'holdout':
        print(f"   Test Size: {ConfigLTS.HOLDOUT_TEST_SIZE}")
    else:
        print(f"   N-Folds: {ConfigLTS.KFOLD_N_SPLITS}")
    print(f"   Epsilon (Robustez): {ConfigLTS.EPSILON}")
    print(f"   Outlier Threshold: {ConfigLTS.OUTLIER_THRESHOLD} desvios padrão")
    print(f"   SHAP: {'Habilitado' if ConfigLTS.SHAP_ENABLED else 'Desabilitado'}\n")
    
    print("ℹ️  LTS é resistente a outliers. Detecta e minimiza seu impacto automaticamente.\n")
    
    try:
        # 1. Preparação
        df = prepare_data(ConfigLTS.DATASET_PATH)
        
        # 2. Treinamento
        results = train_model(df)
        
        # 3. Métricas
        print_metrics(results)
        
        # 4. Visualizações
        if ConfigLTS.VALIDATION_STRATEGY == 'holdout':
            plot_results_holdout(results)
            plot_coefficients(results)
        else:
            plot_results_kfold(results)
            plot_coefficients(results)
        
        # 5. SHAP
        explain_with_shap(results)
        
        # 6. Salva resultados
        if ConfigLTS.SAVE_RESULTS:
            save_results(results, ConfigLTS.RESULTS_FILE)
        
        print("\n" + "="*80)
        print("  ✅ PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        raise


def save_results(results: dict, filename: str) -> None:
    """Salva resultados em JSON."""
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'algorithm': 'Least Trimmed Squares (HuberRegressor)',
        'configuration': {
            'strategy': ConfigLTS.VALIDATION_STRATEGY,
            'epsilon': ConfigLTS.EPSILON,
            'features': ConfigLTS.FEATURES,
        }
    }
    
    if ConfigLTS.VALIDATION_STRATEGY == 'holdout':
        results_to_save['metrics'] = {
            k: float(v) for k, v in results['metrics'].items()
        }
        results_to_save['outlier_summary'] = {
            'train_outliers': int(results['outlier_mask_train'].sum()),
            'test_outliers': int(results['outlier_mask_test'].sum()),
        }
    else:
        results_to_save['cv_scores'] = {
            'r2_mean': float(results['cv_r2_mean']),
            'r2_std': float(results['cv_r2_std']),
            'mae_mean': float(results['cv_mae_mean']),
            'mae_std': float(results['cv_mae_std']),
        }
        results_to_save['outlier_count'] = int(results['outlier_mask'].sum())
    
    # Coeficientes
    results_to_save['coefficients'] = {}
    for _, row in results['feature_importance'].iterrows():
        results_to_save['coefficients'][row['Feature']] = float(row['Coefficient'])
    
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"✓ Resultados salvos: {filename}")


if __name__ == '__main__':
    main()