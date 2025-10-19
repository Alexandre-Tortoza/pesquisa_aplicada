#!/usr/bin/env python3
"""
Previsão de Congestionamento de Trânsito usando Random Forest Regressor

Pipeline modular para prever tamanho_congestionamento baseado em população
e outras features. Suporta validação cruzada, holdout e análise SHAP.

Dataset: ../../dataset/preparedData/dataset.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import json
from datetime import datetime

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  Para usar SHAP: pip install shap")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES GLOBAIS - ALTERE CONFORME NECESSÁRIO
# ============================================================================

class ConfigRandomForest:
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
    
    # 🌲 Hiperparâmetros do Random Forest
    N_ESTIMATORS = 100  # Número de árvores (maior = melhor mas mais lento)
    MAX_DEPTH = None  # Profundidade máxima (None = sem limite)
    MIN_SAMPLES_SPLIT = 2  # Mínimo de amostras para dividir nó
    MIN_SAMPLES_LEAF = 1  # Mínimo de amostras em folha
    MAX_FEATURES = 'sqrt'  # Recurso ao dividir ('sqrt', 'log2' ou None)
    BOOTSTRAP = True  # Usar bootstrap
    N_JOBS = -1  # Usar todos os cores (-1 = all cores)
    
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
    PLOT_FEATURE_IMPORTANCE = True  # Plot importância das features
    
    # 📝 Logs
    VERBOSE = True
    SAVE_RESULTS = True
    RESULTS_FILE = f"resultados_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


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
                if ConfigRandomForest.VERBOSE:
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
    if ConfigRandomForest.VERBOSE:
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
    
    if ConfigRandomForest.VERBOSE:
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
    if ConfigRandomForest.VERBOSE:
        print("🔤 Codificando features categóricas...")
    
    encoders = {}
    categorical_cols = ['via_expressa', 'regiao', 'sexo']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
            encoders[col] = le
            
            if ConfigRandomForest.VERBOSE:
                print(f"  ✓ {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    if ConfigRandomForest.VERBOSE:
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
    df = load_data(filepath, delimiter=ConfigRandomForest.DELIMITER)
    
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
    # Nota: seu dataset tem múltiplas linhas por hora (Homens/Mulheres)
    # Agregamos por data/hora/via/região
    print("🔀 Agregando dados por data/hora/via/região...")
    df_agg = df.groupby(['data', 'hora', 'via_expressa', 'regiao']).agg({
        'pop_total': 'sum',  # Soma população
        'tamanho_congestionamento': 'first',  # Congestionamento é igual
        'hora_numeric': 'first',
        'dia_semana': 'first',
        'mes': 'first',
        'via_expressa_encoded': 'first',
        'regiao_encoded': 'first',
        'sexo_encoded': 'mean',  # Média de sexo (para representar proporção)
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
    print(f"  Mediana: {df_agg['tamanho_congestionamento'].median():.2f}\n")
    
    return df_agg


# ============================================================================
# FUNÇÕES DE TREINAMENTO
# ============================================================================

def train_rf_holdout(X_train: np.ndarray, X_test: np.ndarray, 
                     y_train: np.ndarray, y_test: np.ndarray,
                     feature_names: list) -> dict:
    """
    Treina Random Forest com validação holdout (80/20).
    
    Nota: Random Forest não requer normalização de features!
    
    Args:
        X_train, X_test: Features de treino/teste (sem escala necessária)
        y_train, y_test: Target de treino/teste
        feature_names: Nomes das features
        
    Returns:
        Dicionário com modelo, métricas e dados
    """
    print(f"🌲 Treinando Random Forest (n_estimators={ConfigRandomForest.N_ESTIMATORS})...")
    
    # Treina modelo
    model = RandomForestRegressor(
        n_estimators=ConfigRandomForest.N_ESTIMATORS,
        max_depth=ConfigRandomForest.MAX_DEPTH,
        min_samples_split=ConfigRandomForest.MIN_SAMPLES_SPLIT,
        min_samples_leaf=ConfigRandomForest.MIN_SAMPLES_LEAF,
        max_features=ConfigRandomForest.MAX_FEATURES,
        bootstrap=ConfigRandomForest.BOOTSTRAP,
        n_jobs=ConfigRandomForest.N_JOBS,
        random_state=ConfigRandomForest.RANDOM_STATE,
        verbose=0
    )
    model.fit(X_train, y_train)
    
    # Predições
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
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
    }


def train_rf_kfold(X: np.ndarray, y: np.ndarray, 
                   feature_names: list) -> dict:
    """
    Treina Random Forest com validação cruzada K-Fold.
    
    Args:
        X: Features (sem escala necessária)
        y: Target
        feature_names: Nomes das features
        
    Returns:
        Dicionário com scores e estatísticas
    """
    print(f"🌲 Treinando Random Forest com {ConfigRandomForest.KFOLD_N_SPLITS}-Fold CV...")
    
    model = RandomForestRegressor(
        n_estimators=ConfigRandomForest.N_ESTIMATORS,
        max_depth=ConfigRandomForest.MAX_DEPTH,
        min_samples_split=ConfigRandomForest.MIN_SAMPLES_SPLIT,
        min_samples_leaf=ConfigRandomForest.MIN_SAMPLES_LEAF,
        max_features=ConfigRandomForest.MAX_FEATURES,
        bootstrap=ConfigRandomForest.BOOTSTRAP,
        n_jobs=ConfigRandomForest.N_JOBS,
        random_state=ConfigRandomForest.RANDOM_STATE,
        verbose=0
    )
    
    kfold = KFold(n_splits=ConfigRandomForest.KFOLD_N_SPLITS, 
                  shuffle=True, 
                  random_state=ConfigRandomForest.RANDOM_STATE)
    
    # Calcula scores
    scores_r2 = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    scores_mae = cross_val_score(model, X, y, cv=kfold, 
                                 scoring='neg_mean_absolute_error')
    
    # Treina modelo final para feature importance
    model.fit(X, y)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'X': X,
        'y': y,
        'cv_r2_scores': scores_r2,
        'cv_mae_scores': -scores_mae,  # Nega porque sklearn retorna negativo
        'cv_r2_mean': scores_r2.mean(),
        'cv_r2_std': scores_r2.std(),
        'cv_mae_mean': -scores_mae.mean(),
        'cv_mae_std': scores_mae.std(),
        'feature_importance': feature_importance,
    }


def train_model(df: pd.DataFrame) -> dict:
    """
    Pipeline de treinamento adaptado à estratégia de validação.
    
    Nota: Random Forest não necessita de normalização!
    
    Args:
        df: DataFrame preparado
        
    Returns:
        Resultados do treinamento
    """
    print("="*80)
    print("  🌲 TREINAMENTO DO MODELO - RANDOM FOREST")
    print("="*80)
    
    # Verifica features disponíveis
    missing_features = [f for f in ConfigRandomForest.FEATURES if f not in df.columns]
    if missing_features:
        print(f"❌ Features não encontradas: {missing_features}")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        raise ValueError("Features faltando no dataset")
    
    # Remove NaN
    df_clean = df.dropna(subset=ConfigRandomForest.FEATURES + [ConfigRandomForest.TARGET])
    print(f"✓ Dados limpos: {len(df)} → {len(df_clean)} linhas")
    print(f"  Proporção mantida: {len(df_clean)/len(df)*100:.2f}%\n")
    
    # Separa features e target
    X = df_clean[ConfigRandomForest.FEATURES].values
    y = df_clean[ConfigRandomForest.TARGET].values
    
    print("⚠️  Random Forest não necessita normalização (trabalha com árvores)\n")
    
    # Treina conforme estratégia
    if ConfigRandomForest.VALIDATION_STRATEGY == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=ConfigRandomForest.HOLDOUT_TEST_SIZE,
            random_state=ConfigRandomForest.RANDOM_STATE
        )
        print(f"✓ Split: {len(X_train)} treino | {len(X_test)} teste\n")
        
        results = train_rf_holdout(X_train, X_test, y_train, y_test, 
                                   ConfigRandomForest.FEATURES)
        
    elif ConfigRandomForest.VALIDATION_STRATEGY == 'kfold':
        results = train_rf_kfold(X, y, ConfigRandomForest.FEATURES)
    
    else:
        raise ValueError(f"Estratégia desconhecida: {ConfigRandomForest.VALIDATION_STRATEGY}")
    
    # Adiciona informações adicionais
    results['features'] = ConfigRandomForest.FEATURES
    results['df'] = df_clean
    
    return results


def print_metrics(results: dict) -> None:
    """Imprime métricas de forma formatada."""
    print("\n" + "="*80)
    print("  📊 MÉTRICAS DO MODELO - RANDOM FOREST")
    print("="*80)
    
    if ConfigRandomForest.VALIDATION_STRATEGY == 'holdout':
        metrics = results['metrics']
        print(f"\n{'Métrica':<25} {'Treino':>12} {'Teste':>12}")
        print("-" * 50)
        print(f"{'MAE':<25} {metrics['train_mae']:>12.4f} {metrics['test_mae']:>12.4f}")
        print(f"{'RMSE':<25} {metrics['train_rmse']:>12.4f} {metrics['test_rmse']:>12.4f}")
        print(f"{'R² Score':<25} {metrics['train_r2']:>12.4f} {metrics['test_r2']:>12.4f}")
        
    elif ConfigRandomForest.VALIDATION_STRATEGY == 'kfold':
        print(f"\n{ConfigRandomForest.KFOLD_N_SPLITS}-Fold Cross Validation:")
        print("-" * 50)
        print(f"R² Scores: {results['cv_r2_scores']}")
        print(f"R² Média: {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")
        print(f"MAE Média: {results['cv_mae_mean']:.4f} (+/- {results['cv_mae_std']:.4f})")
    
    print(f"\n{'Feature Importance':<25}")
    print("-" * 50)
    for idx, row in results['feature_importance'].head(7).iterrows():
        bar = "█" * int(row['Importance'] * 50)
        print(f"{row['Feature']:<25} {bar} {row['Importance']:.4f}")
    
    print()


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================================

def plot_results_holdout(results: dict) -> None:
    """Plota resultados para validação holdout."""
    if not ConfigRandomForest.PLOT_RESULTS:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Real vs Previsto (Teste)
    ax = axes[0, 0]
    ax.scatter(results['y_test'], results['y_test_pred'], 
               alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    min_val = min(results['y_test'].min(), results['y_test_pred'].min())
    max_val = max(results['y_test'].max(), results['y_test_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Previstos')
    ax.set_title('Teste: Real vs Previsto')
    ax.grid(True, alpha=0.3)
    textstr = f"R² = {results['metrics']['test_r2']:.4f}\nMAE = {results['metrics']['test_mae']:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Gráfico 2: Real vs Previsto (Treino)
    ax = axes[0, 1]
    ax.scatter(results['y_train'], results['y_train_pred'], 
               alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    min_val = min(results['y_train'].min(), results['y_train_pred'].min())
    max_val = max(results['y_train'].max(), results['y_train_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Previstos')
    ax.set_title('Treino: Real vs Previsto')
    ax.grid(True, alpha=0.3)
    textstr = f"R² = {results['metrics']['train_r2']:.4f}\nMAE = {results['metrics']['train_mae']:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Gráfico 3: Distribuição de erros (Teste)
    ax = axes[1, 0]
    errors = results['y_test'] - results['y_test_pred']
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Erro de Previsão')
    ax.set_ylabel('Frequência')
    ax.set_title(f'Distribuição de Erros (Teste) - Média: {errors.mean():.2f}')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Resíduos vs Previstos
    ax = axes[1, 1]
    residuals = results['y_test'] - results['y_test_pred']
    ax.scatter(results['y_test_pred'], residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Valores Previstos')
    ax.set_ylabel('Residuais')
    ax.set_title('Análise de Resíduos (Teste)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rf_results_holdout.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo: rf_results_holdout.png")
    plt.show()


def plot_results_kfold(results: dict) -> None:
    """Plota resultados para validação K-Fold."""
    if not ConfigRandomForest.PLOT_RESULTS:
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
    ax.set_title('Scores R² por Fold - Random Forest')
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
    ax.set_title('Scores MAE por Fold - Random Forest')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('rf_results_kfold.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo: rf_results_kfold.png")
    plt.show()


def plot_feature_importance(results: dict) -> None:
    """Plota importância das features."""
    if not ConfigRandomForest.PLOT_FEATURE_IMPORTANCE:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    importance_df = results['feature_importance'].head(10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black')
    ax.set_xlabel('Importância (Gini)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Adiciona valores nas barras
    for i, (feature, importance) in enumerate(zip(importance_df['Feature'], 
                                                   importance_df['Importance'])):
        ax.text(importance, i, f' {importance:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo: rf_feature_importance.png")
    plt.show()


# ============================================================================
# FUNÇÕES SHAP
# ============================================================================

def explain_with_shap(results: dict) -> None:
    """Análise de explicabilidade com SHAP (holdout only)."""
    if not ConfigRandomForest.SHAP_ENABLED or not HAS_SHAP:
        if ConfigRandomForest.VERBOSE:
            print("⚠️  SHAP desabilitado ou não disponível")
        return
    
    if ConfigRandomForest.VALIDATION_STRATEGY != 'holdout':
        print("⚠️  SHAP disponível apenas para validação holdout")
        return
    
    print("\n" + "="*80)
    print("  🔍 ANÁLISE SHAP - RANDOM FOREST")
    print("="*80)
    print(f"Executando SHAP com {ConfigRandomForest.SHAP_N_SAMPLES} amostras...")
    
    # Usa TreeExplainer para Random Forest (muito mais rápido)
    explainer = shap.TreeExplainer(results['model'])
    
    # Calcula SHAP values para amostra do teste
    n_samples = min(ConfigRandomForest.SHAP_N_SAMPLES, len(results['X_test']))
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
    plt.savefig('rf_shap_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico salvo: rf_shap_analysis.png\n")
    plt.show()


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Pipeline principal."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  PREVISÃO DE CONGESTIONAMENTO COM RANDOM FOREST - PIPELINE MODULAR".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print(f"\n📋 CONFIGURAÇÕES:")
    print(f"   Dataset: {ConfigRandomForest.DATASET_PATH}")
    print(f"   Features: {ConfigRandomForest.FEATURES}")
    print(f"   Target: {ConfigRandomForest.TARGET}")
    print(f"   Validação: {ConfigRandomForest.VALIDATION_STRATEGY.upper()}")
    if ConfigRandomForest.VALIDATION_STRATEGY == 'holdout':
        print(f"   Test Size: {ConfigRandomForest.HOLDOUT_TEST_SIZE}")
    else:
        print(f"   N-Folds: {ConfigRandomForest.KFOLD_N_SPLITS}")
    print(f"   N-Estimators: {ConfigRandomForest.N_ESTIMATORS}")
    print(f"   Max-Depth: {ConfigRandomForest.MAX_DEPTH}")
    print(f"   SHAP: {'Habilitado' if ConfigRandomForest.SHAP_ENABLED else 'Desabilitado'}\n")
    
    try:
        # 1. Preparação
        df = prepare_data(ConfigRandomForest.DATASET_PATH)
        
        # 2. Treinamento
        results = train_model(df)
        
        # 3. Métricas
        print_metrics(results)
        
        # 4. Visualizações
        if ConfigRandomForest.VALIDATION_STRATEGY == 'holdout':
            plot_results_holdout(results)
            plot_feature_importance(results)
        else:
            plot_results_kfold(results)
            plot_feature_importance(results)
        
        # 5. SHAP
        explain_with_shap(results)
        
        # 6. Salva resultados
        if ConfigRandomForest.SAVE_RESULTS:
            save_results(results, ConfigRandomForest.RESULTS_FILE)
        
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
        'algorithm': 'Random Forest Regressor',
        'configuration': {
            'strategy': ConfigRandomForest.VALIDATION_STRATEGY,
            'n_estimators': ConfigRandomForest.N_ESTIMATORS,
            'max_depth': ConfigRandomForest.MAX_DEPTH,
            'features': ConfigRandomForest.FEATURES,
        }
    }
    
    if ConfigRandomForest.VALIDATION_STRATEGY == 'holdout':
        results_to_save['metrics'] = {
            k: float(v) for k, v in results['metrics'].items()
        }
    else:
        results_to_save['cv_scores'] = {
            'r2_mean': float(results['cv_r2_mean']),
            'r2_std': float(results['cv_r2_std']),
            'mae_mean': float(results['cv_mae_mean']),
            'mae_std': float(results['cv_mae_std']),
        }
    
    # Feature importance
    results_to_save['feature_importance'] = {}
    for _, row in results['feature_importance'].iterrows():
        results_to_save['feature_importance'][row['Feature']] = float(row['Importance'])
    
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"✓ Resultados salvos: {filename}")


if __name__ == '__main__':
    main()