#!/usr/bin/env python3
"""
    Previsão de Congestionamento de Trânsito usando KNN
    
    Autor: Alexandre Marques Tortoza Canoa
    Versão do Python: 3.13.7
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
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

# ===========================================================================
# CONFIGURAÇÕES GLOBAIS - ALTERE CONFORME NECESSÁRIO
# ============================================================================

class ConfigKNN:
    """Centraliza todas as configurações do experimento."""
    
    DATASET_PATH = "../../dataset/preparedData/dataset.csv"
    DELIMITER = ";"
    
    # Estratégia de Validação
    # Opções: 'holdout' ou 'kfold'
    VALIDATION_STRATEGY = 'holdout'
    HOLDOUT_TEST_SIZE = 0.2  # Proporção teste (0.2 = 80/20)
    KFOLD_N_SPLITS = 5  # Número de folds para validação cruzada
    RANDOM_STATE = 42
    
    # Hiperparâmetros do KNN
    N_NEIGHBORS = 5  # Número de vizinhos
    WEIGHTS = 'distance'  # 'uniform' ou 'distance'
    METRIC = 'minkowski'  # Métrica de distância
    
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
    
    #  Visualizações
    PLOT_RESULTS = True
    PLOT_SHAP = True
    
    # Logs
    VERBOSE = True
    SAVE_RESULTS = True
    RESULTS_FILE = f"resultados_knn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


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
                df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
                if ConfigKNN.VERBOSE:
                    print(f"\uf05dDataset carregado: {df.shape}")
                    print(f"  Encoding: {encoding}")
                    print(f"  Colunas: {list(df.columns)}\n")
                return df
            except UnicodeDecodeError:
                continue
    except FileNotFoundError:
        print(f"\uea87 Arquivo não encontrado: {filepath}")
        raise
    
    print(f"\uea87 Não foi possível carregar o arquivo com nenhum encoding")
    raise ValueError("Erro ao carregar arquivo")


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Extrai features temporais das colunas 'data' e 'hora'.
        
        Args:
            df: DataFrame com colunas 'data' e 'hora'
            
        Returns:
            DataFrame com novas colunas de features temporais
    """
    if ConfigKNN.VERBOSE:
        print("\uf017 Extraindo features temporais...")
    
    df['datetime'] = pd.to_datetime(
        df['data'] + ' ' + df['hora'],
        format='%Y-%m-%d %H:%M:%S'
    )
    
    df['hora_numeric'] = df['datetime'].dt.hour
    df['dia_semana'] = df['datetime'].dt.dayofweek  # 0=segunda, 6=domingo
    df['mes'] = df['datetime'].dt.month
    df['dia_mes'] = df['datetime'].dt.day
    
    if ConfigKNN.VERBOSE:
        print("  \uf05d Features: hora_numeric, dia_semana, mes, dia_mes\n")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Codifica features categóricas em numéricas.
        
        Args:
            df: DataFrame com features categóricas
            
        Returns:
            DataFrame com features codificadas
    """
    if ConfigKNN.VERBOSE:
        print("\uf15c Codificando features categóricas...")
    
    encoders = {}
    categorical_cols = ['via_expressa', 'regiao', 'sexo']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
            encoders[col] = le
            
            if ConfigKNN.VERBOSE:
                print(f"  \uf05d{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    if ConfigKNN.VERBOSE:
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
    print("  \ueb03 CARREGAMENTO E PREPARAÇÃO DE DADOS")
    print("="*80)
    
    df = load_data(filepath, delimiter=ConfigKNN.DELIMITER)
    
    print(f"\ueb2b Verificando valores ausentes iniciais:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  {missing[missing > 0].to_dict()}\n")
    else:
        print("  \uf05dNenhum valor ausente\n")
    
    df = extract_datetime_features(df)
    
    df = encode_categorical_features(df)
    
    print("\uebab Agregando dados por data/hora/via/região...")
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
    
    print(f"  \uf05d{len(df)} → {len(df_agg)} linhas\n")
    
    print(f"\ueb2b Valores ausentes após preparação:")
    missing = df_agg.isnull().sum()
    if missing.sum() > 0:
        print(f"  {missing[missing > 0].to_dict()}\n")
    else:
        print("  \uf05d Nenhum valor ausente\n")
    
    print(f"\ueb03 Estatísticas do target (tamanho_congestionamento):")
    print(f"  Mínimo: {df_agg['tamanho_congestionamento'].min()}")
    print(f"  Máximo: {df_agg['tamanho_congestionamento'].max()}")
    print(f"  Média: {df_agg['tamanho_congestionamento'].mean():.2f}")
    print(f"  Mediana: {df_agg['tamanho_congestionamento'].median():.2f}\n")
    
    return df_agg


# ============================================================================
# FUNÇÕES DE TREINAMENTO#
# ============================================================================

def train_knn_holdout(X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """
        Treina KNN com validação holdout (80/20).
        
        Args:
            X_train, X_test: Features de treino/teste (já escaladas)
            y_train, y_test: Target de treino/teste
            
        Returns:
            Dicionário com modelo, métricas e dados
    """
    print(f" Treinando KNN (n_neighbors={ConfigKNN.N_NEIGHBORS})...")
    
    model = KNeighborsRegressor(
        n_neighbors=ConfigKNN.N_NEIGHBORS,
        weights=ConfigKNN.WEIGHTS,
        metric=ConfigKNN.METRIC
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


def train_knn_kfold(X: np.ndarray, y: np.ndarray) -> dict:
    """
        Treina KNN com validação cruzada K-Fold.
        
        Args:
            X: Features (já escaladas)
            y: Target
            
        Returns:
            Dicionário com scores e estatísticas
    """
    print(f" Treinando KNN com {ConfigKNN.KFOLD_N_SPLITS}-Fold CV...")
    
    model = KNeighborsRegressor(
        n_neighbors=ConfigKNN.N_NEIGHBORS,
        weights=ConfigKNN.WEIGHTS,
        metric=ConfigKNN.METRIC
    )
    
    kfold = KFold(n_splits=ConfigKNN.KFOLD_N_SPLITS, 
                  shuffle=True, 
                  random_state=ConfigKNN.RANDOM_STATE)
    
    scores_r2 = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    scores_mae = cross_val_score(model, X, y, cv=kfold, 
                                 scoring='neg_mean_absolute_error')
    
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


def train_model(df: pd.DataFrame) -> dict:
    """
        Pipeline de treinamento adaptado à estratégia de validação.
        
        Args:
            df: DataFrame preparado
            
        Returns:
            Resultados do treinamento
    """
    print("="*80)
    print("   TREINAMENTO DO MODELO")
    print("="*80)
    
    missing_features = [f for f in ConfigKNN.FEATURES if f not in df.columns]
    if missing_features:
        print(f"\uea87 Features não encontradas: {missing_features}")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        raise ValueError("Features faltando no dataset")
    
    df_clean = df.dropna(subset=ConfigKNN.FEATURES + [ConfigKNN.TARGET])
    print(f"\uf05d Dados limpos: {len(df)} → {len(df_clean)} linhas\n")
    
    X = df_clean[ConfigKNN.FEATURES].values
    y = df_clean[ConfigKNN.TARGET].values
    
    print(" Normalizando features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  \uf05d Média: {X_scaled.mean(axis=0)}")
    print(f"  \uf05d Std: {X_scaled.std(axis=0)}\n")
    
    if ConfigKNN.VALIDATION_STRATEGY == 'holdout':
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=ConfigKNN.HOLDOUT_TEST_SIZE,
            random_state=ConfigKNN.RANDOM_STATE
        )
        print(f"\uf05d Split: {len(X_train)} treino | {len(X_test)} teste\n")
        
        results = train_knn_holdout(X_train, X_test, y_train, y_test)
        
    elif ConfigKNN.VALIDATION_STRATEGY == 'kfold':
        results = train_knn_kfold(X_scaled, y)
    
    else:
        raise ValueError(f"Estratégia desconhecida: {ConfigKNN.VALIDATION_STRATEGY}")
    
    results['scaler'] = scaler
    results['features'] = ConfigKNN.FEATURES
    results['df'] = df_clean
    
    return results


def print_metrics(results: dict) -> None:
    """Imprime métricas de forma formatada."""
    print("\n" + "="*80)
    print("  \ueb03 MÉTRICAS DO MODELO")
    print("="*80)
    
    if ConfigKNN.VALIDATION_STRATEGY == 'holdout':
        metrics = results['metrics']
        print(f"\n{'Métrica':<25} {'Treino':>12} {'Teste':>12}")
        print("-" * 50)
        print(f"{'MAE':<25} {metrics['train_mae']:>12.4f} {metrics['test_mae']:>12.4f}")
        print(f"{'RMSE':<25} {metrics['train_rmse']:>12.4f} {metrics['test_rmse']:>12.4f}")
        print(f"{'R² Score':<25} {metrics['train_r2']:>12.4f} {metrics['test_r2']:>12.4f}")
        
    elif ConfigKNN.VALIDATION_STRATEGY == 'kfold':
        print(f"\n5-Fold Cross Validation:")
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
    if not ConfigKNN.PLOT_RESULTS:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
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
    
    ax = axes[1, 0]
    errors = results['y_test'] - results['y_test_pred']
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Erro de Previsão')
    ax.set_ylabel('Frequência')
    ax.set_title(f'Distribuição de Erros (Teste) - Média: {errors.mean():.2f}')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    residuals = results['y_test'] - results['y_test_pred']
    ax.scatter(results['y_test_pred'], residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Valores Previstos')
    ax.set_ylabel('Residuais')
    ax.set_title('Análise de Resíduos (Teste)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('knn_results_holdout.png', dpi=150, bbox_inches='tight')
    print("\uf05dGráfico salvo: knn_results_holdout.png")
    plt.show()


def plot_results_kfold(results: dict) -> None:
    """Plota resultados para validação K-Fold."""
    if not ConfigKNN.PLOT_RESULTS:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
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
    ax.set_title('Scores R² por Fold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    ax = axes[1]
    ax.bar(folds, results['cv_mae_scores'], alpha=0.7, color='lightcoral', edgecolor='black')
    ax.axhline(y=results['cv_mae_mean'], color='r', linestyle='--', linewidth=2, label='Média')
    ax.fill_between(folds,
                     results['cv_mae_mean'] - results['cv_mae_std'],
                     results['cv_mae_mean'] + results['cv_mae_std'],
                     alpha=0.2, color='r')
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE')
    ax.set_title('Scores MAE por Fold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('knn_results_kfold.png', dpi=150, bbox_inches='tight')
    print("\uf05dGráfico salvo: knn_results_kfold.png")
    plt.show()

# ============================================================================
# FUNÇÕES SHAP
# ============================================================================

def explain_with_shap(results: dict) -> None:
    """Análise de explicabilidade com SHAP (holdout only)."""
    if not ConfigKNN.SHAP_ENABLED or not HAS_SHAP:
        if ConfigKNN.VERBOSE:
            print("\uea6c  SHAP desabilitado ou não disponível")
        return
    
    if ConfigKNN.VALIDATION_STRATEGY != 'holdout':
        print("\uea6c  SHAP disponível apenas para validação holdout")
        return
    
    print("\n" + "="*80)
    print("   ANÁLISE SHAP")
    print("="*80)
    print(f"Executando SHAP com {ConfigKNN.SHAP_N_SAMPLES} amostras...")
    
    n_samples = min(ConfigKNN.SHAP_N_SAMPLES, len(results['X_train']))
    X_sample = results['X_train'][:n_samples]
    X_test_sample = results['X_test'][:min(100, len(results['X_test']))]
    
    explainer = shap.KernelExplainer(results['model'].predict, X_sample)
    shap_values = explainer.shap_values(X_test_sample)
    
    print("\uf05dSHAP values calculados\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    plt.sca(axes[0, 0])
    shap.summary_plot(shap_values, X_test_sample, 
                      feature_names=results['features'], show=False)
    axes[0, 0].set_title('Importância Global (SHAP)', fontweight='bold', fontsize=12)
    
    plt.sca(axes[0, 1])
    shap.summary_plot(shap_values, X_test_sample, 
                      feature_names=results['features'], 
                      plot_type="bar", show=False)
    axes[0, 1].set_title('Importância Média Absoluta', fontweight='bold', fontsize=12)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argmax(mean_abs_shap)
    
    plt.sca(axes[1, 0])
    shap.dependence_plot(top_idx, shap_values, X_test_sample,
                         feature_names=results['features'], ax=axes[1, 0], show=False)
    
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
    plt.savefig('knn_shap_analysis.png', dpi=150, bbox_inches='tight')
    print("\uf05dGráfico salvo: knn_shap_analysis.png\n")
    plt.show()

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Pipeline principal."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  PREVISÃO DE CONGESTIONAMENTO COM KNN - PIPELINE MODULAR".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print(f"\n\ued7b CONFIGURAÇÕES:")
    print(f"   Dataset: {ConfigKNN.DATASET_PATH}")
    print(f"   Features: {ConfigKNN.FEATURES}")
    print(f"   Target: {ConfigKNN.TARGET}")
    print(f"   Validação: {ConfigKNN.VALIDATION_STRATEGY.upper()}")
    if ConfigKNN.VALIDATION_STRATEGY == 'holdout':
        print(f"   Test Size: {ConfigKNN.HOLDOUT_TEST_SIZE}")
    else:
        print(f"   N-Folds: {ConfigKNN.KFOLD_N_SPLITS}")
    print(f"   N-Neighbors: {ConfigKNN.N_NEIGHBORS}")
    print(f"   SHAP: {'Habilitado' if ConfigKNN.SHAP_ENABLED else 'Desabilitado'}\n")
    
    try:
        df = prepare_data(ConfigKNN.DATASET_PATH)
        
        results = train_model(df)
        
        print_metrics(results)
        
        if ConfigKNN.VALIDATION_STRATEGY == 'holdout':
            plot_results_holdout(results)
        else:
            plot_results_kfold(results)
        
        explain_with_shap(results)
        
        if ConfigKNN.SAVE_RESULTS:
            save_results(results, ConfigKNN.RESULTS_FILE)
        
        print("\n" + "="*80)
        print("  \uf058 PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n\uea87 ERRO: {str(e)}")
        raise


def save_results(results: dict, filename: str) -> None:
    """Salva resultados em JSON."""
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'strategy': ConfigKNN.VALIDATION_STRATEGY,
            'n_neighbors': ConfigKNN.N_NEIGHBORS,
            'features': ConfigKNN.FEATURES,
        }
    }
    
    if ConfigKNN.VALIDATION_STRATEGY == 'holdout':
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
    
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\uf05d Resultados salvos: {filename}")


if __name__ == '__main__':
    main()    