"""
Configuration file for Gastric Disease Prediction Project
Centralized configuration for easy modifications
"""

import os

# ============================================
# PROJECT PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
VIZ_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, VIZ_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================
# DATA GENERATION PARAMETERS
# ============================================
DATASET_CONFIG = {
    'n_samples': 1000,  # Number of patients in dataset
    'random_state': 42,  # For reproducibility
    'noise_level': 0.1,  # Amount of noise to add (0.0 to 1.0)
    'test_size': 0.2,  # Train-test split ratio
}

# Feature definitions
FEATURES = {
    'demographic': ['age', 'gender', 'bmi'],
    'lifestyle': ['smoking', 'alcohol_consumption', 'spicy_food_intake', 
                  'stress_level', 'irregular_meals', 'family_history'],
    'symptoms': ['abdominal_pain', 'nausea', 'bloating', 'heartburn', 'loss_of_appetite']
}

# All features combined
ALL_FEATURES = (FEATURES['demographic'] + 
                FEATURES['lifestyle'] + 
                FEATURES['symptoms'])

TARGET_VARIABLE = 'gastric_disease'

# ============================================
# MODEL PARAMETERS
# ============================================
MODELS_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'lbfgs'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'probability': True,
        'random_state': 42
    }
}

# Cross-validation folds
CV_FOLDS = 5

# ============================================
# EVALUATION METRICS
# ============================================
METRICS = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
PRIMARY_METRIC = 'roc_auc'  # Metric used for model selection

# ============================================
# VISUALIZATION SETTINGS
# ============================================
VIZ_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'Set2',
    'font_size': 12
}

# ============================================
# RISK SCORING WEIGHTS
# ============================================
# These weights determine how each feature contributes to disease risk
RISK_WEIGHTS = {
    'age_threshold': 50,
    'age_weight': 0.15,
    'smoking_weight': 0.20,
    'heavy_alcohol_weight': 0.15,
    'family_history_weight': 0.20,
    'spicy_food_threshold': 7,
    'spicy_food_weight': 0.10,
    'stress_threshold': 7,
    'stress_weight': 0.10,
    'irregular_meals_weight': 0.10,
    'abdominal_pain_weight': 0.15,
    'nausea_weight': 0.10,
    'bloating_weight': 0.10,
    'heartburn_weight': 0.15,
    'loss_of_appetite_weight': 0.10
}

# ============================================
# OUTPUT FILES
# ============================================
OUTPUT_FILES = {
    'dataset': os.path.join(DATA_DIR, 'synthetic_gastric_data.csv'),
    'eda_viz': os.path.join(VIZ_DIR, 'eda_analysis.png'),
    'correlation_heatmap': os.path.join(VIZ_DIR, 'correlation_heatmap.png'),
    'model_comparison': os.path.join(VIZ_DIR, 'model_comparison.png'),
    'confusion_matrices': os.path.join(VIZ_DIR, 'confusion_matrices.png'),
    'roc_curves': os.path.join(VIZ_DIR, 'roc_curves.png'),
    'feature_importance': os.path.join(VIZ_DIR, 'feature_importance.png'),
    'best_model': os.path.join(MODEL_DIR, 'best_model.pkl'),
    'scaler': os.path.join(MODEL_DIR, 'scaler.pkl'),
    'results': os.path.join(OUTPUT_DIR, 'model_results.json')
}

# ============================================
# LOGGING
# ============================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

print("✓ Configuration loaded successfully")