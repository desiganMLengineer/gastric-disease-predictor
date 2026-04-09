"""
Main Pipeline Script
Runs the complete end-to-end machine learning pipeline
"""

import sys
import os
import joblib
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from data_generator import GastricDataGenerator
from preprocessing import DataPreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from predict import GastricDiseasePredictor


# Toggle pauses (set False for automation)
INTERACTIVE_MODE = False


def wait():
    if INTERACTIVE_MODE:
        input("\nPress ENTER to continue...")


def ensure_directories():
    """Create output directories if they don't exist"""
    for path in config.OUTPUT_FILES.values():
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def print_header(title):
    print("\n\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_complete_pipeline():

    ensure_directories()

    print("=" * 80)
    print(" " * 20 + "GASTRIC DISEASE PREDICTION SYSTEM")
    print("=" * 80)

    # ========================================
    # STAGE 1: DATA GENERATION
    # ========================================
    print_header("STAGE 1: DATA GENERATION")

    generator = GastricDataGenerator(
        n_samples=config.DATASET_CONFIG['n_samples'],
        random_state=config.DATASET_CONFIG['random_state']
    )

    df = generator.generate_dataset(save_path=config.OUTPUT_FILES['dataset'])
    print(f"\n✓ Dataset created: {df.shape}")
    wait()

    # ========================================
    # STAGE 2: EDA
    # ========================================
    print_header("STAGE 2: EXPLORATORY DATA ANALYSIS")

    preprocessor = DataPreprocessor(df)
    preprocessor.validate_data()
    preprocessor.exploratory_analysis(save_path=config.OUTPUT_FILES['eda_viz'])
    print("\n✓ EDA completed")
    wait()

    # ========================================
    # STAGE 3: DATA PREPROCESSING
    # ========================================
    print_header("STAGE 3: DATA PREPROCESSING")

    # Split data
    preprocessor.split_data(test_size=config.DATASET_CONFIG['test_size'])

    # Scale features (no arguments needed)
    X_train_scaled, X_test_scaled = preprocessor.scale_features()
    y_train, y_test = preprocessor.y_train, preprocessor.y_test

    print("\n✓ Data preprocessed")
    wait()

    # ========================================
    # STAGE 4: TRAINING
    # ========================================
    print_header("STAGE 4: MODEL TRAINING")

    trainer = ModelTrainer()
    trainer.initialize_models()
    trained_models = trainer.train_all_models(X_train_scaled, y_train)

    # Feature importance (if available)
    if hasattr(trainer, "get_feature_importance"):
        print("\nTop Features:")
        print(trainer.get_feature_importance().head(10))

    trainer.save_models()
    print(f"\n✓ {len(trained_models)} models trained")
    wait()

    # ========================================
    # STAGE 5: EVALUATION
    # ========================================
    print_header("STAGE 5: MODEL EVALUATION")

    evaluator = ModelEvaluator(trained_models)
    evaluator.evaluate_all_models(X_test_scaled, y_test)
    evaluator.create_visualizations(y_test)
    evaluator.save_results()
    print("\n✓ Models evaluated")
    wait()

    # ========================================
    # STAGE 6: SAVE BEST MODEL
    # ========================================
    print_header("STAGE 6: MODEL SELECTION")

    best_model_name, best_model = evaluator.get_best_model()
    joblib.dump(best_model, config.OUTPUT_FILES['best_model'])
    print(f"\n✓ Best model: {best_model_name}")
    wait()

    # ========================================
    # STAGE 7: PREDICTION DEMO
    # ========================================
    print_header("STAGE 7: PREDICTION DEMO")

    try:
        predictor = GastricDiseasePredictor()

        sample = {
            'age': 45,
            'gender': 1,
            'bmi': 26,
            'smoking': 1,
            'alcohol_consumption': 1,
            'family_history': 0,
            'spicy_food_intake': 7,
            'stress_level': 6,
            'irregular_meals': 1,
            'abdominal_pain': 1,
            'nausea': 0,
            'bloating': 1,
            'heartburn': 1,
            'loss_of_appetite': 0
        }

        result = predictor.predict(sample)
        predictor.explain_prediction(sample, result)

    except Exception as e:
        print(f"Prediction demo skipped: {e}")

    print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    try:
        run_complete_pipeline()
    except KeyboardInterrupt:
        print("\nPipeline interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()