"""
Model Training Module
Trains and compares multiple machine learning models
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import config
import joblib

class ModelTrainer:
    """
    Trains multiple ML models and performs cross-validation
    
    Models:
    1. Logistic Regression - Simple, fast, interpretable
    2. Random Forest - Ensemble method, handles non-linear relationships
    3. Support Vector Machine - Powerful for complex decision boundaries
    """
    
    def __init__(self):
        """Initialize model trainer"""
        self.models = {}
        self.trained_models = {}
        self.cv_scores = {}
        
    def initialize_models(self):
        """
        Initialize all models with configured parameters
        
        Logistic Regression:
        - Linear model for binary classification
        - Fast and interpretable
        - Works well when relationship is approximately linear
        
        Random Forest:
        - Ensemble of decision trees
        - Handles non-linear relationships well
        - Robust to outliers
        - Can capture complex patterns
        
        Support Vector Machine (SVM):
        - Finds optimal decision boundary
        - Good for high-dimensional data
        - Kernel='rbf' allows non-linear boundaries
        """
        print("\n" + "=" * 60)
        print("INITIALIZING MACHINE LEARNING MODELS")
        print("=" * 60)
        
        # Logistic Regression
        print("\n1. Logistic Regression")
        print("   └─ How it works: Finds linear relationship between features and disease probability")
        print("   └─ Best for: Simple, interpretable models")
        self.models['Logistic Regression'] = LogisticRegression(
            **config.MODELS_CONFIG['logistic_regression']
        )
        
        # Random Forest
        print("\n2. Random Forest")
        print("   └─ How it works: Creates multiple decision trees and combines their predictions")
        print("   └─ Best for: Complex patterns, non-linear relationships")
        print(f"   └─ Trees: {config.MODELS_CONFIG['random_forest']['n_estimators']}")
        self.models['Random Forest'] = RandomForestClassifier(
            **config.MODELS_CONFIG['random_forest']
        )
        
        # Support Vector Machine
        print("\n3. Support Vector Machine (SVM)")
        print("   └─ How it works: Finds the best boundary that separates disease/no disease")
        print("   └─ Best for: High accuracy on complex datasets")
        print(f"   └─ Kernel: {config.MODELS_CONFIG['svm']['kernel']}")
        self.models['SVM'] = SVC(
            **config.MODELS_CONFIG['svm']
        )
        
        print(f"\n✓ {len(self.models)} models initialized")
        
    def train_single_model(self, name, model, X_train, y_train):
        """
        Train a single model
        
        Parameters:
        -----------
        name : str
            Model name
        model : sklearn model
            Model to train
        X_train : array
            Training features
        y_train : array
            Training labels
            
        Returns:
        --------
        model : Trained model
        """
        print(f"\n{'─' * 60}")
        print(f"Training: {name}")
        print(f"{'─' * 60}")
        
        # Train the model
        print("  └─ Fitting model to training data...")
        model.fit(X_train, y_train)
        
        print(f"  └─ ✓ {name} training complete")
        
        return model
    
    def cross_validate_model(self, name, model, X_train, y_train):
        """
        Perform cross-validation to assess model stability
        
        Cross-validation:
        - Splits training data into k folds (default: 5)
        - Trains on k-1 folds, tests on remaining fold
        - Repeats k times with different test fold
        - Gives average performance and variance
        
        Why it's important:
        - Shows if model performance is consistent
        - Helps detect overfitting
        - More reliable than single train-test split
        
        Parameters:
        -----------
        name : str
            Model name
        model : sklearn model
            Trained model
        X_train : array
            Training features
        y_train : array
            Training labels
            
        Returns:
        --------
        scores : array
            Cross-validation scores
        """
        print(f"  └─ Performing {config.CV_FOLDS}-fold cross-validation...")
        
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=config.CV_FOLDS,
            scoring='accuracy'
        )
        
        mean_score = scores.mean()
        std_score = scores.std()
        
        print(f"  └─ CV Accuracy: {mean_score:.4f} (±{std_score:.4f})")
        print(f"      ├─ This means the model is {mean_score*100:.1f}% accurate on average")
        print(f"      └─ Variance of ±{std_score*100:.1f}% shows consistency across folds")
        
        self.cv_scores[name] = {
            'scores': scores,
            'mean': mean_score,
            'std': std_score
        }
        
        return scores
    
    def train_all_models(self, X_train, y_train):
        """
        Train all models with cross-validation
        
        Parameters:
        -----------
        X_train : array
            Training features (scaled)
        y_train : array
            Training labels
            
        Returns:
        --------
        dict : Dictionary of trained models
        """
        print("\n" + "=" * 60)
        print("MODEL TRAINING PHASE")
        print("=" * 60)
        print(f"\nTraining on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        for name, model in self.models.items():
            # Train model
            trained_model = self.train_single_model(name, model, X_train, y_train)
            self.trained_models[name] = trained_model
            
            # Cross-validate
            self.cross_validate_model(name, trained_model, X_train, y_train)
        
        print("\n" + "=" * 60)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("=" * 60)
        
        # Summary
        print("\nCross-Validation Summary:")
        print("-" * 60)
        for name in self.models.keys():
            cv_info = self.cv_scores[name]
            print(f"{name:25} {cv_info['mean']:.4f} (±{cv_info['std']:.4f})")
        
        return self.trained_models
    
    def get_feature_importance(self, model_name='Random Forest'):
        """
        Get feature importance from Random Forest model
        
        Feature importance tells us which features are most useful
        for making predictions.
        
        Parameters:
        -----------
        model_name : str
            Name of model (must be Random Forest)
            
        Returns:
        --------
        DataFrame : Feature importances
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.trained_models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"{model_name} does not have feature importances")
        
        import pandas as pd
        
        importances = pd.DataFrame({
            'feature': config.ALL_FEATURES,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importances
    
    def save_models(self):
        """Save all trained models"""
        for name, model in self.trained_models.items():
            filename = f"{name.replace(' ', '_').lower()}_model.pkl"
            filepath = config.MODEL_DIR + '/' + filename
            joblib.dump(model, filepath)
        print(f"\n✓ All models saved to {config.MODEL_DIR}")


def main():
    """Test the model trainer"""
    import pandas as pd
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    df = pd.read_csv(config.OUTPUT_FILES['dataset'])
    preprocessor = DataPreprocessor(df)
    preprocessor.validate_data()
    preprocessor.split_data()
    X_train_scaled, X_test_scaled = preprocessor.scale_features()
    _, _, y_train, y_test = preprocessor.get_processed_data()
    
    # Train models
    trainer = ModelTrainer()
    trainer.initialize_models()
    trainer.train_all_models(X_train_scaled, y_train)
    
    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)
    importance_df = trainer.get_feature_importance()
    print(importance_df)
    
    # Save models
    trainer.save_models()


if __name__ == "__main__":
    main()