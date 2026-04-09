"""
Model Evaluation Module
Evaluates and compares model performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import config
import json

class ModelEvaluator:
    """
    Evaluates trained models and creates comparison visualizations
    """
    
    def __init__(self, trained_models):
        """
        Initialize evaluator with trained models
        
        Parameters:
        -----------
        trained_models : dict
            Dictionary of trained models
        """
        self.models = trained_models
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        
    def evaluate_single_model(self, name, model, X_test, y_test):
        """Evaluate a single model on test data"""
        print(f"\n{'─' * 60}")
        print(f"Evaluating: {name}")
        print(f"{'─' * 60}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nPerformance Metrics:")
        print(f"  ├─ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  ├─ Precision: {precision:.4f}")
        print(f"  ├─ Recall:    {recall:.4f}")
        print(f"  ├─ F1-Score:  {f1:.4f}")
        print(f"  └─ ROC-AUC:   {roc_auc:.4f}")
        
        # Store results
        self.results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Disease', 'Disease'],
                                   zero_division=0))
        
        return self.results[name]
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION PHASE")
        print("=" * 60)
        print(f"\nEvaluating on {X_test.shape[0]} test samples")
        
        for name, model in self.models.items():
            self.evaluate_single_model(name, model, X_test, y_test)
        
        self._select_best_model()
        
    def _select_best_model(self):
        """Select best model based on ROC-AUC"""
        print("\n" + "=" * 60)
        print("MODEL SELECTION")
        print("=" * 60)
        
        best_name = max(self.results, key=lambda x: self.results[x][config.PRIMARY_METRIC])
        best_score = self.results[best_name][config.PRIMARY_METRIC]
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\n🏆 BEST MODEL: {best_name}")
        print(f"   └─ {config.PRIMARY_METRIC.upper()}: {best_score:.4f}")
        
        print("\n" + "-" * 60)
        print("COMPARISON TABLE")
        print("-" * 60)
        
        comparison_df = pd.DataFrame(self.results).T[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        comparison_df = comparison_df.round(4)
        print(comparison_df.to_string())
        
        return self.best_model_name, self.best_model
    
    def create_visualizations(self, y_test):
        """Create comprehensive evaluation visualizations"""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        self._plot_model_comparison()
        self._plot_confusion_matrices(y_test)
        self._plot_roc_curves(y_test)
        self._plot_feature_importance()
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        models = list(self.results.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Plot 1: All metrics
        ax1 = axes[0]
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            values = [self.results[model][metric] for model in models]
            offset = (i - 2) * width
            bars = ax1.bar(x + offset, values, width, label=metric.upper())
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15, ha='right')
        ax1.legend(loc='lower right')
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: ROC-AUC focus
        ax2 = axes[1]
        roc_scores = [self.results[model]['roc_auc'] for model in models]
        colors = ['gold' if model == self.best_model_name else 'skyblue' for model in models]
        
        bars = ax2.bar(models, roc_scores, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
        ax2.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(models, rotation=15, ha='right')
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis='y', alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars, roc_scores)):
            height = bar.get_height()
            label = f'{score:.4f}'
            if models[i] == self.best_model_name:
                label += '\n🏆'
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        save_path = config.OUTPUT_FILES['model_comparison']
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
        print(f"✓ Model comparison saved to: {save_path}")
        plt.close()
    
    def _plot_confusion_matrices(self, y_test):
        """Plot confusion matrices"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            
            axes[idx].set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}',
                              fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=11)
            axes[idx].set_xlabel('Predicted', fontsize=11)
            axes[idx].set_xticklabels(['No Disease', 'Disease'])
            axes[idx].set_yticklabels(['No Disease', 'Disease'])
            
            if name == self.best_model_name:
                for spine in axes[idx].spines.values():
                    spine.set_edgecolor('gold')
                    spine.set_linewidth(3)
        
        plt.tight_layout()
        save_path = config.OUTPUT_FILES['confusion_matrices']
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
        print(f"✓ Confusion matrices saved to: {save_path}")
        plt.close()
    
    def _plot_roc_curves(self, y_test):
        """Plot ROC curves"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for idx, (name, results) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            roc_auc = results['roc_auc']
            
            linestyle = '-' if name == self.best_model_name else '--'
            linewidth = 3 if name == self.best_model_name else 2
            label = f"{name} (AUC = {roc_auc:.3f})"
            if name == self.best_model_name:
                label += " 🏆"
            
            plt.plot(fpr, tpr, color=colors[idx % len(colors)],
                    linestyle=linestyle, linewidth=linewidth, label=label)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
        
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = config.OUTPUT_FILES['roc_curves']
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
        print(f"✓ ROC curves saved to: {save_path}")
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance"""
        if 'Random Forest' not in self.models:
            return
        
        model = self.models['Random Forest']
        
        if not hasattr(model, 'feature_importances_'):
            return
        
        importances = pd.DataFrame({
            'feature': config.ALL_FEATURES,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(importances['importance'] / importances['importance'].max())
        
        plt.barh(importances['feature'], importances['importance'], color=colors, edgecolor='black')
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = config.OUTPUT_FILES['feature_importance']
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
        print(f"✓ Feature importance saved to: {save_path}")
        plt.close()
    
    def save_results(self):
        """Save results to JSON"""
        results_to_save = {}
        for name, metrics in self.results.items():
            results_to_save[name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'roc_auc': float(metrics['roc_auc'])
            }
        
        results_to_save['best_model'] = self.best_model_name
        
        save_path = config.OUTPUT_FILES['results']
        with open(save_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        print(f"\n✓ Results saved to: {save_path}")
    
    def get_best_model(self):
        """Get best model"""
        return self.best_model_name, self.best_model