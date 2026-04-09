
🏥 Gastric Disease Prediction System
[[Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[[scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
A complete machine learning system to predict gastric disease risk using patient demographics, lifestyle habits, and symptoms. This project includes data generation, EDA, model training, evaluation, and a production-ready prediction interface.


                                📋 Table of Contents

Overview
Features
Project Structure
Installation
Usage
Dataset
Machine Learning Pipeline
Model Performance
Visualizations
API Reference
Results
Future Enhancements
Contributing
Project Statistics
Learning Outcomes
License
Author



🎯 Overview :

The Gastric Disease Prediction System is an end-to-end ML solution that can:

Generate realistic synthetic patient datasets
Perform exploratory data analysis (EDA) with visualizations
Train and compare multiple ML algorithms
Select the best performing model
Provide risk assessment for new patients

Why it matters:
-Gastric diseases affect millions worldwide
-Early detection improves treatment outcomes
-ML can identify high-risk patients for preventive care
-Reduces healthcare costs through early intervention

✨ Features :

 🔬 Machine Learning
- Multiple Algorithms: Logistic Regression, Random Forest, SVM
- Cross-Validation: 5-fold CV for robust performance estimation
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Feature Importance: Identifies key risk factors

📊 Data Analysis
-Automated EDA with statistical summaries and plots
-Correlation heatmaps to understand feature relationships
-Class-balanced synthetic dataset generation
-Input validation and error handling

🎨 Visualizations
- Model comparison charts
- Confusion matrices
- ROC curves
- Feature importance plots
- EDA visualizations

🚀 Production Ready
- Modular, reusable code structure
- Comprehensive error handling
- Detailed logging and documentation
- Easy-to-use prediction interface



📁 Project Structure :
gastric-disease-prediction/
│
├── data/
│   └── synthetic_gastric_data.csv        # Generated dataset
│
├── outputs/
│   ├── visualizations/
│   │   ├── eda_analysis.png              # EDA plots
│   │   ├── correlation_heatmap.png       # Feature correlations
│   │   ├── model_comparison.png          # Model metrics comparison
│   │   ├── confusion_matrices.png        # Confusion matrices
│   │   ├── roc_curves.png                # ROC curves
│   │   └── feature_importance.png        # Feature importance
│   │
│   ├── models/
│   │   ├── best_model.pkl                # Best model
│   │   ├── scaler.pkl                    # Scaler object
│   │   └── [model_name]_model.pkl        # Individual models
│   │
│   └── model_results.json                # Performance metrics
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py                 # Synthetic data creation
│   ├── preprocessing.py                  # Data preprocessing & EDA
│   ├── model_trainer.py                  # ML model training
│   ├── evaluator.py                      # Model evaluation
│   └── predictor.py                      # Prediction module
│
├── config.py                             # Parameters & configuration
├── main.py                               # Full pipeline script
├── requirements.txt                      # Dependencies
├── README.md                             # Project overview
├── LICENSE                               # MIT License
└── .gitignore                            # Ignore rules

🔧 Installation :

# Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

# Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/gastric-disease-prediction.git
cd gastric-disease-prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

🚀 Usage :

 Quick Start - Run full Pipeline

```bash
python main.py
```

steps executed the entire pipeline:
1. Generates synthetic dataset (1000 patients)
2. Performs EDA
3. Preprocesses data
4. Trains all models
5. Evaluates and compares models
6. Saves best model
7. Demonstrates predictions

Expected Runtime:30-60 seconds

- Use Prediction Module:

from src.predictor import GastricDiseasePredictor

predictor = GastricDiseasePredictor()

patient = {
    'age': 55, 'gender': 1, 'bmi': 28.5, 'smoking': 1,
    'alcohol_consumption': 2, 'family_history': 1,
    'spicy_food_intake': 8, 'stress_level': 7,
    'irregular_meals': 1, 'abdominal_pain': 1,
    'nausea': 1, 'bloating': 1, 'heartburn': 1,
    'loss_of_appetite': 0
}

result = predictor.predict(patient)
print(f"Prediction: {result['prediction_label']}")
print(f"Probability: {result['probability_disease']*100:.1f}%")
print(f"Risk Level: {result['risk_level']}")

📊 Dataset :
Features (14 total)

-Demographic (3):age,gender,BMI
-Lifestyle (6): smoking, alcohol, family history, spicy food, stress, irregular meals
-Symptoms (5): abdominal pain, nausea, bloating, heartburn, loss of appetite

-Target: gastric_disease (0=No, 1=Risk)

-Target Logic
Weighted scoring based on risk factors and symptoms
+10% noise added to simulate real-world variability

🤖 Machine Learning Pipeline:

-Preprocessing: Data validation, train-test split, feature scaling
-Models: Logistic Regression, Random Forest, SVM
-Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
-Model Selection: Best ROC-AUC model chosen (usually Random Forest)

📈 Model Performance:
| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| -------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression  | 0.75     | 0.73      | 0.78   | 0.75     | 0.82    |
| Random Forest        | 0.85     | 0.84      | 0.86   | 0.85     | 0.91    |
| SVM                  | 0.80     | 0.79      | 0.82   | 0.80     | 0.87    |


📊 Visualizations:

 Generated Visualizations

1. EDA Analysis (`eda_analysis.png`)
   - Class distribution
   - Age and BMI distributions by disease
   - Symptom prevalence
   - Risk factor analysis

2. Correlation Heatmap (`correlation_heatmap.png`)
   - Feature-feature correlations
   - Feature-target relationships

3. Model Comparison (`model_comparison.png`)
   - Side-by-side metric comparison
   - ROC-AUC highlighting

4. Confusion Matrices (`confusion_matrices.png`)
   - True/False Positives/Negatives for each model
   - Best model highlighted

5.  ROC Curves (`roc_curves.png`)
   - ROC curves for all models
   - AUC scores displayed

6.  Feature Importance (`feature_importance.png`)
   - Random Forest feature importance ranking


🔮 Future Enhancements:
-Hyperparameter tuning (GridSearchCV)
-SMOTE for class imbalance
-Feature selection techniques
-Web or API deployment
-Integration with real datasets and EHR
-Deep learning models and SHAP interpretability


🤝 Contributing:
-Fork → Feature branch → Commit → Push → Pull request
-Follow PEP 8 and add docstrings
-Include unit tests for new features


 📊 Project Statistics:

- Lines of Code: ~2000+
- Models Implemented: 3
- Visualizations Created: 6
- Evaluation Metrics: 5
- Dataset Size: 1000 samples (configurable)
- Features: 14


🎓 Learning Outcomes:

By completing this project, you will learn:

✅ End-to-end ML pipeline development  
✅ Data generation and preprocessing  
✅ Multiple classification algorithms  
✅ Model evaluation and comparison  
✅ Visualization techniques  
✅ Production-ready code structure  
✅ Git and GitHub workflow  
✅ Documentation best practices 



📄 License:

MIT License – see LICENSE 



👤 Author : 
 DESIGAPPERUMAL N
🌐 Website
 | 💼 linkdIn: www.linkedin.com/in/desigapperumal-n
 | 🐱 github : https://github.com/desiganMLengineer
 | 📧 mail   : desigapperumal151@gmail.com.com


⚠️ Disclaimer:
-Educational purposes only – not for medical diagnosis
-Always consult a licensed healthcare professional
-Synthetic data may not reflect real-world cases
-No liability -The author assume no responsibility for any medical decisions made using this tool

For actual medical concerns, please consult a licensed healthcare provider.



⭐ If you found this project helpful, please consider giving it a star!


-INFO:
**Last Updated:** [10-04-2026]
Version: 1.0.0