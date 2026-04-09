# 🏥 Gastric Disease Prediction System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete machine learning system to predict gastric disease risk using patient demographics, lifestyle habits, and symptoms.  
This project covers **data generation, EDA, model training, evaluation, and a production-ready prediction interface**.

---

## 📋 Table of Contents
- Overview
- Features
- Project Structure
- Installation
- Usage
- Dataset
- Machine Learning Pipeline
- Model Performance
- Visualizations
- Future Enhancements
- Contributing
- Project Statistics
- Learning Outcomes
- License
- Author
- Disclaimer

---

## 🎯 Overview
The **Gastric Disease Prediction System** is an end-to-end ML solution that can:

- Generate realistic synthetic patient datasets  
- Perform exploratory data analysis (EDA) with visualizations  
- Train and compare multiple ML algorithms  
- Select the best performing model  
- Provide risk assessment for new patients  

**Why it matters:**
- Gastric diseases affect millions worldwide  
- Early detection improves treatment outcomes  
- ML can identify high-risk patients for preventive care  
- Reduces healthcare costs through early intervention  

---

## ✨ Features

**Machine Learning**
- Logistic Regression, Random Forest, SVM  
- 5-fold Cross-Validation  
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC  
- Feature importance analysis  

**Data Analysis**
- Automated EDA with plots and summaries  
- Correlation heatmaps  
- Synthetic dataset generation (balanced classes)  
- Input validation & error handling  

**Visualizations**
- Model comparison charts  
- Confusion matrices  
- ROC curves  
- Feature importance plots  
- EDA visualizations  

**Production Ready**
- Modular code structure  
- Error handling & logging  
- Documentation included  
- Easy prediction interface  

---

## 📁 Project Structure
```
gastric-disease-prediction/
│
├── data/
│   └── synthetic_gastric_data.csv
│
├── outputs/
│   ├── visualizations/
│   │   ├── eda_analysis.png
│   │   ├── correlation_heatmap.png
│   │   ├── model_comparison.png
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   └── feature_importance.png
│   │
│   ├── models/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── [model_name]_model.pkl
│   │
│   └── model_results.json
│
├── src/
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── model_trainer.py
│   ├── evaluator.py
│   └── predictor.py
│
├── config.py
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🔧 Installation

**Prerequisites**
- Python 3.7+  
- pip  

**Clone the Repository**
```bash
git clone https://github.com/yourusername/gastric-disease-prediction.git
cd gastric-disease-prediction
```

**Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

**Run Full Pipeline**
```bash
python main.py
```

Steps executed:
1. Generate synthetic dataset (1000 patients)  
2. Perform EDA  
3. Preprocess data  
4. Train models  
5. Evaluate & compare  
6. Save best model  
7. Demo predictions  

**Expected Runtime:** 30–60 seconds  

**Prediction Module Example**
```python
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
```

---

## 📊 Dataset
- **Features (14):**  
  - Demographic: age, gender, BMI  
  - Lifestyle: smoking, alcohol, family history, spicy food, stress, irregular meals  
  - Symptoms: abdominal pain, nausea, bloating, heartburn, loss of appetite  
- **Target:** gastric_disease (0 = No, 1 = Risk)  
- **Logic:** Weighted scoring + 10% noise  

---

## 🤖 Machine Learning Pipeline
- Preprocessing: validation, split, scaling  
- Models: Logistic Regression, Random Forest, SVM  
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC  
- Selection: Best ROC-AUC model (usually Random Forest)  

---

## 📈 Model Performance
| Model               | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------------------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.75     | 0.73      | 0.78   | 0.75 | 0.82    |
| Random Forest       | 0.85     | 0.84      | 0.86   | 0.85 | 0.91    |
| SVM                 | 0.80     | 0.79      | 0.82   | 0.80 | 0.87    |

---

## 📊 Visualizations
- EDA Analysis (`eda_analysis.png`)  
- Correlation Heatmap (`correlation_heatmap.png`)  
- Model Comparison (`model_comparison.png`)  
- Confusion Matrices (`confusion_matrices.png`)  
- ROC Curves (`roc_curves.png`)  
- Feature Importance (`feature_importance.png`)  

---

## 🔮 Future Enhancements
- Hyperparameter tuning (GridSearchCV)  
- SMOTE for imbalance  
- Feature selection  
- Web/API deployment  
- Real datasets & EHR integration  
- Deep learning + SHAP  

---

## 🤝 Contributing
- Fork → Branch → Commit → Push → PR  
- Follow PEP 8  
- Add docstrings  
- Include unit tests  

---

## 📊 Project Statistics
- ~2000+ LOC  
- 3 Models  
- 6 Visualizations  
- 5 Metrics  
- 1000 samples dataset  
- 14 features  

---

## 🎓 Learning Outcomes
- End-to-end ML pipeline  
- Data generation & preprocessing  
- Classification algorithms  
- Model evaluation & comparison  
- Visualization techniques  
- Production-ready code  
- Git/GitHub workflow  
- Documentation best practices  

---

## 📄 License
MIT License – see LICENSE file  

---

## 👤 Author
**DESIGAPPERUMAL N**  
💼 LinkedIn: linkedin.com/in/desigapperumal-n 
🐱 GitHub: [github.com/desiganMLengineer](https://github.com/desiganMLengineer)  
📧 Email: desigapperumal151@gmail.com  

---

## ⚠️ Disclaimer
- Educational purposes only – not for medical diagnosis  
- Always consult a licensed healthcare professional  
- Synthetic data may not reflect real-world cases  
- No liability – author assumes no responsibility for medical decisions made using this tool  
For actual medical concerns, please consult a licensed healthcare provider.

⭐ If you found this project helpful, please consider giving it a star!
