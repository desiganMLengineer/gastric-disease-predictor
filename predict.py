"""
Prediction System Module
Makes predictions for new patients using the best trained model
"""

import pandas as pd
import numpy as np
import joblib
import config

class GastricDiseasePredictor:
    """
    Production-ready prediction system
    
    Loads the best model and makes predictions for new patients
    """
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize predictor with saved model and scaler
        
        Parameters:
        -----------
        model_path : str, optional
            Path to saved model file
        scaler_path : str, optional
            Path to saved scaler file
        """
        # Load model
        if model_path is None:
            model_path = config.OUTPUT_FILES['best_model']
        
        if scaler_path is None:
            scaler_path = config.OUTPUT_FILES['scaler']
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Model loaded from: {model_path}")
            print(f"✓ Scaler loaded from: {scaler_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model or scaler not found. Please train the model first.\n{e}"
            )
    
    def validate_input(self, patient_data):
        """
        Validate patient input data
        
        Parameters:
        -----------
        patient_data : dict
            Patient features
            
        Returns:
        --------
        bool : True if valid
        """
        # Check all required features present
        missing_features = set(config.ALL_FEATURES) - set(patient_data.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Check value ranges
        validations = {
            'age': (18, 80, "Age must be between 18 and 80"),
            'gender': (0, 1, "Gender must be 0 (Female) or 1 (Male)"),
            'bmi': (15, 45, "BMI must be between 15 and 45"),
            'smoking': (0, 1, "Smoking must be 0 (No) or 1 (Yes)"),
            'alcohol_consumption': (0, 2, "Alcohol consumption must be 0, 1, or 2"),
            'spicy_food_intake': (0, 10, "Spicy food intake must be 0-10"),
            'stress_level': (0, 10, "Stress level must be 0-10"),
            'irregular_meals': (0, 1, "Irregular meals must be 0 or 1"),
            'family_history': (0, 1, "Family history must be 0 or 1"),
            'abdominal_pain': (0, 1, "Abdominal pain must be 0 or 1"),
            'nausea': (0, 1, "Nausea must be 0 or 1"),
            'bloating': (0, 1, "Bloating must be 0 or 1"),
            'heartburn': (0, 1, "Heartburn must be 0 or 1"),
            'loss_of_appetite': (0, 1, "Loss of appetite must be 0 or 1")
        }
        
        for feature, (min_val, max_val, msg) in validations.items():
            value = patient_data[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{msg}. Got: {value}")
        
        return True
    
    def predict(self, patient_data):
        """
        Make prediction for a single patient
        
        Parameters:
        -----------
        patient_data : dict
            Patient features
            
        Returns:
        --------
        dict : Prediction results with probability
        """
        # Validate input
        self.validate_input(patient_data)
        
        # Convert to DataFrame with correct feature order
        patient_df = pd.DataFrame([patient_data])[config.ALL_FEATURES]
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_df)
        
        # Make prediction
        prediction = self.model.predict(patient_scaled)[0]
        probabilities = self.model.predict_proba(patient_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Gastric Disease Risk' if prediction == 1 else 'No Significant Risk',
            'probability_no_disease': float(probabilities[0]),
            'probability_disease': float(probabilities[1]),
            'confidence': float(max(probabilities)),
            'risk_level': self._get_risk_level(probabilities[1])
        }
        
        return result
    
    def _get_risk_level(self, disease_probability):
        """
        Convert probability to risk level
        
        Parameters:
        -----------
        disease_probability : float
            Probability of disease (0-1)
            
        Returns:
        --------
        str : Risk level (Low, Medium, High, Very High)
        """
        if disease_probability < 0.3:
            return "Low"
        elif disease_probability < 0.5:
            return "Medium"
        elif disease_probability < 0.7:
            return "High"
        else:
            return "Very High"
    
    def predict_batch(self, patients_list):
        """
        Make predictions for multiple patients
        
        Parameters:
        -----------
        patients_list : list of dict
            List of patient data dictionaries
            
        Returns:
        --------
        list : List of prediction results
        """
        results = []
        for idx, patient_data in enumerate(patients_list):
            try:
                result = self.predict(patient_data)
                result['patient_id'] = idx
                results.append(result)
            except Exception as e:
                print(f"Error predicting for patient {idx}: {e}")
                results.append({
                    'patient_id': idx,
                    'error': str(e)
                })
        
        return results
    
    def explain_prediction(self, patient_data, result):
        """
        Provide explanation for the prediction
        
        Parameters:
        -----------
        patient_data : dict
            Patient features
        result : dict
            Prediction result
        """
        print("\n" + "=" * 70)
        print("GASTRIC DISEASE RISK ASSESSMENT")
        print("=" * 70)
        
        # Patient profile
        print("\nPATIENT PROFILE:")
        print("-" * 70)
        print(f"Age: {patient_data['age']} years")
        print(f"Gender: {'Male' if patient_data['gender'] == 1 else 'Female'}")
        print(f"BMI: {patient_data['bmi']}")
        
        print("\nLIFESTYLE FACTORS:")
        print("-" * 70)
        print(f"Smoking: {'Yes' if patient_data['smoking'] == 1 else 'No'}")
        alcohol_labels = {0: 'None', 1: 'Moderate', 2: 'Heavy'}
        print(f"Alcohol Consumption: {alcohol_labels[patient_data['alcohol_consumption']]}")
        print(f"Spicy Food Intake: {patient_data['spicy_food_intake']}/10")
        print(f"Stress Level: {patient_data['stress_level']}/10")
        print(f"Irregular Meals: {'Yes' if patient_data['irregular_meals'] == 1 else 'No'}")
        print(f"Family History: {'Yes' if patient_data['family_history'] == 1 else 'No'}")
        
        print("\nSYMPTOMS:")
        print("-" * 70)
        symptoms = {
            'Abdominal Pain': patient_data['abdominal_pain'],
            'Nausea': patient_data['nausea'],
            'Bloating': patient_data['bloating'],
            'Heartburn': patient_data['heartburn'],
            'Loss of Appetite': patient_data['loss_of_appetite']
        }
        for symptom, present in symptoms.items():
            status = '✓ Yes' if present == 1 else '✗ No'
            print(f"{symptom:20} {status}")
        
        # Prediction
        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)
        print(f"\n🏥 Diagnosis: {result['prediction_label']}")
        print(f"📊 Risk Level: {result['risk_level']}")
        print(f"📈 Confidence: {result['confidence']*100:.1f}%")
        
        print(f"\nProbability Breakdown:")
        print(f"  No Disease: {result['probability_no_disease']*100:.1f}%")
        print(f"  Disease:    {result['probability_disease']*100:.1f}%")
        
        # Risk factors present
        print("\n" + "=" * 70)
        print("RISK FACTORS IDENTIFIED")
        print("=" * 70)
        
        risk_factors = []
        if patient_data['age'] > 50:
            risk_factors.append("• Age over 50")
        if patient_data['smoking'] == 1:
            risk_factors.append("• Smoking")
        if patient_data['alcohol_consumption'] == 2:
            risk_factors.append("• Heavy alcohol consumption")
        if patient_data['family_history'] == 1:
            risk_factors.append("• Family history of gastric disease")
        if patient_data['spicy_food_intake'] > 7:
            risk_factors.append("• High spicy food intake")
        if patient_data['stress_level'] > 7:
            risk_factors.append("• High stress level")
        if patient_data['irregular_meals'] == 1:
            risk_factors.append("• Irregular eating habits")
        
        # Count symptoms
        symptom_count = sum(symptoms.values())
        if symptom_count > 0:
            risk_factors.append(f"• {symptom_count} symptom(s) present")
        
        if risk_factors:
            for factor in risk_factors:
                print(factor)
        else:
            print("No major risk factors identified")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        if result['prediction'] == 1:
            print("\n⚠️  We recommend:")
            print("  1. Consult a gastroenterologist for proper diagnosis")
            print("  2. Get an endoscopy if symptoms persist")
            print("  3. Reduce risk factors (smoking, alcohol, spicy food)")
            print("  4. Maintain regular meal times")
            print("  5. Manage stress levels")
        else:
            print("\n✓ Risk appears low, but maintain healthy habits:")
            print("  1. Continue healthy lifestyle")
            print("  2. Regular check-ups")
            print("  3. Monitor any new symptoms")
            print("  4. Maintain balanced diet")
        
        print("\n⚠️  DISCLAIMER: This is a prediction model for educational purposes.")
        print("   Always consult healthcare professionals for medical advice.")
        print("=" * 70)


def main():
    """
    Test the prediction system with example patients
    """
    print("\n" + "=" * 70)
    print("GASTRIC DISEASE PREDICTION SYSTEM - DEMO")
    print("=" * 70)
    
    # Initialize predictor
    try:
        predictor = GastricDiseasePredictor()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the complete pipeline first:")
        print("  python main.py")
        return
    
    # Example patient 1: High risk
    print("\n\n" + "🔴" * 35)
    print("EXAMPLE 1: HIGH RISK PATIENT")
    print("🔴" * 35)
    
    high_risk_patient = {
        'age': 58,
        'gender': 1,
        'bmi': 29.5,
        'smoking': 1,
        'alcohol_consumption': 2,
        'family_history': 1,
        'spicy_food_intake': 9,
        'stress_level': 8,
        'irregular_meals': 1,
        'abdominal_pain': 1,
        'nausea': 1,
        'bloating': 1,
        'heartburn': 1,
        'loss_of_appetite': 1
    }
    
    result1 = predictor.predict(high_risk_patient)
    predictor.explain_prediction(high_risk_patient, result1)
    
    # Example patient 2: Low risk
    print("\n\n" + "🟢" * 35)
    print("EXAMPLE 2: LOW RISK PATIENT")
    print("🟢" * 35)
    
    low_risk_patient = {
        'age': 28,
        'gender': 0,
        'bmi': 22.0,
        'smoking': 0,
        'alcohol_consumption': 0,
        'family_history': 0,
        'spicy_food_intake': 3,
        'stress_level': 4,
        'irregular_meals': 0,
        'abdominal_pain': 0,
        'nausea': 0,
        'bloating': 0,
        'heartburn': 0,
        'loss_of_appetite': 0
    }
    
    result2 = predictor.predict(low_risk_patient)
    predictor.explain_prediction(low_risk_patient, result2)


if __name__ == "__main__":
    main()