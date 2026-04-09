"""
Data Generator Module
Generates realistic synthetic gastric disease dataset
"""

import pandas as pd
import numpy as np
import config

class GastricDataGenerator:
    """
    Generates synthetic gastric disease prediction dataset
    
    The dataset simulates real-world patient data with features that
    are known risk factors for gastric diseases.
    """
    
    def __init__(self, n_samples=1000, random_state=42):
        """
        Initialize the data generator
        
        Parameters:
        -----------
        n_samples : int
            Number of patient records to generate
        random_state : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_demographic_features(self):
        """
        Generate demographic features
        
        Returns:
        --------
        dict : Dictionary containing demographic features
        
        Features:
        - age: 18-80 years (normal distribution around 45)
        - gender: 0=Female, 1=Male (roughly 50/50)
        - bmi: Body Mass Index (normal distribution around 25)
        """
        return {
            'age': np.random.normal(45, 15, self.n_samples).clip(18, 80).astype(int),
            'gender': np.random.choice([0, 1], self.n_samples),
            'bmi': np.random.normal(25, 5, self.n_samples).clip(15, 45).round(1)
        }
    
    def generate_lifestyle_features(self):
        """
        Generate lifestyle and habit features
        
        Returns:
        --------
        dict : Dictionary containing lifestyle features
        
        Features:
        - smoking: 0=No, 1=Yes (30% smoke)
        - alcohol_consumption: 0=None, 1=Moderate, 2=Heavy
        - spicy_food_intake: Scale 0-10 (how often they eat spicy food)
        - stress_level: Scale 0-10 (self-reported stress)
        - irregular_meals: 0=Regular, 1=Irregular (40% irregular)
        - family_history: 0=No, 1=Yes (25% have family history)
        """
        return {
            'smoking': np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            'alcohol_consumption': np.random.choice([0, 1, 2], self.n_samples, p=[0.5, 0.3, 0.2]),
            'spicy_food_intake': np.random.randint(0, 11, self.n_samples),
            'stress_level': np.random.randint(0, 11, self.n_samples),
            'irregular_meals': np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4]),
            'family_history': np.random.choice([0, 1], self.n_samples, p=[0.75, 0.25])
        }
    
    def generate_symptom_features(self):
        """
        Generate symptom features
        
        Returns:
        --------
        dict : Dictionary containing symptom features
        
        Features (all binary: 0=No, 1=Yes):
        - abdominal_pain: Stomach pain
        - nausea: Feeling of wanting to vomit
        - bloating: Feeling of fullness/swelling
        - heartburn: Burning sensation in chest
        - loss_of_appetite: Not feeling hungry
        """
        return {
            'abdominal_pain': np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4]),
            'nausea': np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            'bloating': np.random.choice([0, 1], self.n_samples, p=[0.65, 0.35]),
            'heartburn': np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            'loss_of_appetite': np.random.choice([0, 1], self.n_samples, p=[0.75, 0.25])
        }
    
    def calculate_risk_score(self, df):
        """
        Calculate gastric disease risk score based on features
        
        This mimics real-world medical knowledge where certain factors
        increase disease risk. Higher scores indicate higher risk.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with all features
            
        Returns:
        --------
        array : Risk scores for each patient
        """
        weights = config.RISK_WEIGHTS
        
        risk_score = (
            # Demographic risks
            (df['age'] > weights['age_threshold']).astype(int) * weights['age_weight'] +
            
            # Lifestyle risks
            df['smoking'] * weights['smoking_weight'] +
            (df['alcohol_consumption'] == 2).astype(int) * weights['heavy_alcohol_weight'] +
            df['family_history'] * weights['family_history_weight'] +
            (df['spicy_food_intake'] > weights['spicy_food_threshold']).astype(int) * weights['spicy_food_weight'] +
            (df['stress_level'] > weights['stress_threshold']).astype(int) * weights['stress_weight'] +
            df['irregular_meals'] * weights['irregular_meals_weight'] +
            
            # Symptom risks
            df['abdominal_pain'] * weights['abdominal_pain_weight'] +
            df['nausea'] * weights['nausea_weight'] +
            df['bloating'] * weights['bloating_weight'] +
            df['heartburn'] * weights['heartburn_weight'] +
            df['loss_of_appetite'] * weights['loss_of_appetite_weight']
        )
        
        return risk_score
    
    def add_realistic_noise(self, df, noise_level=0.1):
        """
        Add noise to make dataset more realistic
        
        In real world, diagnosis isn't perfect and there are edge cases.
        This function randomly flips some labels to simulate this.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with target variable
        noise_level : float
            Proportion of labels to flip (0.0 to 1.0)
            
        Returns:
        --------
        DataFrame : DataFrame with noisy labels
        """
        n_noise = int(noise_level * len(df))
        noise_indices = np.random.choice(df.index, size=n_noise, replace=False)
        df.loc[noise_indices, config.TARGET_VARIABLE] = 1 - df.loc[noise_indices, config.TARGET_VARIABLE]
        return df
    
    def generate_dataset(self, save_path=None):
        """
        Generate complete synthetic dataset
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the dataset CSV
            
        Returns:
        --------
        DataFrame : Complete synthetic dataset
        """
        print("=" * 60)
        print("GENERATING SYNTHETIC GASTRIC DISEASE DATASET")
        print("=" * 60)
        
        # Generate all feature groups
        print("\n1. Generating demographic features...")
        demographic_data = self.generate_demographic_features()
        
        print("2. Generating lifestyle features...")
        lifestyle_data = self.generate_lifestyle_features()
        
        print("3. Generating symptom features...")
        symptom_data = self.generate_symptom_features()
        
        # Combine all features
        print("4. Combining all features...")
        all_data = {**demographic_data, **lifestyle_data, **symptom_data}
        df = pd.DataFrame(all_data)
        
        # Calculate risk and create target variable
        print("5. Calculating disease risk scores...")
        risk_scores = self.calculate_risk_score(df)
        
        # Convert to binary classification (above median = disease)
        threshold = risk_scores.median()
        df[config.TARGET_VARIABLE] = (risk_scores > threshold).astype(int)
        
        # Add realistic noise
        print(f"6. Adding {config.DATASET_CONFIG['noise_level']*100}% noise for realism...")
        df = self.add_realistic_noise(df, config.DATASET_CONFIG['noise_level'])
        
        # Display summary
        print("\n" + "=" * 60)
        print("DATASET GENERATION COMPLETE")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"Total features: {len(df.columns) - 1}")
        print(f"\nClass Distribution:")
        class_dist = df[config.TARGET_VARIABLE].value_counts()
        print(f"  No Disease (0): {class_dist[0]} ({class_dist[0]/len(df)*100:.1f}%)")
        print(f"  Disease (1): {class_dist[1]} ({class_dist[1]/len(df)*100:.1f}%)")
        
        # Save if path provided
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\n✓ Dataset saved to: {save_path}")
        
        return df


def main():
    """Test the data generator"""
    generator = GastricDataGenerator(
        n_samples=config.DATASET_CONFIG['n_samples'],
        random_state=config.DATASET_CONFIG['random_state']
    )
    
    df = generator.generate_dataset(save_path=config.OUTPUT_FILES['dataset'])
    print("\nFirst 5 rows:")
    print(df.head())
    

if __name__ == "__main__":
    main()