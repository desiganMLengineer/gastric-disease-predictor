"""
Preprocessing Module
Handles data preparation, splitting, and scaling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config
import joblib

class DataPreprocessor:
    """
    Handles all data preprocessing tasks
    
    This includes:
    - Data validation
    - Exploratory Data Analysis (EDA)
    - Train-test splitting
    - Feature scaling
    """
    
    def __init__(self, df):
        """
        Initialize preprocessor with dataset
        
        Parameters:
        -----------
        df : DataFrame
            Input dataset
        """
        self.df = df
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
    def validate_data(self):
        """
        Validate dataset integrity
        
        Checks for:
        - Missing values
        - Data types
        - Feature presence
        """
        print("\n" + "=" * 60)
        print("DATA VALIDATION")
        print("=" * 60)
        
        # Check missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠ Warning: Missing values found!")
            print(missing[missing > 0])
        else:
            print("\n✓ No missing values")
        
        # Check required features
        required_features = config.ALL_FEATURES + [config.TARGET_VARIABLE]
        missing_features = set(required_features) - set(self.df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        else:
            print("✓ All required features present")
        
        # Check data types
        print("\n✓ Data types validated")
        
        return True
    
    def exploratory_analysis(self, save_path=None):
        """
        Perform comprehensive Exploratory Data Analysis
        
        Creates visualizations to understand:
        - Data distribution
        - Feature relationships
        - Class balance
        - Correlations
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save EDA visualizations
        """
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 60)
        
        # Basic statistics
        print("\nDataset Shape:", self.df.shape)
        print(f"Features: {self.df.shape[1] - 1}")
        print(f"Samples: {self.df.shape[0]}")
        
        print("\n" + "-" * 60)
        print("Statistical Summary:")
        print("-" * 60)
        print(self.df.describe())
        
        # Class distribution
        print("\n" + "-" * 60)
        print("Class Distribution:")
        print("-" * 60)
        target_counts = self.df[config.TARGET_VARIABLE].value_counts()
        for label, count in target_counts.items():
            disease_status = "Disease" if label == 1 else "No Disease"
            percentage = (count / len(self.df)) * 100
            print(f"  {disease_status}: {count} ({percentage:.1f}%)")
        
        # Create visualizations
        self._create_eda_visualizations(save_path)
        
    def _create_eda_visualizations(self, save_path=None):
        """
        Create EDA visualization plots
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Class Distribution
        ax1 = plt.subplot(3, 3, 1)
        class_counts = self.df[config.TARGET_VARIABLE].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax1.bar(['No Disease', 'Disease'], class_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12)
        for i, v in enumerate(class_counts.values):
            ax1.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        # 2. Age Distribution by Disease
        ax2 = plt.subplot(3, 3, 2)
        self.df.boxplot(column='age', by=config.TARGET_VARIABLE, ax=ax2)
        ax2.set_title('Age Distribution by Disease Status', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Gastric Disease (0=No, 1=Yes)', fontsize=12)
        ax2.set_ylabel('Age', fontsize=12)
        plt.sca(ax2)
        plt.xticks([1, 2], ['No Disease', 'Disease'])
        
        # 3. BMI Distribution by Disease
        ax3 = plt.subplot(3, 3, 3)
        self.df.boxplot(column='bmi', by=config.TARGET_VARIABLE, ax=ax3)
        ax3.set_title('BMI Distribution by Disease Status', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Gastric Disease (0=No, 1=Yes)', fontsize=12)
        ax3.set_ylabel('BMI', fontsize=12)
        plt.sca(ax3)
        plt.xticks([1, 2], ['No Disease', 'Disease'])
        
        # 4. Smoking vs Disease
        ax4 = plt.subplot(3, 3, 4)
        smoking_disease = pd.crosstab(self.df['smoking'], self.df[config.TARGET_VARIABLE], normalize='index') * 100
        smoking_disease.plot(kind='bar', ax=ax4, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax4.set_title('Smoking Status vs Disease', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Smoking (0=No, 1=Yes)', fontsize=12)
        ax4.set_ylabel('Percentage (%)', fontsize=12)
        ax4.legend(['No Disease', 'Disease'], loc='upper right')
        ax4.set_xticklabels(['Non-Smoker', 'Smoker'], rotation=0)
        
        # 5. Symptom Prevalence
        ax5 = plt.subplot(3, 3, 5)
        symptoms = config.FEATURES['symptoms']
        symptom_prev = self.df.groupby(config.TARGET_VARIABLE)[symptoms].mean().T * 100
        symptom_prev.plot(kind='barh', ax=ax5, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax5.set_title('Symptom Prevalence by Disease Status', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Prevalence (%)', fontsize=12)
        ax5.set_ylabel('Symptoms', fontsize=12)
        ax5.legend(['No Disease', 'Disease'], loc='lower right')
        
        # 6. Alcohol Consumption Distribution
        ax6 = plt.subplot(3, 3, 6)
        alcohol_dist = self.df['alcohol_consumption'].value_counts().sort_index()
        ax6.bar(['None', 'Moderate', 'Heavy'], alcohol_dist.values, color=['#3498db', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax6.set_title('Alcohol Consumption Distribution', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Alcohol Consumption Level', fontsize=12)
        ax6.set_ylabel('Count', fontsize=12)
        
        # 7. Stress Level Distribution
        ax7 = plt.subplot(3, 3, 7)
        for disease in [0, 1]:
            disease_label = 'Disease' if disease == 1 else 'No Disease'
            color = '#e74c3c' if disease == 1 else '#2ecc71'
            subset = self.df[self.df[config.TARGET_VARIABLE] == disease]['stress_level']
            ax7.hist(subset, bins=10, alpha=0.6, label=disease_label, color=color, edgecolor='black')
        ax7.set_title('Stress Level Distribution', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Stress Level (0-10)', fontsize=12)
        ax7.set_ylabel('Frequency', fontsize=12)
        ax7.legend()
        
        # 8. Family History Impact
        ax8 = plt.subplot(3, 3, 8)
        family_disease = pd.crosstab(self.df['family_history'], self.df[config.TARGET_VARIABLE], normalize='index') * 100
        family_disease.plot(kind='bar', ax=ax8, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax8.set_title('Family History vs Disease', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Family History (0=No, 1=Yes)', fontsize=12)
        ax8.set_ylabel('Percentage (%)', fontsize=12)
        ax8.legend(['No Disease', 'Disease'], loc='upper left')
        ax8.set_xticklabels(['No Family History', 'Family History'], rotation=0)
        
        # 9. Gender Distribution
        ax9 = plt.subplot(3, 3, 9)
        gender_disease = pd.crosstab(self.df['gender'], self.df[config.TARGET_VARIABLE], normalize='index') * 100
        gender_disease.plot(kind='bar', ax=ax9, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax9.set_title('Gender vs Disease', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Gender (0=Female, 1=Male)', fontsize=12)
        ax9.set_ylabel('Percentage (%)', fontsize=12)
        ax9.legend(['No Disease', 'Disease'], loc='upper right')
        ax9.set_xticklabels(['Female', 'Male'], rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
            print(f"\n✓ EDA visualizations saved to: {save_path}")
        
        plt.close()
        
        # Create correlation heatmap separately
        self._create_correlation_heatmap()
    
    def _create_correlation_heatmap(self):
        """Create and save correlation heatmap"""
        plt.figure(figsize=(14, 10))
        
        # Calculate correlation matrix
        correlation = self.df.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        save_path = config.OUTPUT_FILES['correlation_heatmap']
        plt.savefig(save_path, dpi=config.VIZ_CONFIG['dpi'], bbox_inches='tight')
        print(f"✓ Correlation heatmap saved to: {save_path}")
        plt.close()
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Why we split:
        - Training set: Used to train the model (80%)
        - Testing set: Used to evaluate model on unseen data (20%)
        
        This prevents overfitting and gives realistic performance estimates.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing (default: 0.2 = 20%)
        random_state : int
            Random seed for reproducibility
        """
        print("\n" + "=" * 60)
        print("DATA SPLITTING")
        print("=" * 60)
        
        # Separate features (X) and target (y)
        X = self.df[config.ALL_FEATURES]
        y = self.df[config.TARGET_VARIABLE]
        
        print(f"\nFeatures (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        
        # Perform stratified split (maintains class proportions)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Ensures same class distribution in train and test
        )
        
        print(f"\nTraining set: {self.X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"Testing set: {self.X_test.shape[0]} samples ({test_size*100:.0f}%)")
        
        # Verify class distribution
        train_dist = self.y_train.value_counts(normalize=True) * 100
        test_dist = self.y_test.value_counts(normalize=True) * 100
        print("\nClass distribution maintained:")
        print(f"  Training - No Disease: {train_dist[0]:.1f}%, Disease: {train_dist[1]:.1f}%")
        print(f"  Testing  - No Disease: {test_dist[0]:.1f}%, Disease: {test_dist[1]:.1f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """
        Scale features using StandardScaler
        
        Why we scale:
        - Different features have different ranges (age: 18-80, BMI: 15-45)
        - ML algorithms work better when features are on similar scales
        - StandardScaler: transforms to mean=0, std=1
        
        Formula: z = (x - mean) / std
        
        Important: We fit on training data only, then transform both train and test
        This prevents data leakage from test set.
        """
        print("\n" + "=" * 60)
        print("FEATURE SCALING")
        print("=" * 60)
        
        if self.X_train is None:
            raise ValueError("Please split data first using split_data()")
        
        print("\nScaling method: StandardScaler")
        print("Formula: (x - mean) / standard_deviation")
        
        # Fit on training data only
        print("\n1. Fitting scaler on training data...")
        self.scaler.fit(self.X_train)
        
        # Transform both train and test
        print("2. Transforming training data...")
        self.X_train_scaled = self.scaler.transform(self.X_train)
        
        print("3. Transforming testing data...")
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\n✓ Feature scaling complete")
        print(f"  Scaled training data shape: {self.X_train_scaled.shape}")
        print(f"  Scaled testing data shape: {self.X_test_scaled.shape}")
        
        # Save scaler for later use
        scaler_path = config.OUTPUT_FILES['scaler']
        joblib.dump(self.scaler, scaler_path)
        print(f"\n✓ Scaler saved to: {scaler_path}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def get_processed_data(self):
        """
        Get all processed data
        
        Returns:
        --------
        tuple : (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test


def main():
    """Test the preprocessor"""
    # Load data
    df = pd.read_csv(config.OUTPUT_FILES['dataset'])
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(df)
    
    # Validate data
    preprocessor.validate_data()
    
    # Perform EDA
    preprocessor.exploratory_analysis(save_path=config.OUTPUT_FILES['eda_viz'])
    
    # Split and scale
    preprocessor.split_data(test_size=config.DATASET_CONFIG['test_size'])
    preprocessor.scale_features()
    

if __name__ == "__main__":
    main()