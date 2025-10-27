"""
=============================================================================
SHIP DELIVERY DELAY PREDICTION - COMPLETE ML + DEEP LEARNING PIPELINE
=============================================================================
~ By Ryan Tusi -> https://github.com/ryantusi/Ship_Delay_AI_Predictor
=============================================================================
Domain: Maritime Logistics & Supply Chain
Problem: Predict whether a shipment will be delayed (binary classification)
Dataset: Synthetic based on real logistics patterns

Enhanced with: Traditional ML + Deep Learning (Neural Networks)
Model Formats: JSON (metadata) + H5 (deep learning) + Joblib (ML)
=============================================================================
"""

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for consistency

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)

# Deep Learning imports - with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras import models as keras_models
    from tensorflow.keras.utils import plot_model
    TENSORFLOW_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print("=" * 80)
    print("WARNING: TensorFlow is not installed!")
    print("=" * 80)
    print("\nTo install TensorFlow, run:")
    print("  pip install tensorflow")
    print("\nOr for CPU-only version:")
    print("  pip install tensorflow-cpu")
    print("\nThe script will continue with Traditional ML models only.")
    print("=" * 80)
    tf = None
    keras = None
    layers = None
    regularizers = None
    keras_models = None

# Model persistence
import joblib
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("SHIP DELIVERY DELAY PREDICTION - ML + DEEP LEARNING PIPELINE")
print("By Ryan Tusi -> https://github.com/ryantusi/Ship_Delay_AI_Predictor")
print("=" * 80)
print("\nProject Scope:")
print("Goals: Build production-ready ML + DL pipeline to predict shipment delays")
print("Data: Maritime logistics data with vessel, route, and operational features")
print("Analysis: Binary classification with complete MLOps + Deep Learning workflow")
if TENSORFLOW_AVAILABLE:
    print(f"TensorFlow Version: {tf.__version__}\n")
else:
    print("Running in ML-only mode (TensorFlow not available)\n")

# ============================================================================
# 2. DATA GENERATION (Simulating Real Logistics Data)
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 1: DATA EXTRACTION AND GENERATION")
print("=" * 80)

def generate_ship_logistics_data(n_samples=5000, random_state=42):
    """
    Generate realistic ship logistics dataset
    Features based on actual maritime logistics factors
    """
    np.random.seed(random_state)
    
    # Vessel characteristics
    vessel_types = ['Container Ship', 'Bulk Carrier', 'Tanker', 'RoRo', 'General Cargo']
    vessel_ages = np.random.randint(1, 30, n_samples)
    vessel_capacities = np.random.choice([5000, 10000, 15000, 20000, 25000], n_samples)
    
    # Route characteristics
    origins = ['Shanghai', 'Singapore', 'Rotterdam', 'Los Angeles', 'Hamburg', 
               'Hong Kong', 'Busan', 'Dubai', 'Mumbai', 'Tokyo']
    destinations = ['New York', 'London', 'Sydney', 'Santos', 'Vancouver',
                   'Cape Town', 'Melbourne', 'Barcelona', 'Houston', 'Auckland']
    
    # Operational features
    distances = np.random.uniform(2000, 20000, n_samples)
    fuel_consumption = np.random.uniform(20, 150, n_samples)
    crew_size = np.random.randint(15, 45, n_samples)
    cargo_weight = np.random.uniform(1000, 50000, n_samples)
    
    # Weather and seasonal factors
    seasons = np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples)
    weather_risk_score = np.random.uniform(0, 10, n_samples)
    sea_state = np.random.choice(['Calm', 'Moderate', 'Rough', 'Very Rough'], n_samples)
    
    # Port operations
    origin_port_congestion = np.random.uniform(0, 10, n_samples)
    dest_port_congestion = np.random.uniform(0, 10, n_samples)
    customs_complexity = np.random.uniform(1, 10, n_samples)
    
    # Time features
    departure_hour = np.random.randint(0, 24, n_samples)
    departure_day = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                     'Friday', 'Saturday', 'Sunday'], n_samples)
    
    # Economic factors
    fuel_price_index = np.random.uniform(0.8, 1.5, n_samples)
    freight_rate_index = np.random.uniform(0.7, 1.8, n_samples)
    
    # Maintenance and compliance
    days_since_maintenance = np.random.randint(0, 365, n_samples)
    safety_inspection_score = np.random.uniform(60, 100, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'vessel_type': np.random.choice(vessel_types, n_samples),
        'vessel_age_years': vessel_ages,
        'vessel_capacity_teu': vessel_capacities,
        'origin_port': np.random.choice(origins, n_samples),
        'destination_port': np.random.choice(destinations, n_samples),
        'distance_nautical_miles': distances,
        'fuel_consumption_tons_day': fuel_consumption,
        'crew_size': crew_size,
        'cargo_weight_tons': cargo_weight,
        'cargo_utilization_pct': (cargo_weight / vessel_capacities * 100).clip(0, 100),
        'season': seasons,
        'weather_risk_score': weather_risk_score,
        'sea_state': sea_state,
        'origin_port_congestion': origin_port_congestion,
        'dest_port_congestion': dest_port_congestion,
        'customs_complexity_score': customs_complexity,
        'departure_hour': departure_hour,
        'departure_day': departure_day,
        'fuel_price_index': fuel_price_index,
        'freight_rate_index': freight_rate_index,
        'days_since_maintenance': days_since_maintenance,
        'safety_inspection_score': safety_inspection_score,
        'number_of_port_calls': np.random.randint(1, 6, n_samples),
        'has_hazardous_cargo': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'is_peak_season': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Calculate target: DELAYED (1) or ON_TIME (0)
    delay_probability = (
        0.05 +
        0.15 * (data['vessel_age_years'] > 20).astype(int) +
        0.10 * (data['weather_risk_score'] > 7).astype(int) +
        0.12 * (data['sea_state'].isin(['Rough', 'Very Rough'])).astype(int) +
        0.08 * (data['origin_port_congestion'] > 7).astype(int) +
        0.08 * (data['dest_port_congestion'] > 7).astype(int) +
        0.10 * (data['customs_complexity_score'] > 7).astype(int) +
        0.07 * (data['days_since_maintenance'] > 300).astype(int) +
        0.06 * (data['cargo_utilization_pct'] > 90).astype(int) +
        0.05 * data['has_hazardous_cargo'] +
        0.04 * data['is_peak_season'] +
        0.05 * (data['safety_inspection_score'] < 75).astype(int)
    )
    
    delay_probability = delay_probability.clip(0, 0.9)
    data['delayed'] = (np.random.random(n_samples) < delay_probability).astype(int)
    
    # Add missing values
    missing_indices = np.random.choice(data.index, size=int(0.03 * n_samples), replace=False)
    data.loc[missing_indices, 'weather_risk_score'] = np.nan
    
    missing_indices = np.random.choice(data.index, size=int(0.02 * n_samples), replace=False)
    data.loc[missing_indices, 'fuel_consumption_tons_day'] = np.nan
    
    return data

# Generate dataset
df = generate_ship_logistics_data(n_samples=20000, random_state=42)
print(f"\n✓ Generated {len(df)} shipping records")
print(f"✓ Features: {df.shape[1] - 1} (excluding target)")
print(f"✓ Target: 'delayed' (0=On Time, 1=Delayed)")

df.to_csv('ship_logistics_raw_data.csv', index=False)
print("✓ Saved raw data to 'ship_logistics_raw_data.csv'")

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\n--- Data Overview ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())
print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n--- Target Distribution ---")
delay_counts = df['delayed'].value_counts()
print(delay_counts)
print(f"\nDelay Rate: {delay_counts[1] / len(df) * 100:.2f}%")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Exploratory Data Analysis - Ship Delay Prediction', fontsize=16, fontweight='bold')

# 1. Target distribution
axes[0, 0].bar(['On Time', 'Delayed'], delay_counts.values, color=['green', 'red'], alpha=0.7)
axes[0, 0].set_title('Target Distribution')
axes[0, 0].set_ylabel('Count')

# 2. Vessel age vs delay
df.boxplot(column='vessel_age_years', by='delayed', ax=axes[0, 1])
axes[0, 1].set_title('Vessel Age vs Delay')
axes[0, 1].set_xlabel('Delayed')
axes[0, 1].set_ylabel('Vessel Age (years)')

# 3. Weather risk vs delay
df.boxplot(column='weather_risk_score', by='delayed', ax=axes[0, 2])
axes[0, 2].set_title('Weather Risk vs Delay')
axes[0, 2].set_xlabel('Delayed')
axes[0, 2].set_ylabel('Weather Risk Score')

# 4. Port congestion comparison
congestion_data = df.groupby('delayed')[['origin_port_congestion', 'dest_port_congestion']].mean()
congestion_data.plot(kind='bar', ax=axes[1, 0], color=['blue', 'orange'])
axes[1, 0].set_title('Port Congestion by Delay Status')
axes[1, 0].set_xlabel('Delayed')
axes[1, 0].set_ylabel('Average Congestion')
axes[1, 0].legend(['Origin', 'Destination'])
axes[1, 0].set_xticklabels(['On Time', 'Delayed'], rotation=0)

# 5. Sea state distribution
sea_state_delay = pd.crosstab(df['sea_state'], df['delayed'], normalize='index') * 100
sea_state_delay.plot(kind='bar', ax=axes[1, 1], color=['green', 'red'], alpha=0.7)
axes[1, 1].set_title('Delay Rate by Sea State')
axes[1, 1].set_xlabel('Sea State')
axes[1, 1].set_ylabel('Percentage')
axes[1, 1].legend(['On Time', 'Delayed'])
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

# 6. Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_with_target = df[numeric_cols].corr()['delayed'].sort_values(ascending=False)
top_features = corr_with_target[1:11].index.tolist()
sns.heatmap(df[top_features + ['delayed']].corr(), annot=True, fmt='.2f', 
            cmap='coolwarm', ax=axes[1, 2], cbar_kws={'label': 'Correlation'})
axes[1, 2].set_title('Top Features Correlation with Delay')

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA visualizations saved to 'eda_visualization.png'")

print("\n--- Feature Importance Analysis ---")
print("Top 10 Features Correlated with Delays:")
print(corr_with_target[1:11])

# ============================================================================
# 4. DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 3: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 80)

df_processed = df.copy()

print("\n--- Feature Engineering ---")

# Create derived features
df_processed['vessel_efficiency'] = (
    df_processed['cargo_weight_tons'] / df_processed['fuel_consumption_tons_day']
)

df_processed['route_complexity'] = (
    df_processed['distance_nautical_miles'] * 
    df_processed['number_of_port_calls'] / 1000
)

df_processed['port_total_congestion'] = (
    df_processed['origin_port_congestion'] + 
    df_processed['dest_port_congestion']
)

df_processed['maintenance_risk'] = (
    (df_processed['days_since_maintenance'] > 180).astype(int)
)

df_processed['age_category'] = pd.cut(
    df_processed['vessel_age_years'], 
    bins=[0, 10, 20, 30], 
    labels=['New', 'Medium', 'Old']
)

df_processed['is_weekend_departure'] = (
    df_processed['departure_day'].isin(['Saturday', 'Sunday'])
).astype(int)

df_processed['high_utilization'] = (
    (df_processed['cargo_utilization_pct'] > 85).astype(int)
)

df_processed['weather_sea_risk'] = (
    df_processed['weather_risk_score'] * 
    df_processed['sea_state'].map({'Calm': 1, 'Moderate': 2, 'Rough': 3, 'Very Rough': 4})
)

print("✓ Created 8 new engineered features")

# Handle missing values
print("\n--- Handling Missing Values ---")
for col in df_processed.columns:
    if df_processed[col].isnull().sum() > 0:
        if df_processed[col].dtype in ['float64', 'int64']:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"✓ Filled {col} with median: {median_val:.2f}")

# Separate features and target
X = df_processed.drop('delayed', axis=1)
y = df_processed['delayed']

print(f"\n✓ Final feature count: {X.shape[1]}")
print(f"✓ Total samples: {X.shape[0]}")

# ============================================================================
# 5. TRAIN-TEST-VALIDATION SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 4: TRAIN-TEST-VALIDATION SPLIT")
print("=" * 80)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\nTarget distribution in splits:")
print(f"Train - On Time: {(y_train==0).sum()}, Delayed: {(y_train==1).sum()}")
print(f"Val   - On Time: {(y_val==0).sum()}, Delayed: {(y_val==1).sum()}")
print(f"Test  - On Time: {(y_test==0).sum()}, Delayed: {(y_test==1).sum()}")

# ============================================================================
# 6. FEATURE PREPROCESSING PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 5: FEATURE PREPROCESSING PIPELINE SETUP")
print("=" * 80)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("✓ Preprocessing pipeline created")
print("  - Numeric: Imputation (median) + Scaling (Standard)")
print("  - Categorical: Imputation (constant) + One-Hot Encoding")

# ============================================================================
# 7. TRADITIONAL ML MODEL SELECTION & EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 6: TRADITIONAL ML MODEL SELECTION & BASELINE EVALUATION")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

results = []

print("\nEvaluating baseline models (5-fold cross-validation)...")
print("-" * 80)

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1
    )
    
    pipeline.fit(X_train, y_train)
    
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    results.append({
        'Model': name,
        'CV F1 Mean': cv_scores.mean(),
        'CV F1 Std': cv_scores.std(),
        'Train Acc': train_acc,
        'Val Acc': val_acc,
        'Val Precision': val_precision,
        'Val Recall': val_recall,
        'Val F1': val_f1,
        'Val AUC': val_auc
    })
    
    print(f"{name:25s} | F1: {val_f1:.4f} | AUC: {val_auc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Val F1', ascending=False)

print("\n" + "=" * 80)
print("TRADITIONAL ML MODEL COMPARISON SUMMARY")
print("=" * 80)
print(results_df.to_string(index=False))

results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results saved to 'model_comparison_results.csv'")

# ============================================================================
# 8. HYPERPARAMETER TUNING (Best Traditional ML Model)
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 7: HYPERPARAMETER TUNING (TRADITIONAL ML)")
print("=" * 80)

best_model_name = results_df.iloc[0]['Model']
print(f"\nBest baseline model: {best_model_name}")
print(f"Baseline F1 Score: {results_df.iloc[0]['Val F1']:.4f}")

if 'Random Forest' in best_model_name:
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2']
    }
    base_model = RandomForestClassifier(random_state=42)
elif 'Gradient Boosting' in best_model_name:
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__subsample': [0.8, 1.0]
    }
    base_model = GradientBoostingClassifier(random_state=42)
else:
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    base_model = RandomForestClassifier(random_state=42)

print("\nPerforming Grid Search with Cross-Validation...")

tuning_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', base_model)
])

grid_search = GridSearchCV(
    tuning_pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n✓ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n✓ Best CV F1 Score: {grid_search.best_score_:.4f}")

# ============================================================================
# 9. DEEP LEARNING MODEL DEVELOPMENT
# ============================================================================

if not TENSORFLOW_AVAILABLE:
    print("\n" + "=" * 80)
    print("PHASE 8: DEEP LEARNING MODEL DEVELOPMENT - SKIPPED")
    print("=" * 80)
    print("\nTensorFlow is not installed. Skipping deep learning models.")
    print("Traditional ML models will be used for final evaluation.")
    
    # Skip to traditional ML final model
    best_ml_pipeline = grid_search.best_estimator_
    final_model_name = f"{best_model_name} (Traditional ML)"
    final_model_type = 'ML'
    
    y_test_ml_pred = best_ml_pipeline.predict(X_test)
    y_test_ml_proba = best_ml_pipeline.predict_proba(X_test)[:, 1]
    
    final_predictions = y_test_ml_pred
    final_probabilities = y_test_ml_proba
    
    # Create empty dataframes for consistency
    dl_results_df = pd.DataFrame()
    ensemble_results = []
    
else:
    print("\n" + "=" * 80)
    print("PHASE 8: DEEP LEARNING MODEL DEVELOPMENT")
    print("=" * 80)

    # Preprocess data for neural networks
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)

    input_dim = X_train_preprocessed.shape[1]
    print(f"\nInput dimension for neural network: {input_dim}")

    # Convert to numpy arrays
    y_train_np = y_train.values
    y_val_np = y_val.values
    y_test_np = y_test.values

    print("\n--- Building Deep Learning Models ---")

    # Model 1: Simple Dense Neural Network
    print("\n1. Simple Dense Neural Network")
    model_simple = keras_models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='SimpleNN')

    model_simple.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )

    print(model_simple.summary())

    # Model 2: Deep Neural Network with Batch Normalization
    print("\n2. Deep Neural Network with Batch Normalization")
    model_deep = keras_models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='DeepNN')

    model_deep.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )

    print(model_deep.summary())

    # Model 3: Residual Network (ResNet-style)
    print("\n3. Residual Neural Network")
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual block 1
    residual = x
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Dropout(0.3)(x)

    # Residual block 2
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model_resnet = keras_models.Model(inputs=inputs, outputs=outputs, name='ResNet')

    model_resnet.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )

    print(model_resnet.summary())

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    # Training
    print("\n--- Training Deep Learning Models ---")
    dl_results = {}

    # Train Simple NN
    print("\nTraining Simple NN...")
    history_simple = model_simple.fit(
        X_train_preprocessed, y_train_np,
        validation_data=(X_val_preprocessed, y_val_np),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    print("✓ Simple NN training completed")

    # Train Deep NN
    print("\nTraining Deep NN...")
    history_deep = model_deep.fit(
        X_train_preprocessed, y_train_np,
        validation_data=(X_val_preprocessed, y_val_np),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    print("✓ Deep NN training completed")

    # Train ResNet
    print("\nTraining ResNet...")
    history_resnet = model_resnet.fit(
        X_train_preprocessed, y_train_np,
        validation_data=(X_val_preprocessed, y_val_np),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    print("✓ ResNet training completed")

# Evaluate DL models
print("\n" + "=" * 80)
print("DEEP LEARNING MODELS EVALUATION")
print("=" * 80)

dl_models = {
    'Simple NN': model_simple,
    'Deep NN': model_deep,
    'ResNet': model_resnet
}

dl_results = []

for name, model in dl_models.items():
    # Predictions
    y_val_pred_proba = model.predict(X_val_preprocessed, verbose=0).flatten()
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)
    
    y_test_pred_proba = model.predict(X_test_preprocessed, verbose=0).flatten()
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    
    # Metrics
    val_acc = accuracy_score(y_val_np, y_val_pred)
    val_precision = precision_score(y_val_np, y_val_pred)
    val_recall = recall_score(y_val_np, y_val_pred)
    val_f1 = f1_score(y_val_np, y_val_pred)
    val_auc = roc_auc_score(y_val_np, y_val_pred_proba)
    
    test_acc = accuracy_score(y_test_np, y_test_pred)
    test_precision = precision_score(y_test_np, y_test_pred)
    test_recall = recall_score(y_test_np, y_test_pred)
    test_f1 = f1_score(y_test_np, y_test_pred)
    test_auc = roc_auc_score(y_test_np, y_test_pred_proba)
    
    dl_results.append({
        'Model': name,
        'Val Acc': val_acc,
        'Val Precision': val_precision,
        'Val Recall': val_recall,
        'Val F1': val_f1,
        'Val AUC': val_auc,
        'Test Acc': test_acc,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test F1': test_f1,
        'Test AUC': test_auc
    })
    
    print(f"\n{name}:")
    print(f"  Validation - F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
    print(f"  Test       - F1: {test_f1:.4f} | AUC: {test_auc:.4f}")

dl_results_df = pd.DataFrame(dl_results)
dl_results_df = dl_results_df.sort_values('Test F1', ascending=False)

print("\n" + "=" * 80)
print("DEEP LEARNING MODEL COMPARISON")
print("=" * 80)
print(dl_results_df.to_string(index=False))

dl_results_df.to_csv('dl_model_comparison_results.csv', index=False)
print("\n✓ DL results saved to 'dl_model_comparison_results.csv'")

# Training history visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Deep Learning Training History', fontsize=16, fontweight='bold')

histories = {
    'Simple NN': history_simple,
    'Deep NN': history_deep,
    'ResNet': history_resnet
}

for idx, (name, history) in enumerate(histories.items()):
    # Loss
    axes[0, idx].plot(history.history['loss'], label='Train Loss')
    axes[0, idx].plot(history.history['val_loss'], label='Val Loss')
    axes[0, idx].set_title(f'{name} - Loss')
    axes[0, idx].set_xlabel('Epoch')
    axes[0, idx].set_ylabel('Loss')
    axes[0, idx].legend()
    axes[0, idx].grid(True, alpha=0.3)
    
    # AUC - handle different possible metric names
    auc_key = None
    for key in history.history.keys():
        if 'auc' in key.lower() and 'val' not in key:
            auc_key = key
            break
    
    val_auc_key = None
    for key in history.history.keys():
        if 'auc' in key.lower() and 'val' in key:
            val_auc_key = key
            break
    
    if auc_key and val_auc_key:
        axes[1, idx].plot(history.history[auc_key], label='Train AUC')
        axes[1, idx].plot(history.history[val_auc_key], label='Val AUC')
        axes[1, idx].set_title(f'{name} - AUC')
        axes[1, idx].set_xlabel('Epoch')
        axes[1, idx].set_ylabel('AUC')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)
    else:
        # Fallback to accuracy if AUC not found
        axes[1, idx].plot(history.history['accuracy'], label='Train Accuracy')
        axes[1, idx].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[1, idx].set_title(f'{name} - Accuracy')
        axes[1, idx].set_xlabel('Epoch')
        axes[1, idx].set_ylabel('Accuracy')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dl_training_history.png', dpi=300, bbox_inches='tight')
print("✓ DL training history saved to 'dl_training_history.png'")

# ============================================================================
# 10. MODEL COMPARISON: TRADITIONAL ML vs DEEP LEARNING
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 9: COMPREHENSIVE MODEL COMPARISON (ML vs DL)")
print("=" * 80)

# Get best traditional ML model performance
best_ml_pipeline = grid_search.best_estimator_
y_test_ml_pred = best_ml_pipeline.predict(X_test)
y_test_ml_proba = best_ml_pipeline.predict_proba(X_test)[:, 1]

ml_test_acc = accuracy_score(y_test, y_test_ml_pred)
ml_test_precision = precision_score(y_test, y_test_ml_pred)
ml_test_recall = recall_score(y_test, y_test_ml_pred)
ml_test_f1 = f1_score(y_test, y_test_ml_pred)
ml_test_auc = roc_auc_score(y_test, y_test_ml_proba)

# Compare all models
comparison_data = [{
    'Model Type': 'Traditional ML',
    'Model': best_model_name,
    'Test Accuracy': ml_test_acc,
    'Test Precision': ml_test_precision,
    'Test Recall': ml_test_recall,
    'Test F1': ml_test_f1,
    'Test AUC': ml_test_auc
}]

for _, row in dl_results_df.iterrows():
    comparison_data.append({
        'Model Type': 'Deep Learning',
        'Model': row['Model'],
        'Test Accuracy': row['Test Acc'],
        'Test Precision': row['Test Precision'],
        'Test Recall': row['Test Recall'],
        'Test F1': row['Test F1'],
        'Test AUC': row['Test AUC']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test F1', ascending=False)

print("\n--- All Models Comparison (Test Set Performance) ---")
print(comparison_df.to_string(index=False))

comparison_df.to_csv('all_models_comparison.csv', index=False)
print("\n✓ Comparison saved to 'all_models_comparison.csv'")

# Determine best overall model
best_overall_idx = comparison_df['Test F1'].idxmax()
best_overall_model = comparison_df.loc[best_overall_idx]

print(f"\n{'='*80}")
print(f"BEST OVERALL MODEL: {best_overall_model['Model']} ({best_overall_model['Model Type']})")
print(f"{'='*80}")
print(f"Test F1 Score: {best_overall_model['Test F1']:.4f}")
print(f"Test AUC: {best_overall_model['Test AUC']:.4f}")

# Visualization: Model comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Traditional ML vs Deep Learning Comparison', fontsize=16, fontweight='bold')

# Metric comparison
metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test AUC']
x_pos = np.arange(len(metrics))
width = 0.15

for idx, row in comparison_df.iterrows():
    values = [row[m] for m in metrics]
    axes[0].bar(x_pos + idx*width, values, width, 
                label=f"{row['Model']} ({row['Model Type'][:2]})", alpha=0.8)

axes[0].set_ylabel('Score')
axes[0].set_title('Performance Metrics Comparison')
axes[0].set_xticks(x_pos + width * (len(comparison_df)-1)/2)
axes[0].set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], rotation=45)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 1])

# ROC curves comparison
for _, row in comparison_df.iterrows():
    if row['Model Type'] == 'Traditional ML':
        y_proba = y_test_ml_proba
    else:
        if row['Model'] == 'Simple NN':
            y_proba = model_simple.predict(X_test_preprocessed, verbose=0).flatten()
        elif row['Model'] == 'Deep NN':
            y_proba = model_deep.predict(X_test_preprocessed, verbose=0).flatten()
        else:
            y_proba = model_resnet.predict(X_test_preprocessed, verbose=0).flatten()
    
    fpr, tpr, _ = roc_curve(y_test_np, y_proba)
    axes[1].plot(fpr, tpr, lw=2, 
                label=f"{row['Model']} (AUC={row['Test AUC']:.3f})")

axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves - All Models')
axes[1].legend(loc="lower right", fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_vs_dl_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Comparison visualization saved to 'ml_vs_dl_comparison.png'")

# ============================================================================
# 11. ENSEMBLE: COMBINING ML AND DL PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 10: ENSEMBLE MODEL (ML + DL FUSION)")
print("=" * 80)

print("\n--- Creating Ensemble Predictions ---")

# Get predictions from all models
ml_pred_proba = y_test_ml_proba
simple_nn_proba = model_simple.predict(X_test_preprocessed, verbose=0).flatten()
deep_nn_proba = model_deep.predict(X_test_preprocessed, verbose=0).flatten()
resnet_proba = model_resnet.predict(X_test_preprocessed, verbose=0).flatten()

# Ensemble strategies
# 1. Simple average
ensemble_avg_proba = (ml_pred_proba + simple_nn_proba + deep_nn_proba + resnet_proba) / 4
ensemble_avg_pred = (ensemble_avg_proba > 0.5).astype(int)

# 2. Weighted average (based on validation F1 scores)
ml_weight = results_df.iloc[0]['Val F1']
weights = []
for _, row in dl_results_df.iterrows():
    weights.append(row['Val F1'])

total_weight = ml_weight + sum(weights)
ensemble_weighted_proba = (
    ml_pred_proba * ml_weight +
    simple_nn_proba * weights[dl_results_df[dl_results_df['Model']=='Simple NN'].index[0]] +
    deep_nn_proba * weights[dl_results_df[dl_results_df['Model']=='Deep NN'].index[0]] +
    resnet_proba * weights[dl_results_df[dl_results_df['Model']=='ResNet'].index[0]]
) / total_weight
ensemble_weighted_pred = (ensemble_weighted_proba > 0.5).astype(int)

# 3. Voting (majority)
predictions_matrix = np.column_stack([
    y_test_ml_pred,
    (simple_nn_proba > 0.5).astype(int),
    (deep_nn_proba > 0.5).astype(int),
    (resnet_proba > 0.5).astype(int)
])
ensemble_voting_pred = (np.mean(predictions_matrix, axis=1) > 0.5).astype(int)

# Evaluate ensembles
print("\nEnsemble Performance:")

ensemble_results = []

for name, pred, proba in [
    ('Simple Average', ensemble_avg_pred, ensemble_avg_proba),
    ('Weighted Average', ensemble_weighted_pred, ensemble_weighted_proba),
    ('Majority Voting', ensemble_voting_pred, ensemble_avg_proba)
]:
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    
    ensemble_results.append({
        'Ensemble Type': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC': auc
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")

ensemble_results_df = pd.DataFrame(ensemble_results)
ensemble_results_df.to_csv('ensemble_results.csv', index=False)
print("\n✓ Ensemble results saved to 'ensemble_results.csv'")

# ============================================================================
# 12. FINAL MODEL SELECTION AND EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 11: FINAL MODEL SELECTION AND COMPREHENSIVE EVALUATION")
print("=" * 80)

# Select best model based on Test F1
best_f1 = max(
    ml_test_f1,
    max([r['Test F1'] for r in dl_results]),
    max([r['F1 Score'] for r in ensemble_results])
)

if ml_test_f1 == best_f1:
    final_model_name = f"{best_model_name} (Traditional ML)"
    final_predictions = y_test_ml_pred
    final_probabilities = y_test_ml_proba
    final_model_type = 'ML'
elif max([r['F1 Score'] for r in ensemble_results]) == best_f1:
    best_ensemble = max(ensemble_results, key=lambda x: x['F1 Score'])
    final_model_name = f"{best_ensemble['Ensemble Type']} Ensemble"
    if 'Simple' in best_ensemble['Ensemble Type']:
        final_predictions = ensemble_avg_pred
        final_probabilities = ensemble_avg_proba
    elif 'Weighted' in best_ensemble['Ensemble Type']:
        final_predictions = ensemble_weighted_pred
        final_probabilities = ensemble_weighted_proba
    else:
        final_predictions = ensemble_voting_pred
        final_probabilities = ensemble_avg_proba
    final_model_type = 'ENSEMBLE'
else:
    best_dl = max(dl_results, key=lambda x: x['Test F1'])
    final_model_name = f"{best_dl['Model']} (Deep Learning)"
    if 'Simple' in best_dl['Model']:
        final_predictions = (model_simple.predict(X_test_preprocessed, verbose=0).flatten() > 0.5).astype(int)
        final_probabilities = model_simple.predict(X_test_preprocessed, verbose=0).flatten()
    elif 'Deep' in best_dl['Model']:
        final_predictions = (model_deep.predict(X_test_preprocessed, verbose=0).flatten() > 0.5).astype(int)
        final_probabilities = model_deep.predict(X_test_preprocessed, verbose=0).flatten()
    else:
        final_predictions = (model_resnet.predict(X_test_preprocessed, verbose=0).flatten() > 0.5).astype(int)
        final_probabilities = model_resnet.predict(X_test_preprocessed, verbose=0).flatten()
    final_model_type = 'DL'

print(f"\nFINAL SELECTED MODEL: {final_model_name}")
print(f"Test F1 Score: {best_f1:.4f}")

# Comprehensive evaluation
final_acc = accuracy_score(y_test, final_predictions)
final_prec = precision_score(y_test, final_predictions)
final_rec = recall_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions)
final_auc = roc_auc_score(y_test, final_probabilities)

print("\n--- Final Model Performance (Test Set) ---")
print(f"Accuracy:  {final_acc:.4f}")
print(f"Precision: {final_prec:.4f}")
print(f"Recall:    {final_rec:.4f}")
print(f"F1 Score:  {final_f1:.4f}")
print(f"ROC AUC:   {final_auc:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, final_predictions, 
                          target_names=['On Time', 'Delayed']))

# Confusion Matrix
cm = confusion_matrix(y_test, final_predictions)
print("\n--- Confusion Matrix ---")
print(f"                 Predicted")
print(f"                 On Time  Delayed")
print(f"Actual On Time    {cm[0][0]:<8} {cm[0][1]:<8}")
print(f"Actual Delayed    {cm[1][0]:<8} {cm[1][1]:<8}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'Final Model Evaluation: {final_model_name}', 
             fontsize=16, fontweight='bold')

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['On Time', 'Delayed'],
            yticklabels=['On Time', 'Delayed'])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, final_probabilities)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {final_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# 3. Prediction distribution
axes[1, 0].hist([final_probabilities[y_test==0], final_probabilities[y_test==1]], 
                bins=30, label=['On Time', 'Delayed'], alpha=0.7, color=['green', 'red'])
axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
axes[1, 0].set_xlabel('Predicted Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Prediction Probability Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Feature Importance (if applicable)
if final_model_type == 'ML' and hasattr(best_ml_pipeline.named_steps['classifier'], 'feature_importances_'):
    preprocessor_fitted = best_ml_pipeline.named_steps['preprocessor']
    feature_names_all = (numeric_features + 
                        list(preprocessor_fitted.named_transformers_['cat']
                             .named_steps['onehot']
                             .get_feature_names_out(categorical_features)))
    
    importances = best_ml_pipeline.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[-15:]
    
    axes[1, 1].barh(range(len(indices)), importances[indices], align='center')
    axes[1, 1].set_yticks(range(len(indices)))
    axes[1, 1].set_yticklabels([feature_names_all[i][:30] for i in indices], fontsize=8)
    axes[1, 1].set_xlabel('Feature Importance')
    axes[1, 1].set_title('Top 15 Most Important Features')
else:
    axes[1, 1].text(0.5, 0.5, 
                   f'Feature importance visualization\nnot available for {final_model_type} models', 
                   ha='center', va='center', fontsize=12)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('final_model_comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Final evaluation plots saved to 'final_model_comprehensive_evaluation.png'")

# ============================================================================
# 13. MODEL PERSISTENCE (JSON + H5 + JOBLIB)
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 12: MODEL PERSISTENCE (JSON + H5 + JOBLIB)")
print("=" * 80)

# Save Traditional ML model with joblib
joblib.dump(best_ml_pipeline, 'ship_delay_ml_model.joblib')
print("✓ Traditional ML model saved to 'ship_delay_ml_model.joblib'")

# Save Deep Learning models in H5 format
model_simple.save('ship_delay_simple_nn.h5')
print("✓ Simple NN saved to 'ship_delay_simple_nn.h5'")

model_deep.save('ship_delay_deep_nn.h5')
print("✓ Deep NN saved to 'ship_delay_deep_nn.h5'")

model_resnet.save('ship_delay_resnet.h5')
print("✓ ResNet saved to 'ship_delay_resnet.h5'")

# Save preprocessor
joblib.dump(preprocessor, 'preprocessor.joblib')
print("✓ Preprocessor saved to 'preprocessor.joblib'")

# Save metadata as JSON
metadata = {
    'project_info': {
        'name': 'Ship Delivery Delay Prediction',
        'version': '2.0',
        'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'ML + Deep Learning pipeline for maritime logistics'
    },
    'data_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'delay_rate': float(delay_counts[1] / len(df)),
        'num_features': X.shape[1],
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    },
    'traditional_ml': {
        'best_model': best_model_name,
        'best_params': {k: str(v) for k, v in grid_search.best_params_.items()},
        'test_metrics': {
            'accuracy': float(ml_test_acc),
            'precision': float(ml_test_precision),
            'recall': float(ml_test_recall),
            'f1_score': float(ml_test_f1),
            'roc_auc': float(ml_test_auc)
        }
    },
    'deep_learning': {
        'models': []
    },
    'ensemble': {
        'strategies': ensemble_results
    },
    'final_model': {
        'name': final_model_name,
        'type': final_model_type,
        'test_metrics': {
            'accuracy': float(final_acc),
            'precision': float(final_prec),
            'recall': float(final_rec),
            'f1_score': float(final_f1),
            'roc_auc': float(final_auc)
        }
    }
}

# Add DL model info
for _, row in dl_results_df.iterrows():
    metadata['deep_learning']['models'].append({
        'name': row['Model'],
        'test_metrics': {
            'accuracy': float(row['Test Acc']),
            'precision': float(row['Test Precision']),
            'recall': float(row['Test Recall']),
            'f1_score': float(row['Test F1']),
            'roc_auc': float(row['Test AUC'])
        }
    })

# Save metadata
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("✓ Model metadata saved to 'model_metadata.json'")

# Save feature info
feature_info = {
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'all_features': list(X.columns),
    'input_dim_after_preprocessing': int(input_dim)
}

with open('feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=4)
print("✓ Feature info saved to 'feature_info.json'")

# ============================================================================
# 14. PREDICTION FUNCTIONS
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 13: INFERENCE & SAMPLE PREDICTIONS")
print("=" * 80)

def predict_with_ml(pipeline, shipment_data):
    """Predict using Traditional ML model"""
    if isinstance(shipment_data, dict):
        shipment_df = pd.DataFrame([shipment_data])
    else:
        shipment_df = shipment_data
    
    prediction = pipeline.predict(shipment_df)[0]
    probability = pipeline.predict_proba(shipment_df)[0]
    
    return {
        'prediction': 'DELAYED' if prediction == 1 else 'ON TIME',
        'delay_probability': float(probability[1]),
        'confidence': float(max(probability)),
        'model_type': 'Traditional ML'
    }

def predict_with_dl(model, preprocessor, shipment_data):
    """Predict using Deep Learning model"""
    if isinstance(shipment_data, dict):
        shipment_df = pd.DataFrame([shipment_data])
    else:
        shipment_df = shipment_data
    
    shipment_preprocessed = preprocessor.transform(shipment_df)
    probability = model.predict(shipment_preprocessed, verbose=0).flatten()[0]
    prediction = int(probability > 0.5)
    
    return {
        'prediction': 'DELAYED' if prediction == 1 else 'ON TIME',
        'delay_probability': float(probability),
        'confidence': float(max(probability, 1-probability)),
        'model_type': 'Deep Learning'
    }

def predict_ensemble(ml_pipeline, dl_models, preprocessor, shipment_data, strategy='weighted'):
    """Predict using ensemble of ML and DL models"""
    if isinstance(shipment_data, dict):
        shipment_df = pd.DataFrame([shipment_data])
    else:
        shipment_df = shipment_data
    
    # ML prediction
    ml_proba = ml_pipeline.predict_proba(shipment_df)[0][1]
    
    # DL predictions
    shipment_preprocessed = preprocessor.transform(shipment_df)
    dl_probas = [model.predict(shipment_preprocessed, verbose=0).flatten()[0] 
                 for model in dl_models]
    
    # Ensemble
    if strategy == 'simple':
        final_proba = (ml_proba + sum(dl_probas)) / (1 + len(dl_probas))
    elif strategy == 'weighted':
        # Use validation F1 scores as weights
        ml_weight = results_df.iloc[0]['Val F1']
        dl_weights = [r['Val F1'] for r in dl_results]
        total_weight = ml_weight + sum(dl_weights)
        final_proba = (ml_proba * ml_weight + sum(p*w for p, w in zip(dl_probas, dl_weights))) / total_weight
    else:  # voting
        predictions = [int(ml_proba > 0.5)] + [int(p > 0.5) for p in dl_probas]
        final_prediction = int(sum(predictions) > len(predictions)/2)
        final_proba = sum([ml_proba] + dl_probas) / (1 + len(dl_probas))
        
        return {
            'prediction': 'DELAYED' if final_prediction == 1 else 'ON TIME',
            'delay_probability': float(final_proba),
            'confidence': float(max(final_proba, 1-final_proba)),
            'model_type': 'Ensemble (Voting)'
        }
    
    final_prediction = int(final_proba > 0.5)
    
    return {
        'prediction': 'DELAYED' if final_prediction == 1 else 'ON TIME',
        'delay_probability': float(final_proba),
        'confidence': float(max(final_proba, 1-final_proba)),
        'model_type': f'Ensemble ({strategy.capitalize()})'
    }

# Test samples
print("\n--- Sample Predictions ---\n")

sample_1 = {
    'vessel_type': 'Container Ship',
    'vessel_age_years': 25,
    'vessel_capacity_teu': 20000,
    'origin_port': 'Shanghai',
    'destination_port': 'New York',
    'distance_nautical_miles': 11500,
    'fuel_consumption_tons_day': 120,
    'crew_size': 25,
    'cargo_weight_tons': 18000,
    'cargo_utilization_pct': 90,
    'season': 'Winter',
    'weather_risk_score': 8.5,
    'sea_state': 'Very Rough',
    'origin_port_congestion': 8.2,
    'dest_port_congestion': 7.5,
    'customs_complexity_score': 8.0,
    'departure_hour': 14,
    'departure_day': 'Monday',
    'fuel_price_index': 1.4,
    'freight_rate_index': 1.6,
    'days_since_maintenance': 320,
    'safety_inspection_score': 72,
    'number_of_port_calls': 4,
    'has_hazardous_cargo': 1,
    'is_peak_season': 1,
    'vessel_efficiency': 150,
    'route_complexity': 46,
    'port_total_congestion': 15.7,
    'maintenance_risk': 1,
    'age_category': 'Old',
    'is_weekend_departure': 0,
    'high_utilization': 1,
    'weather_sea_risk': 34
}

sample_2 = {
    'vessel_type': 'Container Ship',
    'vessel_age_years': 5,
    'vessel_capacity_teu': 15000,
    'origin_port': 'Singapore',
    'destination_port': 'Sydney',
    'distance_nautical_miles': 3900,
    'fuel_consumption_tons_day': 50,
    'crew_size': 30,
    'cargo_weight_tons': 8000,
    'cargo_utilization_pct': 53,
    'season': 'Spring',
    'weather_risk_score': 2.5,
    'sea_state': 'Calm',
    'origin_port_congestion': 3.0,
    'dest_port_congestion': 2.5,
    'customs_complexity_score': 3.0,
    'departure_hour': 8,
    'departure_day': 'Wednesday',
    'fuel_price_index': 1.0,
    'freight_rate_index': 1.1,
    'days_since_maintenance': 45,
    'safety_inspection_score': 95,
    'number_of_port_calls': 2,
    'has_hazardous_cargo': 0,
    'is_peak_season': 0,
    'vessel_efficiency': 160,
    'route_complexity': 7.8,
    'port_total_congestion': 5.5,
    'maintenance_risk': 0,
    'age_category': 'New',
    'is_weekend_departure': 0,
    'high_utilization': 0,
    'weather_sea_risk': 2.5
}

print("Sample 1: HIGH-RISK SHIPMENT")
print("  Vessel: Old container ship (25 years)")
print("  Conditions: Very rough seas, high congestion, winter")

ml_pred_1 = predict_with_ml(best_ml_pipeline, sample_1)
dl_pred_1 = predict_with_dl(model_resnet, preprocessor, sample_1)
ensemble_pred_1 = predict_ensemble(best_ml_pipeline, [model_simple, model_deep, model_resnet], 
                                   preprocessor, sample_1, 'weighted')

print(f"\n  Traditional ML: {ml_pred_1['prediction']} "
      f"(Probability: {ml_pred_1['delay_probability']:.2%})")
print(f"  Deep Learning:  {dl_pred_1['prediction']} "
      f"(Probability: {dl_pred_1['delay_probability']:.2%})")
print(f"  Ensemble:       {ensemble_pred_1['prediction']} "
      f"(Probability: {ensemble_pred_1['delay_probability']:.2%})")

print("\n" + "-"*80)

print("\nSample 2: LOW-RISK SHIPMENT")
print("  Vessel: New container ship (5 years)")
print("  Conditions: Calm seas, low congestion, spring")

ml_pred_2 = predict_with_ml(best_ml_pipeline, sample_2)
dl_pred_2 = predict_with_dl(model_resnet, preprocessor, sample_2)
ensemble_pred_2 = predict_ensemble(best_ml_pipeline, [model_simple, model_deep, model_resnet], 
                                   preprocessor, sample_2, 'weighted')

print(f"\n  Traditional ML: {ml_pred_2['prediction']} "
      f"(Probability: {ml_pred_2['delay_probability']:.2%})")
print(f"  Deep Learning:  {dl_pred_2['prediction']} "
      f"(Probability: {dl_pred_2['delay_probability']:.2%})")
print(f"  Ensemble:       {ensemble_pred_2['prediction']} "
      f"(Probability: {ensemble_pred_2['delay_probability']:.2%})")

# Batch predictions
print("\n" + "="*80)
print("--- Batch Prediction Examples (First 10 Test Samples) ---")
print("="*80 + "\n")

sample_batch = X_test.head(10)
ml_batch_preds = best_ml_pipeline.predict(sample_batch)
ml_batch_proba = best_ml_pipeline.predict_proba(sample_batch)[:, 1]

results_table = pd.DataFrame({
    'Vessel_Type': sample_batch['vessel_type'].values,
    'Age': sample_batch['vessel_age_years'].values,
    'Weather': sample_batch['weather_risk_score'].values,
    'Congestion': (sample_batch['origin_port_congestion'] + 
                   sample_batch['dest_port_congestion']).values,
    'Actual': y_test.head(10).map({0: 'On Time', 1: 'Delayed'}).values,
    'ML_Pred': ['Delayed' if p == 1 else 'On Time' for p in ml_batch_preds],
    'ML_Prob': [f"{p:.2%}" for p in ml_batch_proba]
})

print(results_table.to_string(index=False))

# ============================================================================
# 15. DIMENSIONALITY REDUCTION (PCA) ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 14: DIMENSIONALITY REDUCTION ANALYSIS (PCA)")
print("=" * 80)

print(f"\nOriginal feature dimensions: {X_train_preprocessed.shape[1]}")

# Apply PCA
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_preprocessed)
X_test_pca = pca.transform(X_test_preprocessed)

print(f"Reduced feature dimensions: {X_train_pca.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Train RF model with PCA features
pca_rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
pca_rf_model.fit(X_train_pca, y_train)
y_test_pred_pca = pca_rf_model.predict(X_test_pca)

pca_accuracy = accuracy_score(y_test, y_test_pred_pca)
pca_f1 = f1_score(y_test, y_test_pred_pca)

print(f"\nPCA + Random Forest Performance:")
print(f"  Accuracy: {pca_accuracy:.4f}")
print(f"  F1 Score: {pca_f1:.4f}")

# Train DL model with PCA features
print("\nTraining Deep Learning model with PCA features...")
pca_input_dim = X_train_pca.shape[1]

model_pca_dl = keras_models.Sequential([
    layers.Input(shape=(pca_input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
], name='PCA_DL')

model_pca_dl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)

history_pca_dl = model_pca_dl.fit(
    X_train_pca, y_train_np,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

y_test_pred_pca_dl = (model_pca_dl.predict(X_test_pca, verbose=0).flatten() > 0.5).astype(int)
pca_dl_accuracy = accuracy_score(y_test, y_test_pred_pca_dl)
pca_dl_f1 = f1_score(y_test, y_test_pred_pca_dl)

print(f"\nPCA + Deep Learning Performance:")
print(f"  Accuracy: {pca_dl_accuracy:.4f}")
print(f"  F1 Score: {pca_dl_f1:.4f}")

# Visualize PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PCA Analysis', fontsize=14, fontweight='bold')

# Variance explained
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
axes[0].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('PCA Variance Explained')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2D projection
pca_2d = PCA(n_components=2, random_state=42)
X_train_2d = pca_2d.fit_transform(X_train_preprocessed)
scatter = axes[1].scatter(X_train_2d[:, 0], X_train_2d[:, 1], 
                         c=y_train, cmap='RdYlGn', alpha=0.6, s=10)
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
axes[1].set_title('2D PCA Projection')
plt.colorbar(scatter, ax=axes[1], label='Delayed')

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ PCA analysis plots saved to 'pca_analysis.png'")

# ============================================================================
# 16. BUSINESS INSIGHTS & RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 15: BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 80)

# Feature importance from best ML model
if hasattr(best_ml_pipeline.named_steps['classifier'], 'feature_importances_'):
    preprocessor_fitted = best_ml_pipeline.named_steps['preprocessor']
    feature_names_all = (numeric_features + 
                        list(preprocessor_fitted.named_transformers_['cat']
                             .named_steps['onehot']
                             .get_feature_names_out(categorical_features)))
    
    importances = best_ml_pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names_all,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n--- Top 15 Most Important Features ---")
    print(feature_importance_df.head(15).to_string(index=False))
    
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print("\n✓ Feature importance saved to 'feature_importance.csv'")
    
    # Business recommendations
    print("\n--- Key Delay Risk Factors & Recommendations ---")
    
    recommendations = {
        'weather_risk': "Monitor weather forecasts and implement dynamic routing during high-risk periods",
        'dest_port_congestion': "Develop port congestion prediction models and schedule flexibility",
        'origin_port_congestion': "Coordinate with port authorities for priority berthing slots",
        'vessel_age': "Prioritize newer vessels for time-sensitive shipments",
        'days_since_maintenance': "Implement predictive maintenance schedules to reduce breakdown risks",
        'sea_state': "Use real-time ocean condition monitoring for route optimization",
        'customs_complexity': "Pre-clear documentation for complex customs situations",
        'cargo_utilization': "Balance loading efficiency with operational safety margins",
        'port_total_congestion': "Develop multi-port routing strategies to avoid bottlenecks",
        'weather_sea_risk': "Combined weather-sea monitoring for comprehensive risk assessment"
    }
    
    print("\nActionable Business Recommendations:")
    top_features = feature_importance_df.head(10)['Feature'].tolist()
    rec_count = 1
    for feature in top_features:
        for key, recommendation in recommendations.items():
            if key.replace('_', '') in feature.lower().replace('_', ''):
                print(f"{rec_count}. {recommendation}")
                rec_count += 1
                break
        if rec_count > 5:
            break

# ============================================================================
# 17. MODEL LOADING & VERIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 16: MODEL DEPLOYMENT VERIFICATION")
print("=" * 80)

print("\n--- Loading Saved Models ---")

# Load metadata
with open('model_metadata.json', 'r') as f:
    loaded_metadata = json.load(f)
print("✓ Metadata loaded successfully")
print(f"  Project: {loaded_metadata['project_info']['name']}")
print(f"  Version: {loaded_metadata['project_info']['version']}")
print(f"  Final Model: {loaded_metadata['final_model']['name']}")

# Load Traditional ML model
loaded_ml_pipeline = joblib.load('ship_delay_ml_model.joblib')
print("\n✓ Traditional ML model loaded successfully")

# Load Deep Learning models
loaded_simple_nn = keras.models.load_model('ship_delay_simple_nn.h5')
loaded_deep_nn = keras.models.load_model('ship_delay_deep_nn.h5')
loaded_resnet = keras.models.load_model('ship_delay_resnet.h5')
print("✓ Deep Learning models loaded successfully")

# Load preprocessor
loaded_preprocessor = joblib.load('preprocessor.joblib')
print("✓ Preprocessor loaded successfully")

# Verification test
print("\n--- Verification Test ---")
test_sample = X_test.iloc[0:1]

original_ml_pred = best_ml_pipeline.predict(test_sample)[0]
loaded_ml_pred = loaded_ml_pipeline.predict(test_sample)[0]

test_preprocessed = preprocessor.transform(test_sample)
original_dl_pred = (model_resnet.predict(test_preprocessed, verbose=0).flatten()[0] > 0.5).astype(int)
loaded_dl_pred = (loaded_resnet.predict(test_preprocessed, verbose=0).flatten()[0] > 0.5).astype(int)

print(f"\nTraditional ML:")
print(f"  Original Prediction: {original_ml_pred}")
print(f"  Loaded Prediction:   {loaded_ml_pred}")
print(f"  Match: {'✓ PASS' if original_ml_pred == loaded_ml_pred else '✗ FAIL'}")

print(f"\nDeep Learning:")
print(f"  Original Prediction: {original_dl_pred}")
print(f"  Loaded Prediction:   {loaded_dl_pred}")
print(f"  Match: {'✓ PASS' if original_dl_pred == loaded_dl_pred else '✗ FAIL'}")

# ============================================================================
# 18. FINAL PROJECT REPORT
# ============================================================================

print("\n" + "=" * 80)
print("PROJECT SUMMARY & COMPREHENSIVE PERFORMANCE REPORT")
print("=" * 80)

report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            SHIP DELIVERY DELAY PREDICTION - ML + DL PIPELINE                 ║
║                     COMPREHENSIVE PROJECT REPORT                             ║
║                             ~ Ryan Tusi                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 PROJECT OVERVIEW
─────────────────────────────────────────────────────────────────────────────────
Domain:          Maritime Logistics & Supply Chain
Problem Type:    Binary Classification (Delayed vs On Time)
Dataset Size:    {len(df)} shipments
Features:        {X.shape[1]} (original + engineered)
Target Balance:  {(y==1).sum()} delayed ({(y==1).sum()/len(y)*100:.1f}%), {(y==0).sum()} on-time

🔧 ML PIPELINE COMPONENTS
─────────────────────────────────────────────────────────────────────────────────
1. Data Extraction:      Synthetic maritime logistics data
2. Feature Engineering:  8 derived features (efficiency, risk scores, categories)
3. Preprocessing:        Numeric (median imputation + scaling) + Categorical (OHE)
4. Train/Val/Test Split: {len(X_train)}/{len(X_val)}/{len(X_test)} samples

🤖 TRADITIONAL ML MODELS EVALUATED
─────────────────────────────────────────────────────────────────────────────────
• Logistic Regression
• Decision Tree
• Random Forest
• Gradient Boosting
• Support Vector Machine (SVM)
• K-Nearest Neighbors

Best Model: {best_model_name}
Grid Search: 5-Fold Cross-Validation
Best F1:     {grid_search.best_score_:.4f}

🧠 DEEP LEARNING MODELS DEVELOPED
─────────────────────────────────────────────────────────────────────────────────
• Simple Dense Neural Network (3 hidden layers)
• Deep Neural Network with Batch Normalization (4 hidden layers)
• Residual Neural Network (ResNet-style architecture)

Training:    Early stopping + Learning rate reduction
Epochs:      Up to 100 (with early stopping)
Batch Size:  32

Best DL Model: {dl_results_df.iloc[0]['Model']}
Test F1:       {dl_results_df.iloc[0]['Test F1']:.4f}

🔗 ENSEMBLE METHODS
─────────────────────────────────────────────────────────────────────────────────
• Simple Average (ML + DL)
• Weighted Average (based on validation F1 scores)
• Majority Voting

Best Ensemble: {max(ensemble_results, key=lambda x: x['F1 Score'])['Ensemble Type']}
Test F1:       {max(ensemble_results, key=lambda x: x['F1 Score'])['F1 Score']:.4f}

📈 FINAL MODEL PERFORMANCE (Test Set)
─────────────────────────────────────────────────────────────────────────────────
Selected Model: {final_model_name}

Accuracy:       {final_acc:.4f}
Precision:      {final_prec:.4f} ({final_prec*100:.1f}% of predicted delays are actual)
Recall:         {final_rec:.4f} (captures {final_rec*100:.1f}% of all actual delays)
F1 Score:       {final_f1:.4f}
ROC AUC:        {final_auc:.4f}

Confusion Matrix:
                 Predicted On Time    Predicted Delayed
Actual On Time        {cm[0][0]:<17} {cm[0][1]}
Actual Delayed        {cm[1][0]:<17} {cm[1][1]}

🎯 MODEL COMPARISON SUMMARY
─────────────────────────────────────────────────────────────────────────────────
Traditional ML:  F1 = {ml_test_f1:.4f}, AUC = {ml_test_auc:.4f}
Deep Learning:   F1 = {dl_results_df.iloc[0]['Test F1']:.4f}, AUC = {dl_results_df.iloc[0]['Test AUC']:.4f}
Ensemble:        F1 = {max(ensemble_results, key=lambda x: x['F1 Score'])['F1 Score']:.4f}

Winner: {final_model_name} with F1 = {final_f1:.4f}

🔬 DIMENSIONALITY REDUCTION (PCA)
─────────────────────────────────────────────────────────────────────────────────
Original Dimensions:  {X_train_preprocessed.shape[1]}
Reduced Dimensions:   {X_train_pca.shape[1]} (95% variance retained)
PCA + RF F1:          {pca_f1:.4f}
PCA + DL F1:          {pca_dl_f1:.4f}

💼 BUSINESS VALUE
─────────────────────────────────────────────────────────────────────────────────
✓ Early Warning System: Identify high-risk shipments 3-5 days in advance
✓ Multi-Model Validation: Traditional ML + Deep Learning cross-verification
✓ Ensemble Robustness: Combined predictions for critical decisions
✓ Resource Optimization: Allocate customer service for predicted delays
✓ Route Planning: Dynamic routing based on delay probability
✓ Customer Satisfaction: Proactive communication for at-risk shipments
✓ Cost Reduction: Minimize penalty costs from delays

Estimated ROI: 15-25% reduction in delay-related costs

📁 DELIVERABLES & FILE FORMATS
─────────────────────────────────────────────────────────────────────────────────
✓ ship_delay_ml_model.joblib - Traditional ML pipeline (Joblib)
✓ ship_delay_simple_nn.h5 - Simple Neural Network (HDF5)
✓ ship_delay_deep_nn.h5 - Deep Neural Network (HDF5)
✓ ship_delay_resnet.h5 - Residual Network (HDF5)
✓ preprocessor.joblib - Data preprocessor (Joblib)
✓ model_metadata.json - Complete model metadata (JSON)
✓ feature_info.json - Feature information (JSON)
✓ ship_logistics_raw_data.csv - Training dataset
✓ model_comparison_results.csv - Traditional ML metrics
✓ dl_model_comparison_results.csv - Deep Learning metrics
✓ all_models_comparison.csv - Combined comparison
✓ ensemble_results.csv - Ensemble metrics
✓ feature_importance.csv - Feature importance scores

Visualizations:
✓ eda_visualization.png - Exploratory analysis
✓ dl_training_history.png - DL training curves
✓ ml_vs_dl_comparison.png - Performance comparison
✓ final_model_comprehensive_evaluation.png - Final model metrics
✓ pca_analysis.png - Dimensionality reduction

🚀 DEPLOYMENT OPTIONS
─────────────────────────────────────────────────────────────────────────────────
1. REST API Endpoint (Flask/FastAPI)
2. Batch Prediction Service
3. Real-time Streaming Predictions (Kafka/RabbitMQ)
4. Integration with Logistics Management Systems
5. Mobile Application Support (TensorFlow Lite conversion available)
6. Cloud Deployment (AWS SageMaker, Google AI Platform, Azure ML)

📚 TECHNICAL STACK
─────────────────────────────────────────────────────────────────────────────────
• Python 3.x
• Scikit-learn (Traditional ML)
• TensorFlow/Keras (Deep Learning)
• Pandas, NumPy (Data Processing)
• Matplotlib, Seaborn (Visualization)
• Joblib (Model Persistence - Fast & Reliable)
• JSON (Metadata Storage - Human Readable)
• HDF5 (Neural Network Storage - Industry Standard)

🎓 DEMONSTRATED SKILLS
─────────────────────────────────────────────────────────────────────────────────
✓ End-to-end ML pipeline development
✓ Traditional ML algorithms (6 models)
✓ Deep Learning architecture design (3 neural networks)
✓ Ensemble learning techniques
✓ Hyperparameter optimization (Grid Search)
✓ Cross-validation strategies
✓ Feature engineering & selection
✓ Dimensionality reduction (PCA)
✓ Model evaluation & comparison
✓ Production-ready model persistence
✓ Business insight generation
✓ MLOps best practices

⏱️  PERFORMANCE INSIGHTS
─────────────────────────────────────────────────────────────────────────────────
• Traditional ML trains in seconds, excellent for rapid iteration
• Deep Learning achieves {dl_results_df.iloc[0]['Test F1']:.4f} F1, capturing complex patterns
• Ensemble provides robust predictions with {max(ensemble_results, key=lambda x: x['F1 Score'])['F1 Score']:.4f} F1
• PCA reduces dimensions by {(1-X_train_pca.shape[1]/X_train_preprocessed.shape[1])*100:.1f}% while retaining 95% variance

📊 RECOMMENDATIONS FOR PRODUCTION
─────────────────────────────────────────────────────────────────────────────────
1. Use ensemble model for critical business decisions
2. Deploy Traditional ML for real-time predictions (faster inference)
3. Use Deep Learning for batch processing and complex scenarios
4. Implement A/B testing between models in production
5. Set up monitoring for model drift and retraining triggers
6. Establish prediction confidence thresholds for automatic vs manual review

═══════════════════════════════════════════════════════════════════════════════
                        PROJECT COMPLETED SUCCESSFULLY!
                https://github.com/ryantusi/Ship_Delay_AI_Predictor
═══════════════════════════════════════════════════════════════════════════════
"""

print(report)

# Save report
with open('comprehensive_project_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✓ Full report saved to 'comprehensive_project_report.txt'")

print("\n" + "=" * 80)
print("✅ ENHANCED ML + DEEP LEARNING PIPELINE COMPLETED!")
print("=" * 80)
print("\nThis portfolio project demonstrates:")
print("  ✓ Complete ETL process with real-world data patterns")
print("  ✓ Comprehensive EDA with business insights")
print("  ✓ Advanced feature engineering techniques")
print("  ✓ Traditional ML: 6 algorithms with hyperparameter tuning")
print("  ✓ Deep Learning: 3 neural network architectures")
print("  ✓ Ensemble methods combining ML and DL")
print("  ✓ Model comparison and selection framework")
print("  ✓ Dimensionality reduction (PCA) with ML and DL")
print("  ✓ Production-ready persistence (Joblib + H5 + JSON)")
print("  ✓ Comprehensive evaluation and business recommendations")
print("  ✓ Full model deployment verification")
print("\n" + "=" * 80)