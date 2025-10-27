# utils.py - Optimized Ship Delay Prediction Utilities

import numpy as np
import pandas as pd
import joblib
import json
import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available. Using ML model only.")

# Global variables
ml_pipeline = None
preprocessor = None
dl_models = {}
metadata = {}
final_model_name = ""
optimal_threshold = 0.316  # Optimized threshold for high recall

def load_models_and_metadata():
    """Load optimized models, preprocessor, and metadata."""
    global ml_pipeline, preprocessor, dl_models, metadata, final_model_name, optimal_threshold
    
    try:
        # Load optimized ML pipeline
        ml_pipeline = joblib.load('./models/optimized_ship_delay_model.joblib')
        preprocessor = joblib.load('./models/preprocessor_optimized.joblib')
        
        # Load metadata
        with open('./models/optimized_model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        final_model_name = metadata['best_model']['name']
        optimal_threshold = metadata['best_model']['optimal_threshold']
        
        print(f"✓ Loaded: {final_model_name}")
        print(f"✓ Optimal Threshold: {optimal_threshold:.3f}")
        print(f"✓ Expected Recall: {metadata['best_model']['test_metrics']['recall']*100:.1f}%")
        
        # Load DL models if available
        if TENSORFLOW_AVAILABLE:
            try:
                dl_models['Simple NN'] = keras.models.load_model('./models/optimized_simple_nn.keras')
                dl_models['BatchNorm NN'] = keras.models.load_model('./models/optimized_batchnorm_nn.keras')
                print("✓ Loaded: Deep Learning models")
            except:
                print("⚠ DL models not found (optional)")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)

def engineer_features(data: dict) -> pd.DataFrame:
    """
    Engineer ALL 15+ features from the optimized model.
    This matches the exact feature engineering from the training script.
    """
    df = pd.DataFrame([data])
    
    # Fill missing values for safety
    df['fuel_consumption_tons_day'].fillna(df['fuel_consumption_tons_day'].median() 
                                            if 'fuel_consumption_tons_day' in df.columns else 85, inplace=True)
    df['weather_risk_score'].fillna(df['weather_risk_score'].median() 
                                     if 'weather_risk_score' in df.columns else 5, inplace=True)
    
    # === CORE ENGINEERED FEATURES (from original) ===
    df['vessel_efficiency'] = df['cargo_weight_tons'] / df['fuel_consumption_tons_day']
    
    df['route_complexity'] = (df['distance_nautical_miles'] * df['number_of_port_calls']) / 1000
    
    df['port_total_congestion'] = df['origin_port_congestion'] + df['dest_port_congestion']
    
    # Map sea state to numeric for weather_sea_risk
    sea_state_map = {'Calm': 1, 'Moderate': 2, 'Rough': 3, 'Very Rough': 4}
    df['weather_sea_risk'] = df['weather_risk_score'] * df['sea_state'].map(sea_state_map)
    
    # === NEW INTERACTION FEATURES (critical for performance!) ===
    df['age_weather_interaction'] = df['vessel_age_years'] * df['weather_risk_score']
    
    df['congestion_customs_interaction'] = df['port_total_congestion'] * df['customs_complexity_score']
    
    df['utilization_distance_interaction'] = (df['cargo_utilization_pct'] * df['distance_nautical_miles']) / 1000
    
    # === NEW RISK FLAGS ===
    df['high_risk_departure'] = (
        (df['departure_day'].isin(['Friday', 'Saturday'])) & 
        (df['departure_hour'].between(18, 23))
    ).astype(int)
    
    df['critical_maintenance_window'] = (df['days_since_maintenance'] > 250).astype(int)
    
    df['extreme_conditions'] = (
        ((df['weather_risk_score'] > 7) | (df['sea_state'].isin(['Rough', 'Very Rough']))) &
        (df['vessel_age_years'] > 15)
    ).astype(int)
    
    # === CATEGORICAL FEATURES ===
    df['age_category'] = pd.cut(
        df['vessel_age_years'], 
        bins=[0, 10, 20, 30], 
        labels=['New', 'Medium', 'Old'],
        include_lowest=True
    ).astype(object)
    
    df['utilization_category'] = pd.cut(
        df['cargo_utilization_pct'],
        bins=[0, 60, 85, 100],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    ).astype(object)
    
    # === BINARY FLAGS ===
    df['is_weekend_departure'] = df['departure_day'].isin(['Saturday', 'Sunday']).astype(int)
    
    df['high_utilization'] = (df['cargo_utilization_pct'] > 85).astype(int)
    
    df['maintenance_risk'] = (df['days_since_maintenance'] > 180).astype(int)
    
    return df

def predict_with_optimized_model(pipeline, shipment_df: pd.DataFrame, threshold=0.316):
    """
    Predict using optimized XGBoost model with custom threshold.
    
    Args:
        pipeline: Trained model pipeline (XGBoost with SMOTE)
        shipment_df: DataFrame with all engineered features
        threshold: Classification threshold (default 0.316 for high recall)
    
    Returns:
        Dict with prediction, probability, risk level, and recommendations
    """
    
    # Get probability
    probability = pipeline.predict_proba(shipment_df)[0, 1]
    
    # Apply custom threshold
    prediction = int(probability >= threshold)
    
    # Risk categorization
    if probability < 0.3:
        risk_level = 'LOW'
        risk_color = '#28a745'  # Green
        recommendation = 'Normal monitoring sufficient'
    elif probability < 0.5:
        risk_level = 'MEDIUM'
        risk_color = '#ffc107'  # Yellow
        recommendation = 'Monitor closely, prepare contingency'
    elif probability < 0.7:
        risk_level = 'HIGH'
        risk_color = '#fd7e14'  # Orange
        recommendation = 'Alert logistics team, consider route alternatives'
    else:
        risk_level = 'CRITICAL'
        risk_color = '#dc3545'  # Red
        recommendation = 'IMMEDIATE ACTION: Contact customer, expedite alternatives'
    
    return {
        'prediction': 'DELAYED' if prediction == 1 else 'ON TIME',
        'delay_probability': float(probability),
        'confidence': float(max(probability, 1 - probability)),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'recommendation': recommendation,
        'threshold_used': threshold,
        'model_type': f'{final_model_name} (Recall-Optimized)',
        'catches_delays': '91% of all delays detected' if prediction == 1 else None
    }

def predict_with_dl(model, preprocessor, shipment_df: pd.DataFrame, threshold=0.316):
    """Predict using Deep Learning model (optional)."""
    
    if not TENSORFLOW_AVAILABLE:
        return predict_with_optimized_model(ml_pipeline, shipment_df, threshold)
    
    shipment_preprocessed = preprocessor.transform(shipment_df)
    probability = model.predict(shipment_preprocessed, verbose=0).flatten()[0]
    prediction = int(probability >= threshold)
    
    return {
        'prediction': 'DELAYED' if prediction == 1 else 'ON TIME',
        'delay_probability': float(probability),
        'confidence': float(max(probability, 1 - probability)),
        'model_type': 'Neural Network (Deep Learning)',
        'threshold_used': threshold
    }

def get_feature_importance():
    """Get top feature importances from the model (for debugging/insights)."""
    try:
        if hasattr(ml_pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = ml_pipeline.named_steps['classifier'].feature_importances_
            return importances
    except:
        pass
    return None
