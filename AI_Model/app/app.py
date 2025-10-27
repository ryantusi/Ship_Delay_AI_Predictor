# app.py - Optimized Ship Delay Prediction

from flask import Flask, render_template, request, jsonify
import pandas as pd
import utils 
import os
import shutil

app = Flask(__name__)

# Load models on first request
MODELS_LOADED = False

@app.before_request
def setup_models():
    """Load models, preprocessor, and metadata upon the first request."""
    global MODELS_LOADED
    if not MODELS_LOADED:
        utils.load_models_and_metadata()
        MODELS_LOADED = True

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    
    # Original 25 features (same as before)
    original_features = [
        'vessel_type', 'vessel_age_years', 'vessel_capacity_teu', 'origin_port',
        'destination_port', 'distance_nautical_miles', 'fuel_consumption_tons_day',
        'crew_size', 'cargo_weight_tons', 'cargo_utilization_pct', 'season',
        'weather_risk_score', 'sea_state', 'origin_port_congestion',
        'dest_port_congestion', 'customs_complexity_score', 'departure_hour',
        'departure_day', 'fuel_price_index', 'freight_rate_index',
        'days_since_maintenance', 'safety_inspection_score',
        'number_of_port_calls', 'has_hazardous_cargo', 'is_peak_season'
    ]

    # Default sample data (low-risk shipment)
    sample_data = {
        'vessel_type': 'Container Ship',
        'vessel_age_years': 6,
        'vessel_capacity_teu': 15000,
        'origin_port': 'Singapore',
        'destination_port': 'Sydney',
        'distance_nautical_miles': 3900,
        'fuel_consumption_tons_day': 55,
        'crew_size': 30,
        'cargo_weight_tons': 8200,
        'cargo_utilization_pct': 54.7,
        'season': 'Spring',
        'weather_risk_score': 2.3,
        'sea_state': 'Calm',
        'origin_port_congestion': 3.2,
        'dest_port_congestion': 2.8,
        'customs_complexity_score': 3.1,
        'departure_hour': 9,
        'departure_day': 'Wednesday',
        'fuel_price_index': 1.05,
        'freight_rate_index': 1.15,
        'days_since_maintenance': 52,
        'safety_inspection_score': 94,
        'number_of_port_calls': 2,
        'has_hazardous_cargo': 0,
        'is_peak_season': 0
    }

    if request.method == 'POST':
        # Collect input data from form
        input_data = {}
        for feature in original_features:
            value = request.form.get(feature)
            
            # Type conversion
            if feature in ['vessel_age_years', 'vessel_capacity_teu', 'crew_size', 
                           'departure_hour', 'days_since_maintenance', 'number_of_port_calls', 
                           'has_hazardous_cargo', 'is_peak_season']:
                input_data[feature] = int(value) if value else 0
                sample_data[feature] = int(value) if value else 0
            elif feature in ['distance_nautical_miles', 'fuel_consumption_tons_day', 'cargo_weight_tons', 
                             'cargo_utilization_pct', 'weather_risk_score', 'origin_port_congestion',
                             'dest_port_congestion', 'customs_complexity_score', 'fuel_price_index', 
                             'freight_rate_index', 'safety_inspection_score']:
                input_data[feature] = float(value) if value else 0.0
                sample_data[feature] = float(value) if value else 0.0
            else:
                input_data[feature] = value
                sample_data[feature] = value
                
        # Engineer features (adds 15+ new features)
        shipment_df = utils.engineer_features(input_data)
        
        # Predict using optimized model (XGBoost with threshold 0.316)
        prediction_result = utils.predict_with_optimized_model(
            utils.ml_pipeline, 
            shipment_df,
            threshold=utils.optimal_threshold
        )

    # Dropdown options for categorical features
    options = {
        'vessel_type': ['Container Ship', 'Bulk Carrier', 'Tanker', 'RoRo', 'General Cargo'],
        'origin_port': ['Shanghai', 'Singapore', 'Rotterdam', 'Los Angeles', 'Hamburg', 
                       'Hong Kong', 'Busan', 'Dubai', 'Mumbai', 'Tokyo'],
        'destination_port': ['New York', 'London', 'Sydney', 'Santos', 'Vancouver', 
                            'Cape Town', 'Melbourne', 'Barcelona', 'Houston', 'Auckland'],
        'season': ['Spring', 'Summer', 'Fall', 'Winter'],
        'sea_state': ['Calm', 'Moderate', 'Rough', 'Very Rough'],
        'departure_day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    }

    return render_template('index.html', 
                           original_features=original_features, 
                           sample_data=sample_data,
                           prediction_result=prediction_result,
                           options=options,
                           final_model=utils.final_model_name,
                           optimal_threshold=utils.optimal_threshold)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """API endpoint for predictions (optional - for programmatic access)"""
    try:
        input_data = request.get_json()
        shipment_df = utils.engineer_features(input_data)
        prediction = utils.predict_with_optimized_model(
            utils.ml_pipeline, 
            shipment_df,
            threshold=utils.optimal_threshold
        )
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Move model files to ./models/ directory
    os.makedirs('./models', exist_ok=True)
    
    model_files = [
        'optimized_ship_delay_model.joblib',
        'preprocessor_optimized.joblib',
        'optimized_simple_nn.keras',
        'optimized_batchnorm_nn.keras',
        'optimized_model_metadata.json'
    ]
    
    for f in model_files:
        try:
            if os.path.exists(f):
                shutil.move(f, f"./models/{f}")
        except FileNotFoundError:
            if not os.path.exists(f"./models/{f}"):
                print(f"Warning: {f} not found. Run optimized script first.")
        except shutil.Error:
            pass  # File already exists in destination
    
    print("\n" + "="*80)
    print("🚀 OPTIMIZED SHIP DELAY PREDICTOR - FLASK APP")
    print("="*80)
    print(f"Model: {utils.final_model_name if MODELS_LOADED else 'XGBoost (Recall-Optimized)'}")
    print(f"Optimal Threshold: 0.316 (vs 0.5 default)")
    print(f"Expected Recall: 91%+")
    print("="*80 + "\n")
    
    app.run(debug=True)
