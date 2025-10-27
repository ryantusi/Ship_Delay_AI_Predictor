# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import utils 
import os
import shutil

app = Flask(__name__)

# --- Replaced deprecated @app.before_first_request (already fixed) ---
MODELS_LOADED = False

@app.before_request
def setup_models():
    """Load models, preprocessor, and metadata upon the first request."""
    global MODELS_LOADED
    if not MODELS_LOADED:
        utils.load_models_and_metadata()
        MODELS_LOADED = True
# ---------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    
    # Define the list of original features
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

    # Sample data to pre-populate the form (default values)
    sample_data = {
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
        'is_peak_season': 0
    }

    if request.method == 'POST':
        # 1. Collect Input Data & UPDATE sample_data (THE FIX IS HERE)
        input_data = {}
        for feature in original_features:
            value = request.form.get(feature)
            
            # Type Conversion Logic (re-used from original code)
            if feature in ['vessel_age_years', 'vessel_capacity_teu', 'crew_size', 
                           'departure_hour', 'days_since_maintenance', 'number_of_port_calls', 
                           'has_hazardous_cargo', 'is_peak_season']:
                input_data[feature] = int(value) if value else 0
                sample_data[feature] = int(value) if value else 0 # FIX: Update sample_data
            elif feature in ['distance_nautical_miles', 'fuel_consumption_tons_day', 'cargo_weight_tons', 
                             'cargo_utilization_pct', 'weather_risk_score', 'origin_port_congestion',
                             'dest_port_congestion', 'customs_complexity_score', 'fuel_price_index', 
                             'freight_rate_index', 'safety_inspection_score']:
                input_data[feature] = float(value) if value else 0.0
                sample_data[feature] = float(value) if value else 0.0 # FIX: Update sample_data
            else:
                input_data[feature] = value
                sample_data[feature] = value # FIX: Update sample_data
                
        # 2. Engineer Features
        shipment_df = utils.engineer_features(input_data)
        
        # 3. Predict using the Final Selected Model
        final_model_name_lower = utils.final_model_name.lower()
        
        if final_model_name_lower.startswith("ensemble"):
            prediction_result = utils.predict_ensemble(utils.ml_pipeline, utils.dl_models, utils.preprocessor, shipment_df)
        elif 'resnet' in final_model_name_lower:
            # Explicitly use the ResNet model (Deep Learning)
            prediction_result = utils.predict_with_dl(
                utils.dl_models['ResNet'], 
                utils.preprocessor, 
                shipment_df,
                model_name="ResNet (Deep Learning)" # Pass the full model name
            )
        else:
            # Fallback to the Traditional ML model (e.g., if RF was the winner)
            prediction_result = utils.predict_with_ml(utils.ml_pipeline, shipment_df)

    # Re-map categorical options for template rendering (unchanged)
    options = {
        'vessel_type': ['Container Ship', 'Bulk Carrier', 'Tanker', 'RoRo', 'General Cargo'],
        'origin_port': ['Shanghai', 'Singapore', 'Rotterdam', 'Los Angeles', 'Hamburg', 'Hong Kong', 'Busan', 'Dubai', 'Mumbai', 'Tokyo'],
        'destination_port': ['New York', 'London', 'Sydney', 'Santos', 'Vancouver', 'Cape Town', 'Melbourne', 'Barcelona', 'Houston', 'Auckland'],
        'season': ['Spring', 'Summer', 'Fall', 'Winter'],
        'sea_state': ['Calm', 'Moderate', 'Rough', 'Very Rough'],
        'departure_day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    }

    return render_template('index.html', 
                           original_features=original_features, 
                           sample_data=sample_data, # This now contains the submitted values
                           prediction_result=prediction_result,
                           options=options,
                           final_model=utils.final_model_name)

if __name__ == '__main__':
    # ... (file moving logic remains the same) ...
    os.makedirs('./models', exist_ok=True) 
    import shutil
    model_files = ['ship_delay_ml_model.joblib', 'preprocessor.joblib', 'ship_delay_simple_nn.h5', 
                   'ship_delay_deep_nn.h5', 'ship_delay_resnet.h5', 'model_metadata.json', 'feature_info.json']
    
    for f in model_files:
        try:
            shutil.move(f, f"./models/{f}")
        except FileNotFoundError:
            if not os.path.exists(f"./models/{f}"):
                print(f"Warning: Model file {f} not found. Did you run the full script first?")
        except shutil.Error:
            pass 

    app.run(debug=True)