from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load Model and Metadata ---
#  IMPORTANT: Changed to relative paths for deployment compatibility
MODEL_PATH = 'D:/HeartDiseasePrediction/venv/models/heart_disease_model.pkl'
METADATA_PATH = 'D:/HeartDiseasePrediction/venv/models/feature_metadata.pkl'

full_pipeline = None
TRAINING_COLUMNS = []
MODEL_LOAD_ERROR = None

try:
    # Load the complete pipeline (preprocessor + classifier)
    full_pipeline = joblib.load(MODEL_PATH)
    # Load the feature order/names used during training
    feature_metadata = joblib.load(METADATA_PATH)
    TRAINING_COLUMNS = feature_metadata['all_features']
    print("Model loaded successfully.")
except Exception as e:
    # Store the error to display it in the log if model fails to load
    MODEL_LOAD_ERROR = f"Model load failed. Please check paths and files: {e}"
    print(MODEL_LOAD_ERROR)


# --- Home Route (Handles GET and POST for the UI) ---
@app.route('/', methods=['GET', 'POST'])
def home_ui():
    if MODEL_LOAD_ERROR:
        # If model loading failed, display the error on the page
        return render_template('index.html', error_message=MODEL_LOAD_ERROR)
    
    result = None
    
    if request.method == 'POST':
        try:
            # 1. Get data from the form
            form_data = request.form.to_dict()
            
            # Convert string numerical inputs to float/int
            # NOTE: We assume all inputs are present based on your form design.
            
            # Map the form data to the correct input structure for DataFrame
            input_data = {}
            for col in TRAINING_COLUMNS:
                # Flask form data is always a string; use the correct key from the form
                # We trust the user to provide all necessary keys
                input_data[col] = [form_data.get(col)]

            # 2. Convert to DataFrame (ensures column order matches training data)
            input_df = pd.DataFrame(input_data, columns=TRAINING_COLUMNS)
            
            # 3. Make Prediction
            prediction = full_pipeline.predict(input_df)[0]
            probability = full_pipeline.predict_proba(input_df)[0].tolist()
            
            result = {
                'prediction_value': int(prediction),
                'result_text': 'Presence of Heart Disease' if prediction == 1 else 'Absence of Heart Disease',
                'probability': probability
            }

        except Exception as e:
            result = {'error': f'Prediction failed: Invalid input or data type mismatch. Details: {e}'}

    # Render the HTML form, passing the result (or None) to the template
    return render_template('index.html', prediction_result=result)


# --- REST API Endpoint (Keeping the old one for compatibility) ---
@app.route('/predict_api', methods=['POST'])
def predict_api():
    if full_pipeline is None:
        return jsonify({'error': 'Model not available.'}), 500

    try:
        # This remains the same as your original API logic
        json_data = request.get_json(force=True)
        input_df = pd.DataFrame(json_data, index=[0])[TRAINING_COLUMNS]
        
        prediction = full_pipeline.predict(input_df)[0]
        probability = full_pipeline.predict_proba(input_df)[0].tolist()
        
        result_text = 'Presence of Heart Disease' if prediction == 1 else 'Absence of Heart Disease'

        return jsonify({
            'prediction_value': int(prediction),
            'result': result_text,
            'probability': probability
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 400


if __name__ == '__main__':
    # Local run command for testing
    # Note: Flask will look for templates/index.html automatically
    app.run(host='0.0.0.0', port=5000, debug=True)