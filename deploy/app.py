from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Load model & preprocessors using correct paths
model = joblib.load(r'C:\Users\lenovo\Desktop\final ml\models\rfmlmodel (1).pkl')
scaler = joblib.load(r'C:\Users\lenovo\Desktop\final ml\models\scaler (1).pkl')
oe = joblib.load(r'C:\Users\lenovo\Desktop\final ml\models\ordinal_encoder (1).pkl')
trained_features = joblib.load(r'C:\Users\lenovo\Desktop\final ml\models\feature_names (1).pkl')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        print("[DEBUG] Input received:", input_data)

        df = pd.DataFrame([input_data])

        # Encode categorical features
        cat_cols = ['Appliance Type', 'Season']
        df[cat_cols] = oe.transform(df[cat_cols])

        # Scale numeric features
        df[['Outdoor Temperature (°C)']] = scaler.transform(df[['Outdoor Temperature (°C)']])

        # Align with model input features
        for col in trained_features:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default 0
        df = df[trained_features]

        # Make prediction
        prediction = model.predict(df)[0]
        return jsonify({'predicted_energy_kwh': round(prediction, 3)})

    except Exception as e:
        print("[ERROR] Prediction failed:", e)
        return jsonify({'error': str(e)}), 500

def run_app(model=True, encoders=True, scaler=True):
    app.run(debug=True)


if __name__ == '__main__':
    run_app()
