from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model dan preprocessing
model = load_model("model_obesitas.keras")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__)

CORS(app)  # 👈 Ini mengizinkan semua origin mengakses API kamu

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data JSON
        data = request.get_json()
        age = data["age"]
        gender = data["gender"].strip().lower()  # 'male' atau 'female'
        height_cm = data["height_cm"]
        weight_kg = data["weight_kg"]
        activity = data["activity"]

        # Hitung BMI
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)

        # Encode gender
        gender_num = 1 if gender == "male" else 0

        # Siapkan input dan scaling
        input_array = np.array([[age, gender_num, height_cm, weight_kg, bmi, activity]])
        input_scaled = scaler.transform(input_array)

        # Prediksi
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        return jsonify({
            "status": "success",
            "predicted_class": int(predicted_class),
            "predicted_label": predicted_label
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
