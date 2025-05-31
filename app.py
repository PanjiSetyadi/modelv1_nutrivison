from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load model dan preprocessing
model = load_model("model_obesitas.keras")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# model 2
data = pd.read_csv("nutrition.csv")
data.dropna(inplace=True)
data["name_clean"] = data["name"].str.lower().str.strip()

model = load_model("model_rekomendasi.keras")
scaler = joblib.load("scaler_nutrisi.pkl")

# Normalisasi fitur numerik
features = ['calories', 'proteins', 'fat', 'carbohydrate']
X = scaler.transform(data[features])

app = Flask(__name__)

CORS(app)  # 👈 Ini mengizinkan semua origin mengakses API kamu



@app.route("/deteksi")
def deteksi_bb():
    return render_template("deteksi_bb.html")

# Route untuk halaman rekomendasi makanan
@app.route("/rekomendasi")
def rekomendasi_makanan():
    return render_template("recommend_makanan.html")


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
    
    
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        req_data = request.get_json()
        food_name = req_data.get("food_name", "").lower().strip()

        # Cek apakah makanan tersedia
        if food_name not in data["name_clean"].values:
            # Coba cari suggestion jika user typo atau beda format
            suggestions = data[data["name_clean"].str.contains(food_name, na=False)]
            if not suggestions.empty:
                return jsonify({
                    "status": "error",
                    "message": f"Makanan tidak ditemukan. Apakah maksud Anda salah satu dari berikut? {suggestions['name'].tolist()}"
                }), 404
            else:
                return jsonify({
                    "status": "error",
                    "message": "Makanan tidak ditemukan. Coba cek penulisan atau gunakan nama lain."
                }), 404

        # Ambil index makanan input
        food_index = data[data["name_clean"] == food_name].index[0]
        input_nutrisi = X[food_index].reshape(1, -1)

        # Embedding dengan model
        input_embed = model.predict(input_nutrisi, verbose=0)
        all_embeddings = model.predict(X, verbose=0)

        # Hitung kemiripan
        similarities = cosine_similarity(input_embed, all_embeddings)[0]
        similar_index = similarities.argsort()[::-1][1]  # Hanya ambil yang paling mirip (index pertama setelah input)

        # Data makanan asli
        base_food = data.iloc[food_index]
        rekomendasi = []
        food = data.iloc[similar_index]
        rekomendasi.append({
            "name": food["name"],
            "calories": food["calories"],
            "proteins": food["proteins"],
            "fat": food["fat"],
            "carbohydrate": food["carbohydrate"],
            "image": food.get("image", "")
        })

        return jsonify({
            "status": "success",
            "input_food": {
                "name": base_food["name"],
                "calories": base_food["calories"],
                "proteins": base_food["proteins"],
                "fat": base_food["fat"],
                "carbohydrate": base_food["carbohydrate"],
                "image": base_food.get("image", "")
            },
            "recommendations": rekomendasi
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
    

if __name__ == "__main__":
    app.run(debug=True)
