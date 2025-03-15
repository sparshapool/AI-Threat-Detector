from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model
model = joblib.load("C:/Users/Sparsh/Documents/AI-Threat-Detector/trained_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = [
            data["Protocol_TCP"],
            data["Protocol_TLS"],
            data["Total_Source_Freq"],
            data["Total_Destination_Freq"],
        ]
        prediction = model.predict([features])[0]
        return jsonify({"prediction": "Threat Detected" if prediction == 1 else "Normal Traffic"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
