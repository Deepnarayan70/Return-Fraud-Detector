from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Return Fraud Detector Backend Running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Input features
        features = [
            data['total_orders'],
            data['total_spent'],
            data['total_returns'],
            data['return_ratio'],
            data['customer_lifetime_days'],
            data['avg_order_value'],
            data['purchase_frequency']
        ]

        input_data = np.array([features])

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        risk_score = round(probability * 100, 2)

        return jsonify({
            "fraud_prediction": int(prediction),
            "risk_score": risk_score
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)