from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json


        total_orders = float(data.get('total_orders', 0))
        total_spent = float(data.get('total_spent', 0))
        total_returns = float(data.get('total_returns', 0))
        return_ratio = float(data.get('return_ratio', 0))
        lifetime = float(data.get('customer_lifetime_days', 0))
        avg_value = float(data.get('avg_order_value', 0))
        freq = float(data.get('purchase_frequency', 0))


        if return_ratio >= 0.8 or total_returns >= total_orders:
            return jsonify({
                "fraud_prediction": 1,
                "risk_score": 95,
                "reason": "High return ratio detected (rule-based override)"
            })


        features = [[
            total_orders,
            total_spent,
            total_returns,
            return_ratio,
            lifetime,
            avg_value,
            freq
        ]]

        input_data = np.array(features)


        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        risk_score = round(probability * 100, 2)


        return jsonify({
            "fraud_prediction": int(prediction),
            "risk_score": risk_score,
            "reason": "ML Model Prediction"
        })

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)