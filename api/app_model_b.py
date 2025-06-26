from flask import Flask, request, jsonify
import pickle
import numpy as np
import datetime
import os

app = Flask(__name__)
model = pickle.load(open("model_b.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    feedback = request.json.get('feedback', None)
    prediction = model.predict([np.array(data)])
    
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "model": "ModelB",
        "features": data,
        "prediction": int(prediction[0]),
        "feedback": feedback
    }

    features_str = ";".join(map(str, data))
    with open("/logs/predictions.log", "a") as f:
        f.write(f"{datetime.datetime.now()},ModelB,{features_str},{int(prediction[0])},{feedback}\n")
    
    return jsonify({'model': 'B', 'prediction': int(prediction[0]), 'logged': True})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)