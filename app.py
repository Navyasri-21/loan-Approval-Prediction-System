from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Loan Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['Gender'], data['Married'], data['Dependents'],
        data['Education'], data['Self_Employed'], data['ApplicantIncome'],
        data['CoapplicantIncome'], data['LoanAmount'],
        data['Loan_Amount_Term'], data['Credit_History'], data['Property_Area']
    ]])
    prediction = model.predict(features)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    return jsonify({'loan_status': result, 'prediction': int(prediction)})

if __name__ == '__main__':
    app.run()
