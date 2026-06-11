from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)


# ---------------- HOME PAGE (HTML FORM) ----------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------- API ENDPOINT (JSON INPUT) ----------------
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    features = np.array([[
        data['Gender'],
        data['Married'],
        data['Dependents'],
        data['Education'],
        data['Self_Employed'],
        data['ApplicantIncome'],
        data['CoapplicantIncome'],
        data['LoanAmount'],
        data['Loan_Amount_Term'],
        data['Credit_History'],
        data['Property_Area']
    ]])

    prediction = model.predict(features)[0]

    result = "Approved" if prediction == 1 else "Rejected"

    return jsonify({
        'loan_status': result,
        'prediction': int(prediction)
    })


# ---------------- HTML FORM SUBMISSION ----------------
@app.route('/predict-form', methods=['POST'])
def predict_form():

    features = np.array([[
        float(request.form['Gender']),
        float(request.form['Married']),
        float(request.form['Dependents']),
        float(request.form['Education']),
        float(request.form['Self_Employed']),
        float(request.form['ApplicantIncome']),
        float(request.form['CoapplicantIncome']),
        float(request.form['LoanAmount']),
        float(request.form['Loan_Amount_Term']),
        float(request.form['Credit_History']),
        float(request.form['Property_Area'])
    ]])

    prediction = model.predict(features)[0]

    result = "Approved" if prediction == 1 else "Rejected"

    return f"<h2>Loan Status: {result}</h2>"


# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)