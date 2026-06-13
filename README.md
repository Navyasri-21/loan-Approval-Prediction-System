# 🏦 Loan Approval Prediction — Machine Learning Project

A Machine Learning project that predicts whether a loan application will be **approved or rejected**, helping financial institutions make smarter, data-driven lending decisions.

---

## 📌 Project Overview

Loan approval is an important task for financial institutions. Banks must evaluate whether an applicant is likely to repay a loan based on several factors such as income, education, employment status, credit history, and property details.

This project builds a Machine Learning model to predict loan approval status using applicant information. The model learns patterns from historical data and predicts whether a loan should be approved or rejected.

The project demonstrates the complete Machine Learning pipeline, including:
- Data preprocessing
- Data visualization
- Feature encoding
- Model training
- Model evaluation

---

## 📂 Dataset Description

The dataset contains loan application details of applicants.

| Feature | Description |
|---|---|
| Gender | Male / Female |
| Married | Applicant marital status |
| Dependents | Number of dependents |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Applicant's monthly income |
| CoapplicantIncome | Co-applicant's monthly income |
| LoanAmount | Requested loan amount |
| Loan_Amount_Term | Loan repayment term |
| Credit_History | Credit history (1 = good, 0 = bad) |
| Property_Area | Urban / Semi-Urban / Rural |

**Target Variable:**

| Value | Meaning |
|---|---|
| Y (1) | Loan Approved |
| N (0) | Loan Rejected |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔄 Project Workflow

### 1. Data Collection
The dataset is loaded using Pandas.
```python
data = pd.read_csv("train.csv")
```

### 2. Data Preprocessing
Steps performed:
- Handling missing values
- Dropping null rows
- Encoding categorical variables
- Converting text values to numeric values

```python
data.replace({'Loan_Status': {'N': 0, 'Y': 1}}, inplace=True)
```

### 3. Data Visualization
Visualization helps understand patterns in the data.

```python
sns.countplot(x='Education', hue='Loan_Status', data=data)
sns.countplot(x='Married',   hue='Loan_Status', data=data)
```

**Sample Visualizations:**

| Education vs Loan Approval | Marital Status vs Loan Approval |
|---|---|
| ![Education](visualizations/education_vs_loan.png) | ![Marital](visualizations/marital_status_vs_loan.png) |

### 4. Feature and Target Separation
```python
X = data.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = data['Loan_Status']
```

### 5. Train Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size    = 0.1,
    stratify     = Y,
    random_state = 2
)
```

### 6. Model Training
The model used in this project is **Logistic Regression**.
```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, Y_train)
```

### 7. Model Evaluation
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_test_prediction = classifier.predict(X_test)

print(f"Accuracy  : {accuracy_score(Y_test, X_test_prediction):.4f}")
print(f"Precision : {precision_score(Y_test, X_test_prediction):.4f}")
print(f"Recall    : {recall_score(Y_test, X_test_prediction):.4f}")
print(f"F1 Score  : {f1_score(Y_test, X_test_prediction):.4f}")
```

---

## 📊 Model Performance

| Metric | Train Data | Test Data |
|---|---|---|
| Accuracy | 0.8333 | 0.8333 |
| Precision | — | 0.8378 |
| Recall | — | 0.9394 |
| F1 Score | — | 0.8857 |

### What these scores mean:
- **Accuracy (83%)** — Model correctly predicts 83 out of every 100 applications
- **High Recall (94%)** — Catches 94% of all actually approved loans — very few missed
- **Precision (84%)** — When model predicts approval, it is correct 84% of the time
- **F1 Score (0.88)** — Strong overall balance — model is well trained and reliable

---

## 📁 Project Structure

```
Loan-Prediction-ML/
│
├── loan_prediction.ipynb         # Main notebook
├── train.csv                     # Training dataset
├── README.md                     # Project documentation
│
└── visualizations/
    ├── education_vs_loan.png
    └── marital_status_vs_loan.png
```

---

## 🔍 Key Insights

From the data analysis:
- Applicants with **good credit history** have significantly higher loan approval chances
- **Graduates** tend to receive approvals more frequently than non-graduates
- **Semi-urban areas** show higher approval rates compared to rural areas
- **Income and credit history** are the strongest indicators for loan approval
- These insights help understand the key factors influencing bank lending decisions

---

## 👤 Author

**Navya Sri**
B.Tech — Artificial Intelligence and Data Science

Interested in:
- Data Science
- Machine Learning
- Data Analysis

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
