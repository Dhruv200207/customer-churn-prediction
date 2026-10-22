# Customer Churn Prediction

An end-to-end machine learning application that predicts telecom customer churn using Python, Scikit-learn, and Streamlit. The application provides a churn prediction and an estimated probability to help demonstrate customer retention analytics.

## Live Demo

**Try the app:** https://customer-churn-prediction-cihtdcuuf6yrvgdedcwthd.streamlit.app/

## Features

- Predict whether a telecom customer is likely to churn or stay.
- Display estimated churn probability.
- Train and compare Logistic Regression and Random Forest models.
- Use a saved preprocessing and prediction pipeline.
- Provide an interactive web interface using Streamlit.

## Model Performance

The selected model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1 Score | 60.40% |
| ROC-AUC | 0.8419 |

*Note: Performance depends on the dataset, test split, and classification threshold. These metrics represent evaluation results, not guaranteed real-world performance.*

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Web Application:** Streamlit

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── requirements.txt
├── README.md
└── artifacts/
    └── churn_prediction_pipeline.joblib
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Dhruv200207/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal.

## Model Artifact

The trained machine learning pipeline is stored at:

`artifacts/churn_prediction_pipeline.joblib`

The pipeline includes the fitted preprocessing and prediction workflow used by the application.

## Disclaimer

This project is intended for educational and demonstration purposes. Predictions are estimates and should not be treated as guarantees or used as the sole basis for business decisions.
