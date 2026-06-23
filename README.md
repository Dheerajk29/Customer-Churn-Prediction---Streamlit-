# K Dheeraj AI Prediction System

> Predict customer churn quickly and accurately using Machine Learning.

---

## Overview

This project uses a **Random Forest** model to predict whether a telecom customer is likely to cancel their subscription (churn). By entering customer details through a clean web interface, you get an instant churn probability, risk assessment, and personalized retention recommendations.

### How It Works

```
Customer Details → Preprocessing → Model Prediction → Result Display
```

1. Enter customer information (tenure, contract type, charges, services, etc.)
2. The app preprocesses the data (encoding + scaling) exactly like the training pipeline
3. A trained Random Forest model predicts the churn probability
4. Results are displayed with visualizations and recommendations

---

## Features

- **Clean, Minimal UI** — White theme with professional styling
- **Real-time Prediction** — Instant churn probability score
- **Risk Classification** — High / Medium / Low risk labels
- **Risk Gauge** — Visual indicator of churn probability
- **Feature Importance** — Shows which factors drive the prediction
- **Retention Recommendations** — Auto-generated business actions
- **Error Handling** — Graceful fallbacks for missing files or invalid inputs
- **Responsive Design** — Works on desktop and mobile

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit (HTML + CSS) |
| Backend | Python |
| ML Model | Random Forest Classifier |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Visualization | Plotly |
| Serialization | Pickle |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest |
| Dataset | Telco Customer Churn (Kaggle) |
| Training Records | 7,043 |
| Features | 19 |
| AUC-ROC | **0.84** |
| Class Imbalance | SMOTE |
| Threshold Tuning | Optimized via Precision-Recall curve |

### Top 5 Important Features

| Feature | Importance |
|---------|-----------|
| Contract Type | 21.1% |
| Online Security | 13.8% |
| Tenure | 10.9% |
| Tech Support | 10.5% |
| Monthly Charges | 9.2% |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Customer-Churn-Prediction-Streamlit

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Download Telco-Customer-Churn.csv from Kaggle and place it in this folder

# 5. Train the model (optional - pre-trained files included)
python churn_model.py

# 6. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Folder Structure

```
Customer-Churn-Prediction-Streamlit/
├── app.py                  # Main application (Streamlit UI)
├── churn_model.py          # Model training script
├── model.pkl               # Trained Random Forest model
├── scaler.pkl              # StandardScaler for feature scaling
├── feature_columns.pkl     # Feature names in training order
├── threshold.pkl           # Optimal decision threshold
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Usage

1. Open the app in your browser
2. Fill in customer details in the input form:
   - Account information (tenure, charges, senior status)
   - Contract and billing details
   - Service subscriptions
   - Personal information
3. Click **Predict** to get the churn probability
4. View the result, risk gauge, and recommendations
5. Click **Reset** to clear all inputs and start over

---

## Screenshots

> *(Add screenshots here — drag and drop images into the repo)*

---

## Future Improvements

- **Cloud Deployment** — Deploy to Streamlit Cloud, Hugging Face Spaces, or AWS
- **Dashboard Analytics** — Add historical predictions and trends
- **Multiple ML Models** — Compare Random Forest, XGBoost, and Neural Networks
- **User Authentication** — Multi-user support with login
- **Batch Prediction** — Upload CSV files for bulk predictions
- **API Endpoint** — REST API for programmatic access

---

## Author

**K Dheeraj**  
AI & Machine Learning Developer

---

## License

This project is open source and available for educational and commercial use.
