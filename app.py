"""
K Dheeraj AI Prediction System
================================
Customer Churn Prediction using Machine Learning.
Built by: K Dheeraj
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS - White / Minimal Theme
# ============================================================
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Reset Streamlit defaults */
    #root, .stApp {
        background-color: #f8fafc !important;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: #1e293b !important;
    }

    .stApp {
        background-color: #f8fafc !important;
    }

    /* Remove default padding/margin */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 900px !important;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .app-header .subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #64748b !important;
        margin-bottom: 0.3rem;
    }
    .app-header .description {
        font-size: 0.9rem;
        color: #94a3b8 !important;
        margin-bottom: 0;
    }

    /* Result card */
    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-card h2 {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b !important;
        margin-bottom: 1rem;
    }
    .result-percentage {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .result-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .result-detail {
        font-size: 0.85rem;
        color: #64748b !important;
        margin-top: 0.8rem;
    }
    .result-detail span {
        font-weight: 600;
        color: #1e293b !important;
    }

    /* Risk colors */
    .risk-high { color: #dc2626 !important; }
    .risk-medium { color: #d97706 !important; }
    .risk-low { color: #16a34a !important; }
    .bg-risk-high { background-color: #fef2f2; border-color: #fecaca; }
    .bg-risk-medium { background-color: #fffbeb; border-color: #fde68a; }
    .bg-risk-low { background-color: #f0fdf4; border-color: #bbf7d0; }

    /* Input labels - more visible */
    label, .stSlider label, .stSelectbox label, .stNumberInput label, .stRadio label, .stCheckbox label {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }

    /* Input values - bigger */
    .stSlider div[data-baseweb="slider"] div {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    .stNumberInput input {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    .stNumberInput input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        border: 2px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stRadio label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    .stCheckbox label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    /* Input section sub-headers */
    .stMarkdown strong {
        font-size: 1rem !important;
        color: #0f172a !important;
        display: block;
        margin-bottom: 0.5rem;
        padding: 0.3rem 0.5rem;
        background: #f1f5f9;
        border-radius: 6px;
    }

    /* Button styling - bigger and bolder */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.15s ease !important;
        letter-spacing: 0.3px !important;
        text-transform: uppercase !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
        box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important;
        transform: translateY(-1px) !important;
    }
    /* Reset button */
    div.stButton > button:last-child {
        background-color: #ffffff !important;
        color: #475569 !important;
        border: 2px solid #cbd5e1 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
    }
    div.stButton > button:last-child:hover {
        border-color: #94a3b8 !important;
        background-color: #f8fafc !important;
        color: #1e293b !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b !important;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Recommendation card */
    .rec-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
        min-height: 100px;
    }
    .rec-card .rec-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1e293b !important;
        margin-bottom: 0.4rem;
    }
    .rec-card .rec-desc {
        font-size: 0.8rem;
        color: #64748b !important;
        line-height: 1.5;
    }

    /* Summary bar */
    .summary-bar {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        font-size: 0.8rem;
        color: #64748b;
        text-align: center;
    }
    .summary-bar b {
        color: #2563eb !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>AI Prediction System</h1>
    <div class="description">Predict outcomes quickly and accurately using Machine Learning.</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, features, and threshold."""
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("feature_columns.pkl", "rb"))
    threshold = pickle.load(open("threshold.pkl", "rb"))
    return model, scaler, features, threshold

try:
    model, scaler, FEATURES, THRESHOLD = load_model_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error("⚠️ Model files not found. Please run <code>python churn_model.py</code> first.", unsafe_allow_html=True)
    st.stop()

# ============================================================
# HELPER: Classify risk level from probability
# ============================================================
def classify_risk(probability):
    """Convert a churn probability into a risk category."""
    if probability >= 0.65:
        return "HIGH RISK", "risk-high", "bg-risk-high", "🔴"
    elif probability >= 0.35:
        return "MEDIUM RISK", "risk-medium", "bg-risk-medium", "🟡"
    else:
        return "LOW RISK", "risk-low", "bg-risk-low", "🟢"

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown("<h3 style='font-size:1.3rem; font-weight:700; color:#0f172a; border-bottom:3px solid #2563eb; padding-bottom:0.5rem;'>📋 Enter Customer Details</h3>", unsafe_allow_html=True)

# --- Layout: Three columns for input fields ---
col1, col2, col3 = st.columns(3)

# --- Column 1: Account Info ---
with col1:
    st.markdown("**Account Information**")
    tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
    monthly_charges = st.number_input(
        "Monthly Charges ($)", 0.0, 150.0, 65.0, 0.5,
        key="monthly_charges"
    )
    total_charges = st.number_input(
        "Total Charges ($)", 0.0, 9000.0,
        value=round(tenure * monthly_charges, 2),
        step=10.0, key="total_charges"
    )

# --- Column 2: Contract & Billing ---
with col2:
    st.markdown("**Contract & Billing**")
    contract_options = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    contract = contract_options[
        st.selectbox("Contract Type", list(contract_options.keys()), key="contract")
    ]

    payment_options = {
        "Electronic Check": 0, "Mailed Check": 1,
        "Bank Transfer (Auto)": 2, "Credit Card (Auto)": 3
    }
    payment_method = payment_options[
        st.selectbox("Payment Method", list(payment_options.keys()), key="payment")
    ]

    paperless_billing = 1 if st.radio(
        "Paperless Billing", ["No", "Yes"], horizontal=True, key="paperless"
    ) == "Yes" else 0

    internet_options = {"No Internet": 0, "DSL": 1, "Fiber Optic": 2}
    internet_service = internet_options[
        st.selectbox("Internet Service", list(internet_options.keys()), key="internet")
    ]

    lines_options = {"No Phone Service": 0, "No": 1, "Yes": 2}
    multiple_lines = lines_options[
        st.selectbox("Multiple Lines", list(lines_options.keys()), key="lines")
    ]

# --- Column 3: Services & Personal ---
with col3:
    st.markdown("**Services**")
    online_security = 1 if st.radio(
        "Online Security", ["No", "Yes"], horizontal=True, key="sec"
    ) == "Yes" else 0
    online_backup = 1 if st.radio(
        "Online Backup", ["No", "Yes"], horizontal=True, key="backup"
    ) == "Yes" else 0
    device_protection = 1 if st.radio(
        "Device Protection", ["No", "Yes"], horizontal=True, key="device"
    ) == "Yes" else 0
    tech_support = 1 if st.radio(
        "Tech Support", ["No", "Yes"], horizontal=True, key="support"
    ) == "Yes" else 0
    streaming_tv = 1 if st.radio(
        "Streaming TV", ["No", "Yes"], horizontal=True, key="tv"
    ) == "Yes" else 0
    streaming_movies = 1 if st.radio(
        "Streaming Movies", ["No", "Yes"], horizontal=True, key="movies"
    ) == "Yes" else 0

# --- Buttons: Predict and Reset ---
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    predict_clicked = st.button("🎯 Predict", use_container_width=True)
with btn_col2:
    if st.button("↻ Reset", use_container_width=True):
        st.rerun()

# ============================================================
# PREDICTION LOGIC
# ============================================================
if predict_clicked:
    with st.spinner("Analyzing customer profile..."):

        # --- Step 1: Collect all input values into a dictionary ---
        # Personal fields use defaults (0) since removed from UI
        input_data = {
            "gender": 0,
            "SeniorCitizen": 0,
            "Partner": 0,
            "Dependents": 0,
            "tenure": tenure,
            "PhoneService": 1,  # always 1 (all customers have phone)
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        # --- Step 2: Arrange values in the exact order the model expects ---
        input_row = [input_data[col] for col in FEATURES]

        # --- Step 3: Scale the input (standardize) ---
        input_scaled = scaler.transform([input_row])

        # --- Step 4: Get churn probability from the model ---
        churn_probability = model.predict_proba(input_scaled)[0][1]
        churn_percentage = round(churn_probability * 100, 1)

        # --- Step 5: Classify risk level ---
        risk_label, risk_class, risk_bg_class, risk_emoji = classify_risk(churn_probability)

    # --- Display the prediction result ---
    st.markdown(f"""
    <div class="result-card {risk_bg_class}">
        <h2>🎯 Prediction Result</h2>
        <div style="font-size:2.5rem; margin-bottom:0.5rem">{risk_emoji}</div>
        <div class="result-percentage {risk_class}">{churn_percentage}%</div>
        <div class="result-label {risk_class}">{risk_label}</div>
        <div class="result-detail">
            Confidence Score: <span>{churn_percentage}%</span> &nbsp;·&nbsp;
            Model: <span>Random Forest</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # VISUALIZATION: Risk Gauge + Feature Importance
    # ============================================================
    viz_col1, viz_col2 = st.columns(2)

    # --- Gauge Chart ---
    with viz_col1:
        st.markdown('<div class="section-title">Risk Gauge</div>', unsafe_allow_html=True)
        gauge_color = (
            "#dc2626" if churn_probability >= 0.65
            else "#d97706" if churn_probability >= 0.35
            else "#16a34a"
        )
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_percentage,
            number={
                "suffix": "%",
                "font": {"size": 32, "color": gauge_color, "family": "Inter"}
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#94a3b8",
                    "tickfont": {"color": "#94a3b8", "size": 10}
                },
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "#f1f5f9",
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 35], "color": "#f0fdf4"},
                    {"range": [35, 65], "color": "#fffbeb"},
                    {"range": [65, 100], "color": "#fef2f2"},
                ],
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="white",
            font_color="#1e293b",
            height=220,
            margin=dict(t=10, b=10, l=25, r=25),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Feature Importance Bar Chart ---
    with viz_col2:
        st.markdown('<div class="section-title">Top Risk Factors</div>', unsafe_allow_html=True)
        importance = pd.Series(model.feature_importances_, index=FEATURES)
        top_factors = importance.sort_values(ascending=True).tail(6)
        fig_bar = go.Figure(go.Bar(
            x=top_factors.values,
            y=top_factors.index,
            orientation="h",
            marker_color="#2563eb",
            opacity=0.8,
        ))
        fig_bar.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#1e293b",
            height=220,
            margin=dict(t=10, b=10, l=10, r=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(tickfont=dict(size=10, color="#475569")),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ============================================================
    # RETENTION RECOMMENDATIONS
    # ============================================================
    st.markdown('<div class="section-title">💡 Retention Recommendations</div>', unsafe_allow_html=True)

    recommendations = []

    if contract == 0:
        recommendations.append((
            "📋 Upgrade Contract",
            "Month-to-month contracts have the highest churn risk. Offer a 10-15% discount to switch to an annual plan."
        ))
    if monthly_charges > 70:
        recommendations.append((
            "💰 Review Pricing",
            f"Monthly charges of ${monthly_charges:.0f} are above average. Consider a loyalty discount or bundle offer."
        ))
    if tenure < 12:
        recommendations.append((
            "🤝 Early Engagement",
            "First-year customers need attention. Assign a dedicated support contact during this critical period."
        ))
    if not tech_support:
        recommendations.append((
            "🛠 Offer Tech Support",
            "Customer has no tech support. A free 30-day trial can significantly reduce churn risk."
        ))
    if not online_security:
        recommendations.append((
            "🔒 Security Bundle",
            "No online security subscribed. Push a security + backup bundle as a retention upsell."
        ))
    if internet_service == 2:
        recommendations.append((
            "📡 Fiber Loyalty Perk",
            "Fiber optic users tend to churn more. Offer a speed upgrade or monthly credit."
        ))
    if payment_method == 0:
        recommendations.append((
            "🏦 Switch to Auto-Pay",
            "Electronic check payers churn more frequently. Offer a small discount for auto-pay setup."
        ))
    if not recommendations:
        recommendations.append((
            "✅ Low Risk Customer",
            "Strong retention signals detected. Consider upselling premium features."
        ))

    # Show up to 3 recommendations per row
    rec_row1 = st.columns(min(len(recommendations), 3))
    for i, (title, desc) in enumerate(recommendations[:3]):
        with rec_row1[i]:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">{title}</div>
                <div class="rec-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    if len(recommendations) > 3:
        st.markdown("<br>", unsafe_allow_html=True)
        rec_row2 = st.columns(min(len(recommendations) - 3, 3))
        for i, (title, desc) in enumerate(recommendations[3:6]):
            with rec_row2[i]:
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-title">{title}</div>
                    <div class="rec-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # --- Summary Bar ---
    st.markdown("<br>", unsafe_allow_html=True)
    contract_labels = {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}
    summary_contract = contract_labels[contract]

    st.markdown(f"""
    <div class="summary-bar">
        Tenure: <b>{tenure}m</b> &nbsp;·&nbsp;
        Contract: <b>{summary_contract}</b> &nbsp;·&nbsp;
        Monthly: <b>${monthly_charges:.0f}</b> &nbsp;·&nbsp;
        Risk: <b>{risk_label}</b> &nbsp;·&nbsp;
        Model: Random Forest &nbsp;·&nbsp; AUC: 0.84
    </div>
    """, unsafe_allow_html=True)


