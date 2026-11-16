
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FraudGuard | Credit Card Security",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: #f3f6fc;
}

/* Main content spacing */
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101d3a 0%, #172c50 100%);
    min-width: 260px;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c5d3eb !important;
}

[data-testid="stSidebar"] hr {
    border-color: #34496b;
}

/* Navigation radio buttons */
[data-testid="stSidebar"] .stRadio label {
    padding: 7px 4px;
}

/* Hide default streamlit footer */
footer {
    visibility: hidden;
}

/* Header */
.hero {
    background: linear-gradient(
        115deg,
        #14294c 0%,
        #244b83 55%,
        #6653b8 100%
    );
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(31, 56, 100, 0.16);
}

.hero h1 {
    color: white !important;
    font-size: 34px !important;
    font-weight: 750 !important;
    margin-bottom: 8px;
}

.hero p {
    color: #dce7ff !important;
    font-size: 15px;
    margin-bottom: 0;
}

/* Section heading */
.section-heading {
    color: #192e50;
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #e6ebf3;
    border-radius: 15px;
    padding: 22px 20px;
    min-height: 130px;
    box-shadow: 0 4px 15px rgba(20, 40, 80, 0.045);
}

.metric-label {
    color: #718096;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    color: #172c50;
    font-size: 29px;
    font-weight: 750;
}

.metric-icon {
    font-size: 23px;
    margin-bottom: 8px;
}

/* Information cards */
.info-card {
    background: white;
    border: 1px solid #e6ebf3;
    padding: 22px;
    border-radius: 15px;
    margin-bottom: 15px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(100deg, #2454a6, #6653b8);
    color: white !important;
    border: none;
    border-radius: 9px;
    padding: 10px 22px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #172c50;
    border: none;
}

/* Download buttons */
.stDownloadButton > button {
    border-radius: 9px;
    font-weight: 600;
}

/* Input fields */
.stTextInput input,
.stNumberInput input {
    border-radius: 8px;
}

/* Dataframes and charts */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #e1e7f0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


@st.cache_data
def load_dataset():
    data = pd.read_csv("dataset/creditcard.csv")
    return data.drop_duplicates().copy()


try:
    model, scaler = load_model()
    df = load_dataset()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# ==========================================
# DATA INFORMATION
# ==========================================

feature_columns = [
    col for col in df.columns if col != "Class"
]

medians = df[feature_columns].median()

total_transactions = len(df)

fraud_count = int((df["Class"] == 1).sum())

genuine_count = int((df["Class"] == 0).sum())

fraud_percentage = (
    fraud_count / total_transactions * 100
)


# ==========================================
# HELPER FUNCTION FOR METRIC CARDS
# ==========================================

def metric_card(icon, label, value, color):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 15px 0 25px 0;">
        <div style="font-size:35px;">💳</div>
        <h2 style="color:white; margin-bottom:0;">
            FraudGuard
        </h2>
        <p style="color:#a9bedf; font-size:12px;">
            INTELLIGENT TRANSACTION SECURITY
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Fraud Prediction",
            "Model Performance",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div style="
        background:#233d64;
        padding:15px;
        border-radius:12px;
        margin-top:20px;
    ">
        <div style="font-size:13px; color:#a9bedf;">
            SYSTEM STATUS
        </div>
        <h4 style="color:#6ee7ad; margin:8px 0;">
            ● Model Active
        </h4>
        <p style="font-size:11px; color:#c5d3eb;">
            Machine learning engine loaded successfully.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#91a6c8; font-size:11px; "
        "margin-top:50px;'>CREDIT CARD FRAUD DETECTION<br>"
        "Machine Learning Project | 2026</p>",
        unsafe_allow_html=True
    )


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Credit Card Fraud Detection</h1>
        <p>
            Intelligent machine learning system for
            detecting suspicious financial transactions.
        </p>
        <p style="margin-top:18px; font-size:12px;">
            SECURITY ANALYTICS &nbsp; | &nbsp;
            MACHINE LEARNING &nbsp; | &nbsp;
            REAL-TIME PREDICTION DEMO
        </p>
    </div>
    """, unsafe_allow_html=True)


    # ==========================================
    # MODERN INTERACTIVE ANALYTICS
    # ==========================================

    st.markdown(
        '<div class="section-heading">Transaction Analytics</div>',
        unsafe_allow_html=True
    )

    # Prepare chart data
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        with st.container(border=True):

            st.subheader("Transaction Distribution")

            pie_data = pd.DataFrame({
                "Type": ["Genuine", "Fraudulent"],
                "Count": [genuine_count, fraud_count]
            })

            fig1 = px.pie(
                pie_data,
                names="Type",
                values="Count",
                hole=0.65,
                color="Type",
                color_discrete_map={
                    "Genuine": "#16A36A",
                    "Fraudulent": "#E05252"
                },
                title="Genuine vs Fraudulent"
            )

            fig1.update_traces(
                textposition="inside",
                textinfo="label+percent",
                hole=0.65
            )

            fig1.update_layout(
                height=350,
                showlegend=True,
                legend_title_text="Transaction Type",
                margin=dict(t=60, b=20, l=20, r=20),
                paper_bgcolor="white",
                plot_bgcolor="white"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

    with chart_col2:

        with st.container(border=True):

            st.subheader("Fraud Transaction Amounts")

            fraud_data = df[df["Class"] == 1]

            fig2 = px.histogram(
                fraud_data,
                x="Amount",
                nbins=30,
                title="Amount Distribution of Fraud",
                color_discrete_sequence=["#E05252"]
            )

            fig2.update_layout(
                height=350,
                xaxis_title="Transaction Amount",
                yaxis_title="Number of Transactions",
                margin=dict(t=60, b=30, l=20, r=20),
                paper_bgcolor="white",
                plot_bgcolor="white"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    # ==========================================
    # SECOND ROW OF ANALYTICS
    # ==========================================

    st.markdown(
        '<div class="section-heading">Detailed Fraud Analysis</div>',
        unsafe_allow_html=True
    )

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:

        with st.container(border=True):

            st.subheader("Transaction Amount Comparison")

            comparison_data = df[df["Amount"] > 0].copy()

            comparison_data["Transaction Type"] = (
                comparison_data["Class"].map({
                    0: "Genuine",
                    1: "Fraudulent"
                })
            )

            fig3 = px.box(
                comparison_data,
                x="Transaction Type",
                y="Amount",
                color="Transaction Type",
                color_discrete_map={
                    "Genuine": "#4267B2",
                    "Fraudulent": "#E05252"
                },
                log_y=True,
                title="Amount Comparison (Log Scale)"
            )

            fig3.update_layout(
                height=350,
                xaxis_title="Transaction Category",
                yaxis_title="Amount (Log Scale)",
                showlegend=False,
                margin=dict(t=60, b=30, l=20, r=20),
                paper_bgcolor="white",
                plot_bgcolor="white"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

    with chart_col4:

        with st.container(border=True):

            st.subheader("Fraud Activity Over Time")

            time_data = fraud_data.copy()

            time_data["Time (Hours)"] = (
                time_data["Time"] / 3600
            )

            fig4 = px.histogram(
                time_data,
                x="Time (Hours)",
                nbins=30,
                title="Fraud Transactions Across Time",
                color_discrete_sequence=["#7956B8"]
            )

            fig4.update_layout(
                height=350,
                xaxis_title="Time (Hours)",
                yaxis_title="Fraud Transactions",
                margin=dict(t=60, b=30, l=20, r=20),
                paper_bgcolor="white",
                plot_bgcolor="white"
            )

            st.plotly_chart(
                fig4,
                use_container_width=True
            )

    # ==========================================
    # FRAUD INSIGHTS
    # ==========================================

    st.markdown(
        '<div class="section-heading">Key Fraud Insights</div>',
        unsafe_allow_html=True
    )

    avg_fraud_amount = fraud_data["Amount"].mean()
    avg_genuine_amount = df[df["Class"] == 0]["Amount"].mean()

    insight1, insight2, insight3 = st.columns(3)

    with insight1:
        metric_card(
            "🚨",
            "Detected Fraud Cases",
            f"{fraud_count:,}",
            "#E05252"
        )

    with insight2:
        metric_card(
            "💰",
            "Average Fraud Amount",
            f"${avg_fraud_amount:.2f}",
            "#7956B8"
        )

    with insight3:
        metric_card(
            "🏦",
            "Average Genuine Amount",
            f"{avg_genuine_amount:.2f}",
            "#15976A"
        )

    # Metric cards

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "📊",
            "Total Transactions",
            f"{total_transactions:,}",
            "#2454a6"
        )

    with c2:
        metric_card(
            "🟢",
            "Genuine Transactions",
            f"{genuine_count:,}",
            "#15976a"
        )

    with c3:
        metric_card(
            "🚨",
            "Fraud Transactions",
            f"{fraud_count:,}",
            "#e05252"
        )

    with c4:
        metric_card(
            "📈",
            "Fraud Percentage",
            f"{fraud_percentage:.3f}%",
            "#7956b8"
        )

    st.markdown(
        '<div class="section-heading">Transaction Analytics</div>',
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        with st.container(border=True):

            st.subheader("Transaction Distribution")

            chart_data = pd.DataFrame({
                "Transaction Type": [
                    "Genuine",
                    "Fraudulent"
                ],
                "Transactions": [
                    genuine_count,
                    fraud_count
                ]
            })

            st.bar_chart(
                chart_data.set_index("Transaction Type"),
                color="#4267b2"
            )

            st.caption(
                "Comparison of genuine and fraudulent transactions."
            )

    with chart_col2:

        with st.container(border=True):

            st.subheader("Fraud vs Genuine")

            pie_data = pd.DataFrame({
                "Transaction Type": [
                    "Genuine",
                    "Fraudulent"
                ],
                "Percentage": [
                    100 - fraud_percentage,
                    fraud_percentage
                ]
            })

            st.bar_chart(
                pie_data.set_index("Transaction Type"),
                color="#8a65bd"
            )

            st.caption(
                "Percentage distribution of both transaction classes."
            )

    st.markdown(
        '<div class="section-heading">Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    graph1, graph2 = st.columns(2)

    with graph1:

        if os.path.exists("graphs/fraud_distribution.png"):
            st.image(
                "graphs/fraud_distribution.png",
                use_container_width=True
            )

        if os.path.exists("graphs/amount_comparison.png"):
            st.image(
                "graphs/amount_comparison.png",
                use_container_width=True
            )

    with graph2:

        if os.path.exists("graphs/amount_distribution.png"):
            st.image(
                "graphs/amount_distribution.png",
                use_container_width=True
            )

        if os.path.exists("graphs/correlation_heatmap.png"):
            st.image(
                "graphs/correlation_heatmap.png",
                use_container_width=True
            )


# ==========================================
# FRAUD PREDICTION PAGE
# ==========================================

elif page == "Fraud Prediction":

    st.markdown("""
    <div class="hero">
        <h1>Transaction Fraud Analysis</h1>
        <p>
            Upload transaction data and let the trained
            machine learning model identify suspicious activity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Upload Transaction Dataset</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Upload a CSV containing the same input features as "
        "the original dataset: Time, V1-V28 and Amount. "
        "The Class column is optional."
    )

    uploaded_file = st.file_uploader(
        "Choose your transaction CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            input_data = pd.read_csv(uploaded_file)

            input_data = input_data.drop(
                columns=["Class"],
                errors="ignore"
            )

            missing_columns = [
                col for col in feature_columns
                if col not in input_data.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:

                input_data = input_data[feature_columns]

                input_data = input_data.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                input_data = input_data.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                input_data = input_data.fillna(medians)

                scaled_data = scaler.transform(input_data)

                predictions = model.predict(scaled_data)

                probabilities = model.predict_proba(
                    scaled_data
                )[:, 1]

                results = input_data.copy()

                results["Prediction"] = [
                    "Fraudulent" if p == 1 else "Genuine"
                    for p in predictions
                ]

                results["Fraud Probability (%)"] = (
                    probabilities * 100
                ).round(2)

                fraud_total = int(
                    (predictions == 1).sum()
                )

                genuine_total = int(
                    (predictions == 0).sum()
                )

                st.success("Transaction analysis completed!")

                a, b, c = st.columns(3)

                with a:
                    metric_card(
                        "📊",
                        "Transactions Tested",
                        f"{len(predictions):,}",
                        "#2454a6"
                    )

                with b:
                    metric_card(
                        "🚨",
                        "Fraud Detected",
                        f"{fraud_total:,}",
                        "#e05252"
                    )

                with c:
                    metric_card(
                        "🟢",
                        "Genuine",
                        f"{genuine_total:,}",
                        "#15976a"
                    )

                st.markdown(
                    '<div class="section-heading">Prediction Results</div>',
                    unsafe_allow_html=True
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True
                )

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇ Download Prediction Report",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(f"Prediction error: {e}")

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">Try Sample Transaction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Test the model using a transaction from the existing dataset."
    )

    if st.button("🔍 Analyze Random Transaction"):

        sample = df.sample(n=1)

        actual_class = int(sample["Class"].iloc[0])

        sample_features = sample[feature_columns]

        scaled_sample = scaler.transform(sample_features)

        prediction = int(
            model.predict(scaled_sample)[0]
        )

        probability = model.predict_proba(
            scaled_sample
        )[0][1]

        st.markdown(
            f"""
            <div class="info-card">
                <h3>Transaction Analysis Result</h3>
                <p><b>Transaction Amount:</b>
                {float(sample["Amount"].iloc[0]):.2f}</p>
                <p><b>Fraud Probability:</b>
                {probability * 100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
        else:
            st.success("✅ GENUINE TRANSACTION")

        st.write(
            "Actual dataset label:",
            "Fraudulent" if actual_class == 1 else "Genuine"
        )

        if prediction == actual_class:
            st.success("Prediction matches the actual dataset label.")
        else:
            st.warning("Prediction differs from the actual dataset label.")


# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    st.markdown("""
    <div class="hero">
        <h1>Model Performance</h1>
        <p>
            Evaluate machine learning algorithms and
            compare their fraud detection capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    comparison_path = "models/model_comparison.csv"

    if os.path.exists(comparison_path):

        results = pd.read_csv(comparison_path)

        st.markdown(
            '<div class="section-heading">Algorithm Comparison</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            results.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "ROC AUC": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        performance_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        st.markdown(
            '<div class="section-heading">Performance Metrics</div>',
            unsafe_allow_html=True
        )

        st.bar_chart(
            results.set_index("Model")[performance_columns]
        )

        best_row = results.loc[
            results["F1 Score"].idxmax()
        ]

        st.success(
            f"Best Model based on F1 Score: {best_row['Model']}"
        )

    else:

        st.warning("Model comparison file not found.")

    st.markdown(
        '<div class="section-heading">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    if os.path.exists("graphs/confusion_matrix.png"):

        st.image(
            "graphs/confusion_matrix.png",
            use_container_width=False,
            width=650
        )

    if os.path.exists("graphs/model_comparison.png"):

        st.image(
            "graphs/model_comparison.png",
            use_container_width=True
        )


# ==========================================
# ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.markdown("""
    <div class="hero">
        <h1>About FraudGuard</h1>
        <p>
            An intelligent approach to financial security
            using machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>Credit Card Fraud Detection System</h3>
        <p>
        This project uses machine learning algorithms to
        identify fraudulent credit card transactions.
        It analyses transaction patterns and classifies
        transactions as genuine or fraudulent.
        </p>
        <p>
        The system uses data preprocessing, SMOTE,
        Logistic Regression and Random Forest to develop
        and evaluate fraud detection models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Technologies Used</div>',
        unsafe_allow_html=True
    )

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("""
        <div class="info-card">
            <h4>🐍 Programming</h4>
            <p>Python</p>
            <p>NumPy</p>
            <p>Pandas</p>
        </div>
        """, unsafe_allow_html=True)

    with tech2:
        st.markdown("""
        <div class="info-card">
            <h4>🤖 Machine Learning</h4>
            <p>Scikit-learn</p>
            <p>SMOTE</p>
            <p>Random Forest</p>
            <p>Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)

    with tech3:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Visualization</h4>
            <p>Matplotlib</p>
            <p>Seaborn</p>
            <p>Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Project Workflow</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h4>01. Dataset Collection</h4>
        <p>Collect credit card transaction data.</p>
        <hr>
        <h4>02. Data Preprocessing</h4>
        <p>Clean and prepare transaction records.</p>
        <hr>
        <h4>03. Exploratory Data Analysis</h4>
        <p>Understand transaction patterns through graphs.</p>
        <hr>
        <h4>04. Handling Imbalanced Data</h4>
        <p>Apply SMOTE to the training dataset.</p>
        <hr>
        <h4>05. Model Training</h4>
        <p>Train Logistic Regression and Random Forest.</p>
        <hr>
        <h4>06. Model Evaluation</h4>
        <p>Compare accuracy, precision, recall and F1 score.</p>
        <hr>
        <h4>07. Fraud Prediction</h4>
        <p>Predict fraudulent transactions using the trained model.</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    color:#8290a5;
    font-size:12px;
    padding:10px;
">
    FraudGuard | Credit Card Fraud Detection System
    <br>
    Machine Learning Project | 2026
</div>
""", unsafe_allow_html=True)
