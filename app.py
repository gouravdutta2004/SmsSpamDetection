import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="SMS Spam Detector Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
}
.title-text {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1E293B;
}
.subtitle-text {
    font-size: 1.1rem;
    color: #64748B;
    margin-bottom: 1.5rem;
}
.stButton>button {
    border-radius: 8px;
}
.metric-card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
.spam-badge {
    background-color: #FEE2E2;
    color: #991B1B;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #FCA5A5;
    text-align: center;
    font-size: 1.25rem;
    font-weight: bold;
}
.ham-badge {
    background-color: #DCFCE7;
    color: #166534;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #86EFAC;
    text-align: center;
    font-size: 1.25rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helper Functions & Resource Loading
# -----------------------------
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "SMSSpamCollection")


from src.preprocessing import clean_text


@st.cache_resource
def load_artifacts():
    # If pre-trained artifacts don't exist, trigger training pipeline
    if not os.path.exists(PIPELINE_PATH) or not os.path.exists(METRICS_PATH):
        from src.train import main as train_main
        train_main()

    pipeline = joblib.load(PIPELINE_PATH)
    
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return pipeline, metrics


import csv

@st.cache_data
def load_raw_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, sep="\t", names=["label", "message"], quoting=csv.QUOTE_NONE)
    return None


pipeline, metrics = load_artifacts()
raw_df = load_raw_dataset()

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <div class="title-text">📱 SMS Spam Detector Pro</div>
    <div class="subtitle-text">Production-Grade NLP SMS Classification & Model Analytics</div>
</div>
""", unsafe_allow_html=True)

# Top Info Bar
best_model_name = metrics.get("best_model", "Linear SVM")
total_msgs = metrics.get("total_samples", len(raw_df) if raw_df is not None else 5574)
spam_cnt = metrics.get("spam_count", (raw_df["label"] == "spam").sum() if raw_df is not None else 747)
ham_cnt = metrics.get("ham_count", (raw_df["label"] == "ham").sum() if raw_df is not None else 4827)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Dataset Messages", f"{total_msgs:,}")
with col_m2:
    st.metric("Spam Count", f"{spam_cnt:,}", f"{spam_cnt/total_msgs*100:.1f}%")
with col_m3:
    st.metric("Legitimate (Ham) Count", f"{ham_cnt:,}", f"{ham_cnt/total_msgs*100:.1f}%")
with col_m4:
    st.metric("Active Model", best_model_name, f"F1: {metrics.get('models', {}).get(best_model_name, {}).get('cv_f1_score', 0.94):.4f}")

st.divider()

# -----------------------------
# App Navigation Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🔮 Single Message Detector",
    "📁 Batch CSV Processing",
    "📊 Model Benchmarks & Data EDA"
])

# -----------------------------
# TAB 1: Single Message Detector
# -----------------------------
with tab1:
    st.subheader("Interactive SMS Analysis")
    st.write("Enter an SMS text below or choose from one of the preset sample messages to test the active classifier.")

    # Preset Sample Buttons
    st.markdown("**Try Sample Inputs:**")
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

    def set_sample_text(sample_text):
        st.session_state["user_text_input"] = sample_text

    if "user_text_input" not in st.session_state:
        st.session_state["user_text_input"] = ""

    with sample_col1:
        st.button(
            "🚨 Spam Example 1",
            width="stretch",
            on_click=set_sample_text,
            args=("WINNER!! You have won a $1,000 Walmart gift card! Call 800-123-4567 NOW to claim your prize.",)
        )
    with sample_col2:
        st.button(
            "🚨 Spam Example 2",
            width="stretch",
            on_click=set_sample_text,
            args=("URGENT! Your mobile number was awarded £2000 bonus cash. Text CLAIM to 88088 now!",)
        )
    with sample_col3:
        st.button(
            "✅ Ham Example 1",
            width="stretch",
            on_click=set_sample_text,
            args=("Hey mate, are we still meeting for dinner at 7pm tonight?",)
        )
    with sample_col4:
        st.button(
            "✅ Ham Example 2",
            width="stretch",
            on_click=set_sample_text,
            args=("Can you please send me the project report when you get a chance?",)
        )

    user_text = st.text_area(
        "SMS Text Input:",
        height=120,
        placeholder="Type or paste your SMS message here...",
        key="user_text_input"
    )

    if st.button("🔍 Classify Message", type="primary", width="stretch"):
        if not user_text.strip():
            st.warning("Please enter or select an SMS message to classify.")
        else:
            cleaned = clean_text(user_text)
            pred_class = pipeline.predict([cleaned])[0]  # 0: ham, 1: spam

            # Using calibrated probabilities from the Pipeline
            probs = pipeline.predict_proba([cleaned])[0]
            spam_prob = probs[1]
            confidence = probs[pred_class]

            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                if pred_class == 1:
                    st.markdown('<div class="spam-badge">🚨 SPAM MESSAGE DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ham-badge">✅ LEGITIMATE (HAM) MESSAGE</div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"**Classification Confidence:** `{confidence * 100:.2f}%`")
                st.progress(float(confidence))
                st.markdown(f"**Estimated Spam Probability:** `{spam_prob * 100:.2f}%`")

# -----------------------------
# TAB 2: Batch CSV Processing
# -----------------------------
with tab2:
    st.subheader("Batch File Classification")
    st.write("Upload a CSV file containing multiple SMS messages to perform bulk spam detection.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if len(batch_df) == 0:
                st.warning("The uploaded file contains no data rows.")
            else:
                st.success(f"Successfully loaded CSV with {len(batch_df)} records.")

                # Select text column
                text_columns = batch_df.select_dtypes(include=["object", "string"]).columns.tolist()
                if not text_columns:
                    st.error("No text columns found in the uploaded file.")
                else:
                    target_col = st.selectbox("Select column containing SMS messages:", text_columns)

                    if st.button("🚀 Process Batch Predictions", type="primary"):
                        with st.spinner("Analyzing messages..."):
                            cleaned_batch = batch_df[target_col].fillna("").astype(str).apply(clean_text)
                            preds = pipeline.predict(cleaned_batch)
                            
                            batch_df["Predicted_Label"] = np.where(preds == 1, "Spam", "Ham")

                            probs = pipeline.predict_proba(cleaned_batch)[:, 1]
                            batch_df["Spam_Probability"] = probs.round(4)

                        st.markdown("### 📋 Batch Results Summary")
                        b_spam = int((batch_df["Predicted_Label"] == "Spam").sum())
                        b_ham = int((batch_df["Predicted_Label"] == "Ham").sum())
                        total_len = len(batch_df)

                        b1, b2, b3 = st.columns(3)
                        b1.metric("Total Messages Processed", total_len)
                        b2.metric("Flagged Spam", b_spam, f"{b_spam/total_len*100:.1f}%")
                        b3.metric("Flagged Legitimate (Ham)", b_ham, f"{b_ham/total_len*100:.1f}%")

                        # Display Dataframe with filter
                        filter_option = st.radio("Filter Output:", ["All", "Spam Only", "Ham Only"], horizontal=True)
                        if filter_option == "Spam Only":
                            display_df = batch_df[batch_df["Predicted_Label"] == "Spam"]
                        elif filter_option == "Ham Only":
                            display_df = batch_df[batch_df["Predicted_Label"] == "Ham"]
                        else:
                            display_df = batch_df

                        st.dataframe(display_df, width="stretch")

                        # Export Download Button
                        csv_data = batch_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_data,
                            file_name="sms_spam_predictions.csv",
                            mime="text/csv",
                            type="primary"
                        )

        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")

# -----------------------------
# TAB 3: Model Benchmarks & Data EDA
# -----------------------------
with tab3:
    st.subheader("Model Benchmarks & Data Insights")

    benchmarks = metrics.get("models", {})
    if benchmarks:
        st.markdown("### 🏆 Classifier Performance Comparison")
        
        bench_df = pd.DataFrame(benchmarks).T.reset_index()
        bench_df.rename(columns={"index": "Model Name"}, inplace=True)
        
        st.dataframe(
            bench_df.style.highlight_max(subset=["cv_accuracy", "cv_precision", "cv_recall", "cv_f1_score"], color="#DCFCE7"),
            width="stretch"
        )

        # Plotly Bar Chart comparing models
        fig_metrics = go.Figure()
        metrics_to_plot = ["cv_accuracy", "cv_precision", "cv_recall", "cv_f1_score"]
        metric_labels = {
            "cv_accuracy": "CV Accuracy",
            "cv_precision": "CV Precision",
            "cv_recall": "CV Recall",
            "cv_f1_score": "CV F1 Score"
        }
        colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]

        for idx, metric_key in enumerate(metrics_to_plot):
            fig_metrics.add_trace(go.Bar(
                x=bench_df["Model Name"],
                y=bench_df[metric_key],
                name=metric_labels.get(metric_key, metric_key),
                marker_color=colors[idx]
            ))

        fig_metrics.update_layout(
            title="Model Evaluation Metrics Comparison (5-Fold Stratified Grouped CV)",
            barmode="group",
            yaxis=dict(range=[0.7, 1.05], title="Score"),
            xaxis=dict(title="Model Architecture"),
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_metrics, width="stretch")

    st.divider()

    if raw_df is not None:
        st.markdown("### 📈 Exploratory Data Analysis")
        eda_col1, eda_col2 = st.columns(2)

        with eda_col1:
            # Class distribution pie chart
            pie_data = raw_df["label"].value_counts().reset_index()
            pie_data.columns = ["Label", "Count"]
            fig_pie = px.pie(
                pie_data,
                values="Count",
                names="Label",
                title="UCI Dataset Class Distribution",
                color="Label",
                color_discrete_map={"ham": "#10B981", "spam": "#EF4444"},
                hole=0.4
            )
            st.plotly_chart(fig_pie, width="stretch")

        with eda_col2:
            # Message Length Distribution Histogram
            raw_df["msg_length"] = raw_df["message"].astype(str).str.len()
            fig_hist = px.histogram(
                raw_df,
                x="msg_length",
                color="label",
                nbins=50,
                title="Message Character Length Distribution by Class",
                color_discrete_map={"ham": "#10B981", "spam": "#EF4444"},
                barmode="overlay",
                opacity=0.7
            )
            fig_hist.update_layout(xaxis_title="Character Length", yaxis_title="Frequency", template="plotly_white")
            st.plotly_chart(fig_hist, width="stretch")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("SMS Spam Detection System • Built with Streamlit, scikit-learn & Plotly")