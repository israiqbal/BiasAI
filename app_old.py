"""
AI Bias Detection Platform
Multi-tier LLM explanation system with Gemini → Groq → Local fallback
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Import LLM utilities
from llm_utils import get_bias_explanation

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Bias Analyzer", layout="wide")

# ----------------------------
# CUSTOM CSS (Premium UI)
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}
.block-container {
    padding-top: 2rem;
}
.explanation-box {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    border-left: 4px solid #3b82f6;
}
.source-badge {
    display: inline-block;
    background: rgba(59, 130, 246, 0.2);
    color: #93c5fd;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("⚙️ Navigation")
section = st.sidebar.radio("Go to", ["Home", "Upload & Analyze"])

# ----------------------------
# HEADER
# ----------------------------
st.markdown('<div class="big-title">AI Bias Detection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enterprise-grade fairness analysis tool</div>', unsafe_allow_html=True)

# ----------------------------
# HOME PAGE
# ----------------------------
if section == "Home":
    st.write("""
    ### 🚀 What this tool does:
    - Detects bias in ML predictions  
    - Applies mitigation techniques  
    - Generates professional reports
    - AI-powered explanations using Gemini/Groq  
    """)

# ----------------------------
# MAIN ANALYSIS
# ----------------------------
if section == "Upload & Analyze":

    file = st.file_uploader("Upload CSV Dataset")

    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())

        target = st.selectbox("Target Column", df.columns)
        sensitive = st.selectbox("Sensitive Attribute", df.columns)

        if st.button("Run Analysis"):

            with st.spinner("Analyzing bias... 🔍"):
                
                # ----------------------------
                # MODEL
                # ----------------------------
                df[target] = df[target].apply(lambda x: 1 if ">50K" in str(x) else 0)

                X = df.drop(columns=[target])
                y = df[target]

                X_encoded = pd.get_dummies(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.2, random_state=42
                )

                scaler = StandardScaler(with_mean=False)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = LogisticRegression(max_iter=5000, solver='liblinear')
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                df_test = df.loc[y_test.index].copy()
                df_test['pred'] = preds

                g1 = df_test[df_test[sensitive] == df[sensitive].unique()[0]]['pred'].mean()
                g2 = df_test[df_test[sensitive] == df[sensitive].unique()[1]]['pred'].mean()

                di_ratio = g2 / g1 if g1 > 0 else 0

                # ----------------------------
                # MITIGATION
                # ----------------------------
                X2 = df.drop(columns=[target, sensitive])
                y2 = df[target]

                X2 = pd.get_dummies(X2)

                X_train2, X_test2, y_train2, y_test2 = train_test_split(
                    X2, y2, test_size=0.2, random_state=42
                )

                idx = X_test2.index

                scaler2 = StandardScaler(with_mean=False)
                X_train2 = scaler2.fit_transform(X_train2)
                X_test2 = scaler2.transform(X_test2)

                model2 = LogisticRegression(max_iter=5000, solver='liblinear')
                model2.fit(X_train2, y_train2)
                preds2 = model2.predict(X_test2)

                df_test2 = df.loc[idx].copy()
                df_test2['pred'] = preds2

                g1_after = df_test2[df_test2[sensitive] == df[sensitive].unique()[0]]['pred'].mean()
                g2_after = df_test2[df_test2[sensitive] == df[sensitive].unique()[1]]['pred'].mean()

            # ----------------------------
            # METRICS DISPLAY
            # ----------------------------
            col1, col2 = st.columns(2)
            col1.metric("Before Bias (Group1)", round(g1, 2))
            col2.metric("Before Bias (Group2)", round(g2, 2))

            col3, col4 = st.columns(2)
            col3.metric("After Bias (Group1)", round(g1_after, 2))
            col4.metric("After Bias (Group2)", round(g2_after, 2))

            # ----------------------------
            # STATUS
            # ----------------------------
            if di_ratio < 0.8:
                st.error("⚠️ High Bias Detected")
            else:
                st.success("✅ Fair Model")

            # ----------------------------
            # LLM EXPLANATION (Gemini → Groq → Local)
            # ----------------------------
            st.subheader("🤖 AI-Powered Analysis")
            
            with st.spinner("Generating explanation..."):
                bias_findings = {
                    'g1_before': g1,
                    'g2_before': g2,
                    'g1_after': g1_after,
                    'g2_after': g2_after,
                    'di_ratio': di_ratio
                }
                
                # Get API keys from Streamlit secrets or environment
                gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
                groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
                
                result = get_bias_explanation(
                    bias_findings,
                    gemini_key=gemini_key,
                    groq_key=groq_key,
                    use_fallback=True
                )
                
                if result['success']:
                    st.markdown(f"""
                    <div class="explanation-box">
                        {result['explanation']}
                        <div class="source-badge">Source: {result['source'].upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Could not generate explanation at this time.")

            # ----------------------------
            # CHARTS
            # ----------------------------
            st.subheader("📊 Bias Visualization")
            
            fig, ax = plt.subplots()
            ax.bar(['Group 1', 'Group 2'], [g1, g2], color=['#3b82f6', '#ef4444'])
            ax.set_title("Before Mitigation")
            ax.set_ylabel("Positive Rate")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.bar(['Group 1', 'Group 2'], [g1_after, g2_after], color=['#10b981', '#f59e0b'])
            ax2.set_title("After Mitigation")
            ax2.set_ylabel("Positive Rate")
            st.pyplot(fig2)

            # ----------------------------
            # PDF REPORT
            # ----------------------------
            def create_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()

                content = []

                content.append(Paragraph("AI Bias Detection Report", styles['Title']))
                content.append(Spacer(1, 10))

                content.append(Paragraph(f"Sensitive Attribute: {sensitive}", styles['Normal']))
                content.append(Paragraph(f"Target Column: {target}", styles['Normal']))
                content.append(Spacer(1, 10))

                content.append(Paragraph("Before Mitigation", styles['Heading2']))
                content.append(Paragraph(f"Group 1 Rate: {g1:.2%}", styles['Normal']))
                content.append(Paragraph(f"Group 2 Rate: {g2:.2%}", styles['Normal']))
                content.append(Paragraph(f"Disparate Impact Ratio: {di_ratio:.2f}", styles['Normal']))
                content.append(Spacer(1, 10))

                content.append(Paragraph("After Mitigation", styles['Heading2']))
                content.append(Paragraph(f"Group 1 Rate: {g1_after:.2%}", styles['Normal']))
                content.append(Paragraph(f"Group 2 Rate: {g2_after:.2%}", styles['Normal']))
                content.append(Spacer(1, 10))

                # Save chart
                fig.savefig("chart_before.png", dpi=100, bbox_inches='tight')
                content.append(Paragraph("Visualization - Before Mitigation", styles['Heading3']))
                content.append(Image("chart_before.png", width=400, height=200))
                content.append(Spacer(1, 10))

                fig2.savefig("chart_after.png", dpi=100, bbox_inches='tight')
                content.append(Paragraph("Visualization - After Mitigation", styles['Heading3']))
                content.append(Image("chart_after.png", width=400, height=200))

                doc.build(content)
                buffer.seek(0)
                return buffer

            pdf = create_pdf()

            st.download_button(
                label="📄 Download Premium Report",
                data=pdf,
                file_name="bias_report.pdf",
                mime="application/pdf"
            )
