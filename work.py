import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import io
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def encode_target_column(df, target):
    """Smart binary encoding of target column for any dataset."""
    col = df[target]
    unique_vals = col.dropna().unique()

    # Already binary 0/1
    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        return col.astype(int)

    # Exactly 2 unique values
    if len(unique_vals) == 2:
        sorted_vals = sorted(unique_vals, key=str)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return col.map(mapping).astype(int)

    # Numeric with many values — median threshold
    if pd.api.types.is_numeric_dtype(col):
        median_val = col.median()
        return (col > median_val).astype(int)

    # Categorical with many values — most frequent = 0, rest = 1
    most_frequent = col.value_counts().index[0]
    return (col != most_frequent).astype(int)


def safe_disparate_impact(g1_rate, g2_rate):
    """Calculate disparate impact ratio safely, avoiding division by zero."""
    if g1_rate > 0:
        return g2_rate / g1_rate
    elif g2_rate > 0:
        return g1_rate / g2_rate
    return 1.0


# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Bias Analyzer", layout="wide")

# ------------------ STYLING ------------------
st.markdown("""
<style>
body { background-color: #0f172a; color: white; }
.big-title { font-size: 42px; font-weight: 800; text-align: center; }
.subtitle { text-align: center; color: #94a3b8; margin-bottom: 30px; }
.card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="big-title">AI Bias Detection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enterprise-grade fairness analysis tool</div>', unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analyze"])

# ------------------ HOME ------------------
if page == "Home":
    st.markdown("""
    ### 🚀 What this tool does:
    - Detects bias in ML predictions  
    - Applies mitigation techniques  
    - Generates professional reports  
    """)

# ------------------ ANALYSIS ------------------
if page == "Analyze":

    file = st.file_uploader("📂 Upload CSV Dataset")

    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())

        target = st.selectbox("🎯 Select Target Column", df.columns)
        sensitive = st.selectbox("⚠️ Select Sensitive Attribute", df.columns)

        # Validate & select groups for sensitive attribute
        unique_groups = df[sensitive].dropna().unique()

        if len(unique_groups) < 2:
            st.error("⚠️ The sensitive attribute must have at least 2 unique values for comparison.")
        else:
            if len(unique_groups) == 2:
                group1, group2 = unique_groups[0], unique_groups[1]
            else:
                st.info(f"ℹ️ The sensitive attribute has {len(unique_groups)} unique values. Select 2 groups to compare.")
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    group1 = st.selectbox("Group 1", unique_groups, index=0, key="group1_select")
                with gcol2:
                    remaining = [g for g in unique_groups if g != group1]
                    group2 = st.selectbox("Group 2", remaining, index=0, key="group2_select")

            if st.button("🚀 Run Analysis"):

                # Animation
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                with st.spinner("Running AI analysis..."):

                    try:
                        # ------------------ MODEL ------------------
                        # Smart target encoding (works with any CSV)
                        df[target] = encode_target_column(df, target)

                        X = df.drop(columns=[target])
                        y = df[target]

                        X = pd.get_dummies(X)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        scaler = StandardScaler(with_mean=False)
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                        model = LogisticRegression(max_iter=5000, solver='liblinear')
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        df_test = df.loc[y_test.index].copy()
                        df_test['pred'] = preds

                        # Safe group rate calculation
                        g1_mask = df_test[sensitive] == group1
                        g2_mask = df_test[sensitive] == group2
                        g1 = df_test.loc[g1_mask, 'pred'].mean() if g1_mask.any() else 0.0
                        g2 = df_test.loc[g2_mask, 'pred'].mean() if g2_mask.any() else 0.0

                        di_ratio = safe_disparate_impact(g1, g2)
                        bias_score = abs(g1 - g2)

                        # ------------------ MITIGATION ------------------
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

                        # Safe group rate calculation after mitigation
                        g1_after_mask = df_test2[sensitive] == group1
                        g2_after_mask = df_test2[sensitive] == group2
                        g1_after = df_test2.loc[g1_after_mask, 'pred'].mean() if g1_after_mask.any() else 0.0
                        g2_after = df_test2.loc[g2_after_mask, 'pred'].mean() if g2_after_mask.any() else 0.0

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("💡 Make sure your target column is suitable for binary classification and the sensitive attribute has distinct groups.")
                        st.stop()

                st.divider()

                # Group labels
                g1_name = str(group1)
                g2_name = str(group2)

                # ------------------ METRICS ------------------
                st.subheader("📊 Key Metrics")

                col1, col2, col3 = st.columns(3)
                col1.metric("Bias Score", round(bias_score, 2))
                col2.metric("Disparate Impact", round(di_ratio, 2))
                col3.metric("Improvement", round(abs(g1 - g2) - abs(g1_after - g2_after), 2))

                # ------------------ INTERACTIVE CHART ------------------
                st.subheader("📈 Interactive Comparison")

                chart_df = pd.DataFrame({
                    "Group": [f"Before {g1_name}", f"Before {g2_name}", f"After {g1_name}", f"After {g2_name}"],
                    "Value": [g1, g2, g1_after, g2_after]
                })

                fig = px.bar(chart_df, x="Group", y="Value", color="Group")
                st.plotly_chart(fig, use_container_width=True)

                # ------------------ STATUS ------------------
                if di_ratio < 0.8:
                    st.error("⚠️ High Bias Detected")
                else:
                    st.success("✅ Model is Fair")

                # ------------------ PDF ------------------
                def create_onepage_pdf():
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer)
                    styles = getSampleStyleSheet()

                    content = []

                    # ------------------ TITLE ------------------
                    content.append(Paragraph("AI Bias Analysis Report", styles['Title']))
                    content.append(Spacer(1, 8))

                    # ------------------ EXEC SUMMARY ------------------
                    bias_diff = abs(g1 - g2)
                    improvement = abs(g1 - g2) - abs(g1_after - g2_after)

                    summary = f"""
                    This report evaluates bias between <b>{sensitive}</b> groups
                    ({g1_name} vs {g2_name})
                    in predicting <b>{target}</b>. Initial disparity: {bias_diff:.2f}.
                    Post-mitigation improvement: {improvement:.2f}.
                    """

                    content.append(Paragraph(summary, styles['Normal']))
                    content.append(Spacer(1, 10))

                    # ------------------ DATA CONTEXT ------------------
                    content.append(Paragraph("Data Context", styles['Heading3']))

                    context_text = f"""
                    Sensitive Attribute: <b>{sensitive}</b><br/>
                    Target Variable: <b>{target}</b><br/>
                    Groups Compared: <b>{g1_name}</b> vs <b>{g2_name}</b><br/>
                    Comparison: Prediction rates across groups
                    """

                    content.append(Paragraph(context_text, styles['Normal']))
                    content.append(Spacer(1, 10))

                    # ------------------ METRICS TABLE ------------------
                    table_data = [
                        ["Metric", "Before", "After"],
                        [g1_name, f"{g1:.2f}", f"{g1_after:.2f}"],
                        [g2_name, f"{g2:.2f}", f"{g2_after:.2f}"],
                        ["Bias Gap", f"{abs(g1-g2):.2f}", f"{abs(g1_after-g2_after):.2f}"]
                    ]

                    table = Table(table_data, colWidths=[120, 80, 80])
                    table.setStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('FONTSIZE', (0, 0), (-1, -1), 8)
                    ])

                    content.append(table)
                    content.append(Spacer(1, 10))

                    # ------------------ CHART ------------------
                    fig_pdf, ax_pdf = plt.subplots(figsize=(4, 2.2))

                    ax_pdf.bar(
                        [f'B-{g1_name[:4]}', f'B-{g2_name[:4]}', f'A-{g1_name[:4]}', f'A-{g2_name[:4]}'],
                        [g1, g2, g1_after, g2_after],
                        color=['#3b82f6', '#ef4444', '#10b981', '#f59e0b']
                    )

                    ax_pdf.set_title("Bias Comparison", fontsize=8)
                    ax_pdf.tick_params(axis='x', labelsize=6)
                    ax_pdf.tick_params(axis='y', labelsize=6)

                    fig_pdf.savefig("chart.png", bbox_inches='tight')
                    plt.close(fig_pdf)

                    content.append(Image("chart.png", width=320, height=150))
                    content.append(Spacer(1, 8))

                    # ------------------ INSIGHTS ------------------
                    content.append(Paragraph("Insights", styles['Heading3']))

                    if bias_diff > 0.2:
                        level = "High bias observed"
                    elif bias_diff > 0.1:
                        level = "Moderate bias observed"
                    else:
                        level = "Low bias observed"

                    insight_text = f"""
                    {level}. Removing <b>{sensitive}</b> improved fairness.
                    Model now shows more balanced predictions.
                    """

                    content.append(Paragraph(insight_text, styles['Normal']))
                    content.append(Spacer(1, 6))

                    # ------------------ RECOMMENDATION ------------------
                    content.append(Paragraph("Recommendation", styles['Heading3']))

                    rec_text = """
                    Avoid using sensitive attributes directly.
                    Apply fairness checks in deployment pipelines.
                    """

                    content.append(Paragraph(rec_text, styles['Normal']))

                    doc.build(content)
                    buffer.seek(0)
                    return buffer

                pdf = create_onepage_pdf()

                st.download_button(
                    label="📄 Download Premium Report",
                    data=pdf,
                    file_name="bias_report.pdf",
                    mime="application/pdf"
                )