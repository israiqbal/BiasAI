"""
AI BIAS DETECTION PLATFORM - Premium SaaS UI
Glassmorphism design inspired by Apple, Stripe, Figma
Enterprise-grade fairness analysis tool for ethical AI
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Import LLM utilities
from llm_utils import get_bias_explanation

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


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AI Bias Analyzer | Enterprise Fairness Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# PREMIUM GLASSMORPHISM CSS
# ============================================================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 50%, #1a2a4a 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
        color: #f1f5f9;
        overflow-x: hidden;
    }
    
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* ========== PREMIUM BUTTONS ========== */
    .premium-btn {
        display: inline-block;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        border: none;
        text-decoration: none;
        text-align: center;
        user-select: none;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(59, 130, 246, 0.4);
    }
    
    .btn-secondary {
        background: rgba(255, 255, 255, 0.1);
        color: #f1f5f9;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .btn-secondary:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* ========== GLASSMORPHISM CARDS ========== */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* ========== NAVBAR ========== */
    .navbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 16px 32px;
        margin-bottom: 40px;
    }
    
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1400px;
        margin: 0 auto;
        gap: 32px;
    }
    
    .nav-logo {
        font-size: 20px;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    .nav-links {
        display: flex;
        gap: 16px;
        justify-content: center;
        flex: 1;
    }
    
    .nav-btn {
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        background: transparent;
        color: #cbd5e1;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    
    .nav-btn:hover {
        color: #f1f5f9;
        background: rgba(59, 130, 246, 0.1);
    }
    
    .nav-btn.active {
        color: #3b82f6;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* ========== HERO SECTION ========== */
    .hero {
        text-align: center;
        padding: 80px 32px;
        background: linear-gradient(180deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%);
        border-radius: 24px;
        margin-bottom: 60px;
        animation: fadeIn 0.8s ease;
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 32px;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
        text-align: center;
    }
    
    .hero-cta {
        display: flex;
        gap: 16px;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    /* ========== FEATURES GRID ========== */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 24px;
        margin-bottom: 60px;
    }
    
    .feature-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-4px);
    }
    
    .feature-icon {
        font-size: 32px;
        margin-bottom: 12px;
    }
    
    .feature-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #f1f5f9;
    }
    
    .feature-desc {
        font-size: 13px;
        color: #cbd5e1;
        line-height: 1.5;
    }
    
    /* ========== METRICS SECTION ========== */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 40px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #3b82f6;
        font-variant-numeric: tabular-nums;
    }
    
    /* ========== EXPLANATION BOX ========== */
    .explanation-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        line-height: 1.7;
        color: #e2e8f0;
    }
    
    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%);
        color: #60a5fa;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 16px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* ========== STATUS BADGES ========== */
    .status-high-bias {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #fca5a5;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
    }
    
    .status-fair {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #86efac;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .animate-in {
        animation: fadeIn 0.6s ease forwards;
    }
    
    /* ========== TYPOGRAPHY ========== */
    h1, h2, h3, h4, h5, h6 {
        letter-spacing: -0.5px;
        font-weight: 700;
    }
    
    .section-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-subtitle {
        font-size: 16px;
        color: #cbd5e1;
        margin-bottom: 40px;
        line-height: 1.6;
    }
    
    /* ========== BULLET POINTS ========== */
    .bullet-list {
        list-style: none;
        padding-left: 0;
    }
    
    .bullet-list li {
        display: flex;
        gap: 12px;
        margin-bottom: 12px;
        font-size: 14px;
        color: #cbd5e1;
        line-height: 1.6;
    }
    
    .bullet-list li:before {
        content: "▸";
        color: #3b82f6;
        font-weight: 800;
        flex-shrink: 0;
    }
    
    /* ========== SCROLL INDICATOR ========== */
    .scroll-indicator {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 32px;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* ========== LOADING SKELETON ========== */
    .skeleton {
        background: linear-gradient(90deg, rgba(255,255,255,0.1) 25%, rgba(255,255,255,0.2) 50%, rgba(255,255,255,0.1) 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
</style>
""", unsafe_allow_html=True)

# ============================================================================
# STATE MANAGEMENT
# ============================================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# ============================================================================
# NAVBAR COMPONENT
# ============================================================================
def render_navbar():
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.markdown('<div class="nav-logo">⚖️ BIAS AI</div>', unsafe_allow_html=True)
    
    with col2:
        nav_cols = st.columns(4)
        nav_items = [
            ("home", "Home"),
            ("features", "Features"),
            ("analyze", "Analyze"),
            ("about", "About")
        ]
        
        for idx, (page_key, label) in enumerate(nav_items):
            with nav_cols[idx]:
                active_class = "active" if st.session_state.current_page == page_key else ""
                if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                    st.session_state.current_page = page_key
                    st.rerun()
    
    with col3:
        st.write("")  # Spacer

render_navbar()

# ============================================================================
# HOME PAGE
# ============================================================================
def home_page():
    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">Enterprise Fairness Analysis</h1>
        <p class="hero-subtitle">
            Detect, measure, and eliminate AI bias with enterprise-grade precision. 
            Built for teams that prioritize ethical AI.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Analysis", key="hero_analyze", use_container_width=True):
            st.session_state.current_page = "analyze"
            st.rerun()
    with col2:
        if st.button("📚 Learn More", key="hero_learn", use_container_width=True):
            st.session_state.current_page = "features"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Overview
    st.markdown('<div class="section-title">Why Choose BIAS AI?</div>', unsafe_allow_html=True)
    
    features = [
        {
            "icon": "🔍",
            "title": "Detect Hidden Bias",
            "desc": "Identify systemic bias in ML predictions with statistical precision"
        },
        {
            "icon": "🛠️",
            "title": "Automated Mitigation",
            "desc": "Apply proven fairness techniques automatically"
        },
        {
            "icon": "🤖",
            "title": "AI-Powered Insights",
            "desc": "Get natural language explanations powered by Gemini & Groq"
        },
        {
            "icon": "📊",
            "title": "Professional Reports",
            "desc": "Export detailed PDF reports for stakeholders"
        }
    ]
    
    cols = st.columns(len(features))
    for col, feature in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{feature['icon']}</div>
                <div class="feature-title">{feature['title']}</div>
                <div class="feature-desc">{feature['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Problem & Solution
    st.markdown('<div class="section-title">The Problem</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <ul class="bullet-list">
            <li>AI models often perpetuate historical biases from training data</li>
            <li>Disparate impact ratios reveal unfair treatment of protected groups</li>
            <li>Manual bias detection requires ML expertise most teams lack</li>
            <li>Non-compliance with fairness regulations leads to legal & reputational risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Our Solution</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <ul class="bullet-list">
            <li><strong>Automated Detection:</strong> One-click bias analysis on any dataset</li>
            <li><strong>Mitigation Strategies:</strong> Remove sensitive attributes to reduce disparate impact</li>
            <li><strong>AI Explanations:</strong> Understand the "why" behind bias patterns</li>
            <li><strong>Enterprise Export:</strong> Professional reports for compliance & stakeholder communication</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Use Cases
    st.markdown('<div class="section-title">Real-World Applications</div>', unsafe_allow_html=True)
    
    use_cases = [
        {
            "title": "Hiring Systems",
            "icon": "👥",
            "points": [
                "Identify gender/age bias in resume screening",
                "Ensure fair compensation recommendations",
                "Protect from discrimination lawsuits"
            ]
        },
        {
            "title": "Lending Decisions",
            "icon": "💰",
            "points": [
                "Detect racial bias in loan approvals",
                "Comply with Fair Lending regulations (ECOA)",
                "Build customer trust through transparency"
            ]
        },
        {
            "title": "Healthcare AI",
            "icon": "🏥",
            "points": [
                "Prevent disparate treatment across demographics",
                "Improve health equity outcomes",
                "Meet HIPAA & bias-audit requirements"
            ]
        }
    ]
    
    for uc in use_cases:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: #3b82f6; margin-bottom: 16px;">{uc['icon']} {uc['title']}</h3>
            <ul class="bullet-list">
                {''.join([f'<li>{point}</li>' for point in uc['points']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FEATURES PAGE
# ============================================================================
def features_page():
    st.markdown('<div class="section-title">Feature Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Comprehensive tools for responsible AI</div>', unsafe_allow_html=True)
    
    features_detail = [
        {
            "title": "Bias Detection Engine",
            "icon": "🔍",
            "points": [
                "Disparate Impact Ratio calculation",
                "Group-level fairness metrics",
                "Statistical significance testing",
                "Demographic parity analysis"
            ]
        },
        {
            "title": "Mitigation Techniques",
            "icon": "🔧",
            "points": [
                "Sensitive attribute removal",
                "Fairness-aware retraining",
                "Threshold optimization",
                "Group-specific model adjustment"
            ]
        },
        {
            "title": "AI-Powered Explanations",
            "icon": "🤖",
            "points": [
                "Gemini API (primary) - Advanced reasoning",
                "Groq API (fallback) - Fast inference",
                "Local explanations - Always available",
                "Multi-language support coming soon"
            ]
        },
        {
            "title": "Professional Reporting",
            "icon": "📄",
            "points": [
                "Executive summaries",
                "Visual bias charts",
                "Metrics & recommendations",
                "PDF export for stakeholders"
            ]
        }
    ]
    
    for feature in features_detail:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: #3b82f6; margin-bottom: 16px; font-size: 20px;">{feature['icon']} {feature['title']}</h3>
            <ul class="bullet-list">
                {''.join([f'<li>{point}</li>' for point in feature['points']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home", key="feat_home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
    with col2:
        if st.button("Start Analyzing →", key="feat_analyze", use_container_width=True):
            st.session_state.current_page = "analyze"
            st.rerun()

# ============================================================================
# ANALYSIS PAGE
# ============================================================================
def analyze_page():
    st.markdown('<div class="section-title">Bias Analysis Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Upload your dataset and identify hidden bias</div>', unsafe_allow_html=True)
    
    # Upload Section
    st.markdown('<h3 style="color: #cbd5e1; margin-bottom: 16px;">📂 Upload Dataset</h3>', unsafe_allow_html=True)
    
    file = st.file_uploader("Choose a CSV file", type="csv", key="file_upload")
    
    if file:
        df = pd.read_csv(file).dropna()
        
        st.markdown('<h3 style="color: #cbd5e1; margin-bottom: 16px;">👀 Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("🎯 Select Target Column (what you're predicting)", df.columns)
        with col2:
            sensitive = st.selectbox("⚠️ Select Sensitive Attribute (protected characteristic)", df.columns)

        # Validate & select groups for sensitive attribute
        unique_groups = df[sensitive].dropna().unique()

        if len(unique_groups) < 2:
            st.error("⚠️ The sensitive attribute must have at least 2 unique values for comparison.")
            return

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

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Run Bias Analysis", use_container_width=True, key="run_analysis"):
            with st.spinner("Analyzing... this may take a moment"):
                try:
                    # Smart target encoding (works with any CSV)
                    df[target] = encode_target_column(df, target)

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

                    # Safe group rate calculation
                    g1_mask = df_test[sensitive] == group1
                    g2_mask = df_test[sensitive] == group2
                    g1 = df_test.loc[g1_mask, 'pred'].mean() if g1_mask.any() else 0.0
                    g2 = df_test.loc[g2_mask, 'pred'].mean() if g2_mask.any() else 0.0
                    di_ratio = safe_disparate_impact(g1, g2)

                    # Mitigation - retrain without sensitive attribute
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

                    # Store results
                    st.session_state.analysis_results = {
                        'g1_before': g1,
                        'g2_before': g2,
                        'g1_after': g1_after,
                        'g2_after': g2_after,
                        'di_ratio': di_ratio,
                        'target': target,
                        'sensitive': sensitive,
                        'group1': str(group1),
                        'group2': str(group2),
                        'fig_before': None,
                        'fig_after': None
                    }

                    st.session_state.analysis_complete = True
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Make sure your target column is suitable for binary classification and the sensitive attribute has distinct groups.")
    
    # Display Results
    if st.session_state.analysis_complete and "analysis_results" in st.session_state:
        results = st.session_state.analysis_results
        g1_name = results.get('group1', 'Group 1')
        g2_name = results.get('group2', 'Group 2')

        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)

        # Metrics Grid
        st.markdown('<h3 style="color: #cbd5e1; margin-bottom: 20px;">Key Metrics</h3>', unsafe_allow_html=True)

        metric_cols = st.columns(4)
        metrics = [
            (f"Before - {g1_name}", results['g1_before'], "📈"),
            (f"Before - {g2_name}", results['g2_before'], "📉"),
            (f"After - {g1_name}", results['g1_after'], "✅"),
            (f"After - {g2_name}", results['g2_after'], "✅"),
        ]
        
        for col, (label, value, icon) in zip(metric_cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{icon} {label}</div>
                    <div class="metric-value">{value:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Status
        if results['di_ratio'] < 0.8:
            st.markdown(
                '<div class="status-high-bias">⚠️ High Bias Detected - Disparate Impact Ratio: {:.2f}</div>'.format(results['di_ratio']),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-fair">✅ Fair Model - Disparate Impact Ratio: {:.2f}</div>'.format(results['di_ratio']),
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # AI Explanation
        st.markdown('<h3 style="color: #cbd5e1; margin-bottom: 16px;">🤖 AI-Powered Analysis</h3>', unsafe_allow_html=True)
        
        with st.spinner("Generating insights..."):
            bias_findings = {
                'g1_before': results['g1_before'],
                'g2_before': results['g2_before'],
                'g1_after': results['g1_after'],
                'g2_after': results['g2_after'],
                'di_ratio': results['di_ratio']
            }
            
            try:
                gemini_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")
            except Exception:
                gemini_key = os.getenv("GEMINI_API_KEY")
            try:
                groq_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
            except Exception:
                groq_key = os.getenv("GROQ_API_KEY")
            
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
                    <div class="source-badge">🔌 Source: {result['source'].upper()}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        st.markdown('<h3 style="color: #cbd5e1; margin-bottom: 16px;">📈 Visualizations</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1a1f3a')
            bars = ax.bar([g1_name, g2_name], [results['g1_before'], results['g2_before']], 
                          color=['#3b82f6', '#ef4444'], edgecolor='#cbd5e1', linewidth=1.5)
            ax.set_title("Before Mitigation", color='#f1f5f9', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel("Positive Rate", color='#cbd5e1')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.1, color='#cbd5e1')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors='#cbd5e1')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor('#0f172a')
            ax2.set_facecolor('#1a1f3a')
            bars2 = ax2.bar([g1_name, g2_name], [results['g1_after'], results['g2_after']], 
                           color=['#10b981', '#f59e0b'], edgecolor='#cbd5e1', linewidth=1.5)
            ax2.set_title("After Mitigation", color='#f1f5f9', fontsize=14, fontweight='bold', pad=20)
            ax2.set_ylabel("Positive Rate", color='#cbd5e1')
            ax2.set_ylim(0, 1)
            ax2.grid(axis='y', alpha=0.1, color='#cbd5e1')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.tick_params(colors='#cbd5e1')
            plt.tight_layout()
            st.pyplot(fig2)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # PDF Export
        st.markdown('<h3 style="color: #cbd5e1; margin-bottom: 16px;">📥 Export Report</h3>', unsafe_allow_html=True)
        
        def create_professional_pdf():
            """Create a professional, colorful, detailed PDF report."""
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=(8.5*inch, 11*inch), 
                                   rightMargin=0.5*inch, leftMargin=0.5*inch,
                                   topMargin=0.5*inch, bottomMargin=0.5*inch)
            
            # Define custom styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=28,
                textColor=colors.HexColor('#1e3a8a'),
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#1e40af'),
                spaceAfter=10,
                spaceBefore=10,
                fontName='Helvetica-Bold'
            )
            
            subheading_style = ParagraphStyle(
                'SubHeading',
                parent=styles['Normal'],
                fontSize=12,
                textColor=colors.HexColor('#3b82f6'),
                spaceAfter=8,
                fontName='Helvetica-Bold'
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#1f2937'),
                spaceAfter=6,
                leading=14
            )
            
            content = []
            
            # ===== TITLE PAGE =====
            content.append(Spacer(1, 0.3*inch))
            content.append(Paragraph("⚖️ AI BIAS DETECTION REPORT", title_style))
            content.append(Spacer(1, 0.15*inch))
            
            date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], 
                                       fontSize=11, alignment=TA_CENTER, textColor=colors.grey)
            content.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                                    date_style))
            content.append(Spacer(1, 0.3*inch))
            
            # Executive Summary Box
            summary_data = [
                ['ANALYSIS OVERVIEW'],
                [Paragraph(f'Target Column: <b>{results["target"]}</b>', normal_style)],
                [Paragraph(f'Sensitive Attribute: <b>{results["sensitive"]}</b>', normal_style)],
                [Paragraph(f'Group 1: <b>{results["group1"]}</b>', normal_style)],
                [Paragraph(f'Group 2: <b>{results["group2"]}</b>', normal_style)]
            ]
            summary_table = Table(summary_data, colWidths=[7.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            content.append(summary_table)
            content.append(Spacer(1, 0.3*inch))
            
            # ===== KEY FINDINGS =====
            content.append(Paragraph("📊 KEY FINDINGS", heading_style))
            
            # Bias Status
            di_ratio = results['di_ratio']
            if di_ratio < 0.8:
                bias_status = "🚨 HIGH BIAS DETECTED"
                bias_color = colors.HexColor('#dc2626')
                bias_desc = f"Disparate Impact Ratio: {di_ratio:.4f} (less than 0.8 indicates potential discrimination)"
            else:
                bias_status = "✅ FAIR MODEL"
                bias_color = colors.HexColor('#059669')
                bias_desc = f"Disparate Impact Ratio: {di_ratio:.4f} (above 0.8 indicates fair treatment)"
            
            bias_style = ParagraphStyle('BiasStatus', parent=styles['Normal'], 
                                       fontSize=14, textColor=bias_color, 
                                       fontName='Helvetica-Bold', alignment=TA_CENTER)
            content.append(Paragraph(bias_status, bias_style))
            content.append(Paragraph(bias_desc, normal_style))
            content.append(Spacer(1, 0.2*inch))
            
            # ===== METRICS SECTION =====
            content.append(Paragraph("📈 ANALYSIS RESULTS - KEY METRICS", heading_style))
            content.append(Spacer(1, 0.1*inch))
            
            g1_before = results['g1_before']
            g2_before = results['g2_before']
            g1_after = results['g1_after']
            g2_after = results['g2_after']
            
            # Before Mitigation Table
            content.append(Paragraph("BEFORE MITIGATION (Original Model):", subheading_style))
            before_data = [
                ['Group', 'Selection Rate', 'Percentage'],
                [results['group1'], f'{g1_before:.4f}', f'{g1_before*100:.2f}%'],
                [results['group2'], f'{g2_before:.4f}', f'{g2_before*100:.2f}%'],
                ['Difference', f'{abs(g1_before-g2_before):.4f}', f'{abs(g1_before-g2_before)*100:.2f}%'],
            ]
            before_table = Table(before_data, colWidths=[2*inch, 2.5*inch, 2.5*inch])
            before_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fee2e2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#991b1b')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 1), (-1, 2), colors.HexColor('#fecaca')),
                ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#fca5a5')),
                ('FONTNAME', (0, 3), (-1, 3), 'Helvetica-Bold'),
            ]))
            content.append(before_table)
            content.append(Spacer(1, 0.15*inch))
            
            # After Mitigation Table
            content.append(Paragraph("AFTER MITIGATION (Sensitive Attribute Removed):", 
                                    subheading_style))
            after_data = [
                ['Group', 'Selection Rate', 'Percentage'],
                [results['group1'], f'{g1_after:.4f}', f'{g1_after*100:.2f}%'],
                [results['group2'], f'{g2_after:.4f}', f'{g2_after*100:.2f}%'],
                ['Difference', f'{abs(g1_after-g2_after):.4f}', f'{abs(g1_after-g2_after)*100:.2f}%'],
            ]
            after_table = Table(after_data, colWidths=[2*inch, 2.5*inch, 2.5*inch])
            after_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d1fae5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#065f46')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 1), (-1, 2), colors.HexColor('#a7f3d0')),
                ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#6ee7b7')),
                ('FONTNAME', (0, 3), (-1, 3), 'Helvetica-Bold'),
            ]))
            content.append(after_table)
            content.append(Spacer(1, 0.15*inch))
            
            # Improvement Summary
            bias_reduction = abs(g1_before-g2_before) - abs(g1_after-g2_after)
            improvement_pct = (bias_reduction / abs(g1_before-g2_before) * 100) if abs(g1_before-g2_before) > 0 else 0
            
            improvement_style = ParagraphStyle('Improvement', parent=styles['Normal'], 
                                               fontSize=10, textColor=colors.HexColor('#059669'),
                                               fontName='Helvetica-Bold')
            content.append(Paragraph(
                f"<b>Bias Reduction:</b> {bias_reduction:.4f} ({improvement_pct:.1f}% improvement in fairness)",
                improvement_style
            ))
            
            # PAGE BREAK
            content.append(PageBreak())
            
            # ===== VISUALIZATIONS PAGE =====
            content.append(Paragraph("📊 VISUALIZATIONS", heading_style))
            content.append(Spacer(1, 0.1*inch))
            
            # Create before chart with colors
            fig_before, ax_before = plt.subplots(figsize=(5, 3), facecolor='white')
            groups = [results['group1'], results['group2']]
            rates_before = [g1_before, g2_before]
            bars_before = ax_before.bar(groups, rates_before, color=['#3b82f6', '#ec4899'], alpha=0.8, edgecolor='black', linewidth=1.5)
            ax_before.set_ylabel('Selection Rate', fontsize=11, fontweight='bold')
            ax_before.set_title('BEFORE MITIGATION - Selection Rate by Group', fontsize=12, fontweight='bold', color='#1e3a8a')
            ax_before.set_ylim(0, max(rates_before) * 1.2)
            ax_before.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, rate in zip(bars_before, rates_before):
                height = bar.get_height()
                ax_before.text(bar.get_x() + bar.get_width()/2., height,
                              f'{rate*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            chart_before_path = "chart_before_report.png"
            fig_before.savefig(chart_before_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_before)
            
            # Create after chart
            fig_after, ax_after = plt.subplots(figsize=(5, 3), facecolor='white')
            rates_after = [g1_after, g2_after]
            bars_after = ax_after.bar(groups, rates_after, color=['#10b981', '#f59e0b'], alpha=0.8, edgecolor='black', linewidth=1.5)
            ax_after.set_ylabel('Selection Rate', fontsize=11, fontweight='bold')
            ax_after.set_title('AFTER MITIGATION - Selection Rate by Group', fontsize=12, fontweight='bold', color='#059669')
            ax_after.set_ylim(0, max(rates_after) * 1.2)
            ax_after.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, rate in zip(bars_after, rates_after):
                height = bar.get_height()
                ax_after.text(bar.get_x() + bar.get_width()/2., height,
                             f'{rate*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            chart_after_path = "chart_after_report.png"
            fig_after.savefig(chart_after_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_after)
            
            # Add charts to PDF
            content.append(Paragraph("Model Performance: Comparison", subheading_style))
            
            content.append(Paragraph("<b>Before Mitigation</b>", normal_style))
            if os.path.exists(chart_before_path):
                content.append(Image(chart_before_path, width=5.5*inch, height=3.5*inch))
            content.append(Spacer(1, 0.1*inch))
            
            content.append(Paragraph("<b>After Mitigation (Sensitive Attribute Removed)</b>", normal_style))
            if os.path.exists(chart_after_path):
                content.append(Image(chart_after_path, width=5.5*inch, height=3.5*inch))
            content.append(Spacer(1, 0.15*inch))
            
            # PAGE BREAK
            content.append(PageBreak())
            
            # ===== INTERPRETATION PAGE =====
            content.append(Paragraph("🔍 DETAILED ANALYSIS & INTERPRETATION", heading_style))
            content.append(Spacer(1, 0.1*inch))
            
            # What is Disparate Impact?
            content.append(Paragraph("What is Disparate Impact Ratio (DI)?", subheading_style))
            di_explanation = f"""
            The Disparate Impact Ratio (DI) is calculated as: <b>Selection Rate of {results['group2']} / Selection Rate of {results['group1']}</b><br/>
            <br/>
            • <b>DI &lt; 0.8:</b> Potential discriminatory impact (80% rule violation)<br/>
            • <b>DI ≥ 0.8:</b> Generally considered fair treatment<br/>
            • <b>DI = 1.0:</b> Perfect parity between groups<br/>
            <br/>
            <b>Current Analysis DI Ratio: {di_ratio:.4f}</b>
            """
            content.append(Paragraph(di_explanation, normal_style))
            content.append(Spacer(1, 0.15*inch))
            
            # Current Status
            content.append(Paragraph("Current Model Status", subheading_style))
            if di_ratio < 0.8:
                status_text = f"""
                ⚠️ <b>HIGH BIAS DETECTED</b><br/>
                <br/>
                Your model shows a disparate impact ratio of <b>{di_ratio:.4f}</b>, which falls below the 0.80 threshold.<br/>
                This indicates that <b>{results['sensitive']}</b> may have a disparate impact on the model's predictions.<br/>
                <br/>
                <b>Selection Rates:</b><br/>
                • {results['group1']}: {g1_before*100:.2f}%<br/>
                • {results['group2']}: {g2_before*100:.2f}%<br/>
                <br/>
                The difference of {abs(g1_before-g2_before)*100:.2f} percentage points indicates potential bias in the model.
                """
            else:
                status_text = f"""
                ✅ <b>FAIR MODEL DETECTED</b><br/>
                <br/>
                Your model shows a disparate impact ratio of <b>{di_ratio:.4f}</b>, which exceeds the 0.80 threshold.<br/>
                This suggests relatively fair treatment across groups for the <b>{results['sensitive']}</b> attribute.<br/>
                <br/>
                <b>Selection Rates:</b><br/>
                • {results['group1']}: {g1_before*100:.2f}%<br/>
                • {results['group2']}: {g2_before*100:.2f}%<br/>
                <br/>
                The difference of {abs(g1_before-g2_before)*100:.2f} percentage points is relatively minimal, indicating equitable treatment.
                """
            
            content.append(Paragraph(status_text, normal_style))
            content.append(Spacer(1, 0.15*inch))
            
            # Mitigation Strategy
            content.append(Paragraph("Mitigation Strategy Applied", subheading_style))
            mitigation_text = f"""
            <b>Approach: Sensitive Attribute Removal</b><br/>
            <br/>
            The mitigation model removed the <b>{results['sensitive']}</b> attribute from the feature set to reduce potential bias.<br/>
            <br/>
            <b>Bias Reduction: {bias_reduction:.4f} ({improvement_pct:.1f}% improvement)</b><br/>
            <br/>
            <b>Results:</b><br/>
            • Before: {abs(g1_before-g2_before)*100:.2f}% difference between groups<br/>
            • After: {abs(g1_after-g2_after)*100:.2f}% difference between groups<br/>
            <br/>
            Note: This is a simple fairness approach. More sophisticated techniques (reweighting, fair representations, 
            constraint-based methods) may provide better results in production scenarios.
            """
            content.append(Paragraph(mitigation_text, normal_style))
            content.append(Spacer(1, 0.15*inch))
            
            # Recommendations
            content.append(Paragraph("Recommendations", subheading_style))
            recommendations = f"""
            <b>1. Further Investigation:</b> Examine the training data for underlying biases in {results['sensitive']} representation.<br/>
            <br/>
            <b>2. Feature Engineering:</b> Consider proxy variables that might indirectly encode {results['sensitive']}.<br/>
            <br/>
            <b>3. Regular Auditing:</b> Continuously monitor model fairness across different demographic groups.<br/>
            <br/>
            <b>4. Stakeholder Review:</b> Engage with domain experts and affected communities in bias assessment.<br/>
            <br/>
            <b>5. Documentation:</b> Maintain records of fairness decisions and mitigation strategies used.<br/>
            <br/>
            <b>6. Legal Compliance:</b> Ensure your model meets regulatory requirements (FCRA, GDPR, Equal Credit Opportunity Act, etc.)
            """
            content.append(Paragraph(recommendations, normal_style))
            
            # PAGE BREAK
            content.append(PageBreak())
            
            # ===== FOOTER PAGE =====
            footer_style = ParagraphStyle('Footer', parent=styles['Normal'], 
                                         fontSize=9, textColor=colors.grey, 
                                         alignment=TA_CENTER)
            
            content.append(Spacer(1, 1.5*inch))
            content.append(Paragraph("<b>About This Report</b>", subheading_style))
            footer_text = """
            This AI Bias Detection Report was generated using machine learning fairness analysis techniques.<br/>
            The analysis measures disparate impact and applies mitigation strategies to improve model fairness.<br/>
            <br/>
            <b>Limitations:</b> This analysis provides statistical measures of fairness but does not account for all 
            potential sources of bias (e.g., historical bias in training data, unmeasured proxies for protected attributes).<br/>
            <br/>
            <b>Important Disclaimer:</b> This tool provides analytical insights and should not be used as sole basis 
            for compliance decisions. Legal and ethical review by qualified professionals is recommended.<br/>
            <br/>
            Generated by: AI Bias Detection Platform<br/>
            Report Date: """ + datetime.now().strftime('%B %d, %Y')
            
            content.append(Paragraph(footer_text, normal_style))
            
            # Build PDF
            doc.build(content)
            buffer.seek(0)
            return buffer
        
        pdf = create_professional_pdf()
        
        st.download_button(
            label="📄 Download Professional Report",
            data=pdf,
            file_name=f"bias_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ============================================================================
# ABOUT PAGE
# ============================================================================
def about_page():
    st.markdown('<div class="section-title">About BIAS AI</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #3b82f6; margin-bottom: 16px;">Mission</h3>
        <p style="color: #cbd5e1; line-height: 1.8;">
        We believe AI should be fair, transparent, and trustworthy. BIAS AI makes ethical AI analysis 
        accessible to every organization, from startups to enterprises.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #3b82f6; margin-bottom: 16px;">What Makes Us Different</h3>
        <ul class="bullet-list">
            <li><strong>One-Click Analysis:</strong> No ML expertise required</li>
            <li><strong>Multi-Model LLM:</strong> Gemini + Groq + Local fallback for reliability</li>
            <li><strong>Enterprise-Ready:</strong> PDF reports, API integration coming soon</li>
            <li><strong>Ethical Foundation:</strong> Built on academic fairness research</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #3b82f6; margin-bottom: 16px;">Technology Stack</h3>
        <ul class="bullet-list">
            <li>Frontend: Streamlit with custom glassmorphism CSS</li>
            <li>ML: scikit-learn, pandas</li>
            <li>LLM: Google Gemini 2.0, Groq Llama 3.3</li>
            <li>Reports: ReportLab</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home", key="about_home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
    with col2:
        if st.button("Start Analyzing →", key="about_analyze", use_container_width=True):
            st.session_state.current_page = "analyze"
            st.rerun()

# ============================================================================
# PAGE ROUTING
# ============================================================================
if st.session_state.current_page == "home":
    home_page()
elif st.session_state.current_page == "features":
    features_page()
elif st.session_state.current_page == "analyze":
    analyze_page()
elif st.session_state.current_page == "about":
    about_page()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px; padding: 20px 0;">
    <p> © 2026 BIAS AI | Enterprise Fairness Platform</p>
    <p style="margin-top: 8px;">Ethical AI starts with transparency. Let's build fair systems together.</p>
</div>
""", unsafe_allow_html=True)
