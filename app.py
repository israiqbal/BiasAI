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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Import LLM utilities
from llm_utils import get_bias_explanation

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
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Run Bias Analysis", use_container_width=True, key="run_analysis"):
            with st.spinner("Analyzing... this may take a moment"):
                
                # Data Processing
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
                
                # Mitigation
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
                
                # Store results
                st.session_state.analysis_results = {
                    'g1_before': g1,
                    'g2_before': g2,
                    'g1_after': g1_after,
                    'g2_after': g2_after,
                    'di_ratio': di_ratio,
                    'target': target,
                    'sensitive': sensitive,
                    'fig_before': None,
                    'fig_after': None
                }
                
                st.session_state.analysis_complete = True
                st.rerun()
    
    # Display Results
    if st.session_state.analysis_complete and "analysis_results" in st.session_state:
        results = st.session_state.analysis_results
        
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        # Metrics Grid
        st.markdown('<h3 style="color: #cbd5e1; margin-bottom: 20px;">Key Metrics</h3>', unsafe_allow_html=True)
        
        metric_cols = st.columns(4)
        metrics = [
            ("Before - Group 1", results['g1_before'], "📈"),
            ("Before - Group 2", results['g2_before'], "📉"),
            ("After - Group 1", results['g1_after'], "✅"),
            ("After - Group 2", results['g2_after'], "✅"),
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
            bars = ax.bar(['Group 1', 'Group 2'], [results['g1_before'], results['g2_before']], 
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
            bars2 = ax2.bar(['Group 1', 'Group 2'], [results['g1_after'], results['g2_after']], 
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
        
        def create_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()
            
            content = []
            
            content.append(Paragraph("AI Bias Detection Report", styles['Title']))
            content.append(Spacer(1, 12))
            
            content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            content.append(Paragraph(f"Target Column: {results['target']}", styles['Normal']))
            content.append(Paragraph(f"Sensitive Attribute: {results['sensitive']}", styles['Normal']))
            content.append(Spacer(1, 12))
            
            content.append(Paragraph("Before Mitigation", styles['Heading2']))
            content.append(Paragraph(f"Group 1 Rate: {results['g1_before']:.2%}", styles['Normal']))
            content.append(Paragraph(f"Group 2 Rate: {results['g2_before']:.2%}", styles['Normal']))
            content.append(Paragraph(f"Disparate Impact Ratio: {results['di_ratio']:.2f}", styles['Normal']))
            content.append(Spacer(1, 12))
            
            content.append(Paragraph("After Mitigation", styles['Heading2']))
            content.append(Paragraph(f"Group 1 Rate: {results['g1_after']:.2%}", styles['Normal']))
            content.append(Paragraph(f"Group 2 Rate: {results['g2_after']:.2%}", styles['Normal']))
            
            doc.build(content)
            buffer.seek(0)
            return buffer
        
        pdf = create_pdf()
        
        st.download_button(
            label="📄 Download Premium Report (PDF)",
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
    <p>Built with ❤️ | © 2026 BIAS AI | Enterprise Fairness Platform</p>
    <p style="margin-top: 8px;">Ethical AI starts with transparency. Let's build fair systems together.</p>
</div>
""", unsafe_allow_html=True)
