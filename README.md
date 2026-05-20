<div align="center">

# ⚖️ BIAS AI Detector
### Enterprise Fairness Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**Detect, measure, and eliminate AI bias with enterprise-grade precision.**  
Built for teams that prioritize ethical, fair, and responsible AI.

[🚀 Quick Start](#-quick-start) · [📖 Features](#-features) · [🏗️ Architecture](#️-architecture) · [📊 How It Works](#-how-it-works) · [🤝 Contributing](#-contributing)

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [Supported CSV Formats](#-supported-csv-formats)
- [API Integration](#-api-integration)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

**BIAS AI DETECTOR** is an open-source, enterprise-grade AI fairness analysis platform that empowers data scientists, ML engineers, and compliance teams to detect and mitigate bias in machine learning models — without requiring deep ML expertise.

Upload any CSV dataset, select a target variable and a sensitive attribute (e.g., gender, race, age), and the platform will:

1. Train a logistic regression classifier on your data
2. Measure **Disparate Impact** across demographic groups
3. Apply **bias mitigation** by removing the sensitive attribute and retraining
4. Generate an **AI-powered explanation** of the bias findings
5. Export a **professional PDF report** for stakeholders

> **Who is this for?**  
> HR teams auditing hiring algorithms · Financial institutions checking lending models · Healthcare orgs ensuring equitable AI · ML teams needing compliance reports

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Universal CSV Support** | Works with any CSV — binary, multi-class, numeric, or categorical targets |
| ⚖️ **Disparate Impact Analysis** | Industry-standard 80% rule (4/5ths rule) measurement |
| 🛠️ **Automated Mitigation** | Removes sensitive attributes and retrains for fairer predictions |
| 🤖 **AI-Powered Explanations** | Three-tier LLM fallback: Gemini → Groq → Local |
| 📊 **Interactive Visualizations** | Before/after comparison charts with dark-mode glassmorphism UI |
| 📄 **PDF Report Export** | One-click professional report generation via ReportLab |
| 🎯 **Smart Group Selection** | Auto-detects groups; lets you pick 2 from multi-class attributes |
| 🔒 **Secrets Management** | Secure API key handling via Streamlit secrets |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit · Custom Glassmorphism CSS |
| **ML Pipeline** | scikit-learn (Logistic Regression, StandardScaler) |
| **Data Processing** | pandas |
| **Visualizations** | matplotlib |
| **LLM (Primary)** | Google Gemini 2.0 Flash |
| **LLM (Fallback)** | Groq · Llama 3.3 70B |
| **Report Generation** | ReportLab |
| **Language** | Python 3.9+ |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BIAS AI Platform                     │
├─────────────┬───────────────────┬───────────────────────┤
│  Streamlit  │   ML Pipeline     │   LLM Fallback Chain  │
│  Frontend   │                   │                       │
│  ─────────  │  CSV Upload       │  1. Gemini 2.0 Flash  │
│  Home       │      ↓            │         ↓             │
│  Features   │  Smart Encoding   │  2. Groq Llama 3.3    │
│  Analyze    │      ↓            │         ↓             │
│  About      │  Train Model      │  3. Local Explanation │
│             │      ↓            │                       │
│             │  Bias Detection   │                       │
│             │      ↓            │                       │
│             │  Mitigation       │                       │
│             │      ↓            │                       │
│             │  PDF Report       │                       │
└─────────────┴───────────────────┴───────────────────────┘
```

### LLM Fallback System (`llm_utils.py`)

```
Request
   │
   ▼
Gemini API ──(fail)──► Groq API ──(fail)──► Local Engine
   │                      │                      │
   └──────────────────────┴──────────────────────┘
                          │
                   Explanation + Source Badge
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/BiasAI.git
cd BiasAI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up API keys** *(optional — app works without them)*
```bash
mkdir .streamlit
```
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key"
GROQ_API_KEY   = "your-groq-api-key"
```

**4. Run the app**
```bash
python -m streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## ⚙️ Configuration

### API Keys (Optional)

The app uses a **three-tier fallback system** — API keys are optional:

| Tier | Provider | Free Tier | Get Key |
|---|---|---|---|
| 1 (Primary) | Google Gemini | ✅ Yes | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| 2 (Fallback) | Groq | ✅ Yes | [console.groq.com](https://console.groq.com/keys) |
| 3 (Always available) | Local Engine | ✅ Always | No key needed |

### Secrets File Location

```
BiasAI/
└── .streamlit/
    └── secrets.toml    ← Streamlit reads THIS file
```

> ⚠️ **Important:** Streamlit only reads from `.streamlit/secrets.toml`.  
> A `secrets.toml` at the project root is **not** read by Streamlit.

### Environment Variables (Alternative)

You can also set keys as environment variables:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-key"
$env:GROQ_API_KEY   = "your-key"

# Linux / macOS
export GEMINI_API_KEY="your-key"
export GROQ_API_KEY="your-key"
```

---

## 📊 How It Works

### Step 1 — Upload & Preview
Upload any CSV file. The app automatically drops rows with missing values and previews the first 5 rows.

### Step 2 — Select Columns
- **Target Column**: The outcome your model predicts (e.g., `income`, `hired`, `approved`)
- **Sensitive Attribute**: The protected characteristic to audit (e.g., `sex`, `race`, `age`)

### Step 3 — Smart Target Encoding
The app automatically converts any target column to binary:

| Target Type | Strategy |
|---|---|
| Already 0/1 | Used as-is |
| Exactly 2 unique values | Maps to 0 and 1 |
| Numeric (many values) | Median split (above = 1) |
| Categorical (many values) | Most frequent = 0, rest = 1 |

### Step 4 — Bias Detection
- Trains a Logistic Regression model on 80% of the data
- Measures positive prediction rates for each group
- Calculates **Disparate Impact Ratio** = Group 2 rate ÷ Group 1 rate

> **Interpretation:**
> - `< 0.8` → ⚠️ High Bias (fails the 80% rule)
> - `0.8 – 1.25` → ✅ Fair
> - `> 1.25` → ⚠️ Reverse bias

### Step 5 — Mitigation
Retrains the model with the sensitive attribute removed.  
Compares before/after rates to quantify improvement.

### Step 6 — AI Explanation + Report
Sends metrics to Gemini/Groq for a natural-language explanation.  
Generates a downloadable PDF report with metrics, charts, and recommendations.

---

## 📂 Supported CSV Formats

The platform handles virtually any tabular CSV:

```
✅ UCI Adult Income Dataset     (adult.csv)
✅ COMPAS Recidivism Dataset
✅ German Credit Dataset
✅ Any custom HR/lending/healthcare CSV
✅ Binary classification datasets
✅ Multi-class datasets (auto-encoded)
✅ Numeric regression targets (median-split)
```

**Minimum requirements:**
- At least 2 columns
- At least 50 rows (recommended: 500+)
- A column suitable as a classification target
- A column with at least 2 distinct groups as sensitive attribute

---

## 🔌 API Integration

### `llm_utils.py` — Public Interface

```python
from llm_utils import get_bias_explanation

result = get_bias_explanation(
    bias_findings={
        'g1_before': 0.45,
        'g2_before': 0.28,
        'g1_after':  0.40,
        'g2_after':  0.35,
        'di_ratio':  0.62
    },
    gemini_key="your-key",   # optional
    groq_key="your-key",     # optional
    use_fallback=True         # always falls back to local
)

# Returns:
# {
#   'explanation': "The model exhibits significant bias...",
#   'source': 'gemini' | 'groq' | 'local',
#   'success': True
# }
```

---

## 📁 Project Structure

```
BiasAI/
├── app.py                  # Main Streamlit app (Premium UI)
├── work.py                 # Alternative Streamlit app (Sidebar UI)
├── llm_utils.py            # Three-tier LLM fallback system
├── requirements.txt        # Python dependencies
├── adult.csv               # Sample dataset (UCI Adult Income)
├── .streamlit/
│   └── secrets.toml        # API keys (not committed to git)
├── chart_before.png        # Sample output chart
├── chart_after.png         # Sample output chart
└── README.md               # This file
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

### Ideas for Contributions
- [ ] Additional fairness metrics (Equalized Odds, Calibration)
- [ ] Support for XGBoost / Random Forest models
- [ ] Multi-attribute intersectional bias analysis
- [ ] REST API endpoint for programmatic access
- [ ] Docker / deployment support
- [ ] Unit test suite

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool is intended to **assist** fairness audits, not replace them. Results should be interpreted by domain experts alongside legal and ethical review. No automated tool can fully substitute for comprehensive human oversight in high-stakes AI systems.

---

<div align="center">

**Built with ❤️ for ethical AI**

⭐ Star this repo if you find it useful!

</div>
