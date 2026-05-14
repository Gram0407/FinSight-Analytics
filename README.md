# FinSight Analytics — Full-Stack Financial Analytics Platform

**MBA Banking Technology · Pondicherry University**
*Submitted by: IYYANAR G (24MBAFTPY0049) · Guide: Dr. A. Balakrishnan*

---

## 🚀 Quick Start

```bash
cd finsight
pip install -r requirements.txt
python app.py
# Open: http://127.0.0.1:5000
```

---

## 🎯 Core Features

### 1. Live Stock Dashboard
- 50+ NSE/BSE stocks with real-time style table
- Green/red color coding for price movements
- Columns: Symbol, Live Price, Change, Change %, Today H/L, 52W H/L, Sector, MCap, Term, Prev Close
- Search & sector filter
- CSV watchlist integration (MC_WATCHLIST_DATA)
- **Predict button** → auto-fills Manual Prediction tab

### 2. Macro Indicators Tab
- GDP Growth, Inflation, Interest Rate, Exchange Rate, M3, IIP charts
- Panel model equation display
- Expected impact table (β signs with economic rationale)

### 3. Regression Analysis Tab
**Multi-Stock (Panel Data):**
- Fixed Effects (Within Estimator)
- Random Effects (GLS)
- Hausman Test → model selection
- Model: `Y_it = α + β₁GDP + β₂Inflation + β₃Interest + β₄Exchange + β₅M3 + β₆IIP + μᵢ + εᵢₜ`

**Single-Stock (OLS):**
- Full coefficient table with t-stats and p-values
- R², Adj R², F-statistic, AIC, BIC

**Diagnostic Tests:**
- ADF (Stationarity)
- VIF (Multicollinearity)
- Breusch-Pagan (Heteroskedasticity)
- Durbin-Watson (Autocorrelation)

### 4. Manual Prediction (Panel Regression)
- Inputs: Exchange, Stock (entity i), Year (time t), Stock Price
- Macro selection: All or custom subset (GDP, Inflation, Interest, Exchange, M3, IIP)
- Optional custom macro values
- Runs Fixed Effects panel model trained on 20+ stocks
- Output: Predicted Price, Trend (UP/DOWN), Confidence, β coefficients, Impact Analysis
- **Saved to History automatically**

### 5. Prediction History
- Full audit trail of all predictions
- Date, Stock, Input Price, Predicted Price, Trend, Confidence, Year
- Delete records | Export CSV

### 6. AI Assistant (Claude-Powered)
- Natural language Q&A on stocks & macroeconomics
- Quick-predict panel with AI suggestion (BUY/HOLD/CAUTION)
- Quick query buttons for common analysis
- Fallback rule-based engine if API unavailable

### 7. Dark/Light Theme
- Toggle stored in database per user
- Applies globally across all pages

---

## 📁 Project Structure

```
finsight/
├── app.py              # Flask routes & API endpoints
├── model.py            # Panel regression, OLS, diagnostics, manual predict
├── ai_assistant.py     # Claude API integration + fallback
├── data_fetch.py       # Stock universe, macro data, CSV integration
├── requirements.txt
├── setup.py
├── dataset/
│   └── MC_WATCHLIST_DATA__1_.csv
└── templates/
    ├── dashboard.html  # Main SPA dashboard
    ├── login.html
    ├── register.html
    └── profile.html
```

---

## 🔬 Research Alignment

**Research Topic:** Analyzing the Impact of Macroeconomic Variables on Stock Price Movements using Panel Data Regression: Evidence from Indian Stock Market

**Model:** `Y_it = α + β₁GDP_t + β₂Inflation_t + β₃Interest_t + β₄Exchange_t + β₅M3_t + β₆IIP_t + μᵢ + εᵢₜ`

| Variable | Expected β | Rationale |
|----------|-----------|-----------|
| GDP Growth | + | Higher growth → better earnings |
| Inflation | − | Erodes real returns |
| Interest Rate | − | Higher cost of capital |
| Exchange Rate | ± | Mixed sector effects |
| M3 (Money Supply) | + | Excess liquidity → equity inflows |
| IIP | + | Industrial output → corporate profits |

⚠️ *For educational and research purposes only. Not financial advice.*
