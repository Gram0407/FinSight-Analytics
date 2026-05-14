"""
ai_assistant.py — AI Assistant for FinSight Analytics
Uses Anthropic Claude API for intelligent stock analysis and Q&A.
"""
import json, requests, numpy as np
from data_fetch import _generate_price_series, MACRO_DATA, SECTOR_MAP, MACRO_VARS

SECTOR_SENSITIVITIES = {
    "Technology":{"GDP_Growth":"high_positive","Inflation":"moderate_negative","Interest_Rate":"moderate_negative","Exchange_Rate":"high_positive","M3_Growth":"moderate_positive","IIP":"moderate_positive"},
    "Banking":{"GDP_Growth":"moderate_positive","Inflation":"moderate_negative","Interest_Rate":"high_positive","Exchange_Rate":"low_negative","M3_Growth":"high_positive","IIP":"low_positive"},
    "Energy":{"GDP_Growth":"moderate_positive","Inflation":"high_positive","Interest_Rate":"low_negative","Exchange_Rate":"high_negative","M3_Growth":"low_positive","IIP":"moderate_positive"},
    "FMCG":{"GDP_Growth":"low_positive","Inflation":"high_negative","Interest_Rate":"low_negative","Exchange_Rate":"low_negative","M3_Growth":"moderate_positive","IIP":"low_positive"},
    "Auto":{"GDP_Growth":"high_positive","Inflation":"moderate_negative","Interest_Rate":"high_negative","Exchange_Rate":"moderate_negative","M3_Growth":"moderate_positive","IIP":"high_positive"},
    "Pharma":{"GDP_Growth":"low_positive","Inflation":"low_negative","Interest_Rate":"low_negative","Exchange_Rate":"high_positive","M3_Growth":"low_positive","IIP":"moderate_positive"},
    "Infrastructure":{"GDP_Growth":"high_positive","Inflation":"moderate_negative","Interest_Rate":"high_negative","Exchange_Rate":"low_negative","M3_Growth":"moderate_positive","IIP":"high_positive"},
    "Finance":{"GDP_Growth":"high_positive","Inflation":"moderate_negative","Interest_Rate":"high_positive","Exchange_Rate":"low_negative","M3_Growth":"high_positive","IIP":"moderate_positive"},
    "Metals":{"GDP_Growth":"moderate_positive","Inflation":"high_positive","Interest_Rate":"low_negative","Exchange_Rate":"high_negative","M3_Growth":"low_positive","IIP":"high_positive"},
}

def get_ai_response(query, context=None):
    """Call Anthropic Claude API for intelligent financial analysis."""
    if context is None: context = {}
    
    # Build context string
    macro_summary = ""
    latest_macro = MACRO_DATA.get(2024, {})
    macro_summary = f"""Current Indian Macroeconomic Context (2024):
- GDP Growth: {latest_macro.get('GDP_Growth', 8.2)}%
- Inflation (CPI): {latest_macro.get('Inflation', 4.7)}%
- Repo Rate: {latest_macro.get('Interest_Rate', 6.5)}%
- Exchange Rate: ₹{latest_macro.get('Exchange_Rate', 84.0)}/USD
- M3 Money Supply Growth: {latest_macro.get('M3_Growth', 9.2)}%
- IIP (Industrial Production): {latest_macro.get('IIP', 4.2)}%"""
    
    stock_context = ""
    if context.get('symbol'):
        sym = context['symbol']
        sector = SECTOR_MAP.get(sym, 'Unknown')
        stock_context = f"\nAnalyzing Stock: {sym} (Sector: {sector})\nCurrent Price: ₹{context.get('price', 'N/A')}"
    
    system_prompt = f"""You are FinSight AI — an expert financial analytics assistant specializing in Indian stock markets and macroeconomic analysis. You are part of the FinSight Analytics platform built for MBA Banking Technology research at Pondicherry University.

Your expertise covers:
- Panel Data Regression (Fixed Effects, Random Effects, Hausman Test)
- Indian Stock Market (NSE/BSE) analysis
- Macroeconomic variable impacts on stock prices
- Model: Y_it = α + β1·GDP + β2·Inflation + β3·Interest + β4·Exchange + β5·M3 + μ_i + ε_it

{macro_summary}
{stock_context}

Response format for stock predictions:
📊 **Trend:** [UP/DOWN/NEUTRAL]
💡 **Suggestion:** [BUY/HOLD/CAUTION]
🎯 **Confidence:** [HIGH/MODERATE/LOW]
📝 **Reason:** [Brief econometric explanation]
⚠️ *Disclaimer: For educational/research purposes only. Not financial advice.*

Be concise, professional, and data-driven. Always reference macroeconomic indicators in your analysis."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": query}]
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            text = "".join(block.get("text","") for block in data.get("content",[]) if block.get("type")=="text")
            return {"success": True, "response": text, "source": "claude-api"}
        else:
            return _fallback_response(query, context)
    except Exception as e:
        return _fallback_response(query, context)

def _fallback_response(query, context):
    """Fallback rule-based response when API unavailable."""
    q = query.lower()
    symbol = context.get('symbol','')
    sector = SECTOR_MAP.get(symbol,'Technology')
    latest = MACRO_DATA[2024]
    
    if any(w in q for w in ['predict','forecast','price','movement','trend']):
        pred = predict_stock_movement(symbol) if symbol else None
        if pred:
            return {"success":True,"response":f"""📊 **Trend:** {pred['trend']}
💡 **Suggestion:** {pred['suggestion']}
🎯 **Confidence:** {pred['confidence']}
📝 **Reason:** Based on panel regression analysis:
{chr(10).join(['• '+r for r in pred['macro_reasons'][:3]])}
⚠️ *Educational purposes only. Not financial advice.*""","source":"fallback"}
    
    if any(w in q for w in ['gdp','inflation','interest','macro','economy']):
        return {"success":True,"response":f"""📈 **India Macro Snapshot (2024):**
• GDP Growth: {latest['GDP_Growth']}% — Strong expansion phase
• Inflation: {latest['Inflation']}% — Within RBI target band (2–6%)
• Repo Rate: {latest['Interest_Rate']}% — Neutral monetary stance
• Exchange Rate: ₹{latest['Exchange_Rate']}/USD — Managed depreciation
• M3 Growth: {latest['M3_Growth']}% — Adequate liquidity
• IIP: {latest['IIP']}% — Moderate industrial output
📊 **Panel Regression Insight:** GDP (β>0) and M3 (β>0) positively impact stock prices; Inflation (β<0) and Interest Rate (β<0) have negative effects.""","source":"fallback"}
    
    if any(w in q for w in ['hausman','fixed','random','panel','regression']):
        return {"success":True,"response":"""📐 **Panel Data Regression — Model Guide:**

**Fixed Effects (FE):** Controls for unobserved firm-specific factors. Uses within-entity variation.
→ *Best when entity effects correlate with regressors.*

**Random Effects (RE):** Assumes entity effects are uncorrelated with regressors. More efficient.
→ *Best for generalization across the panel.*

**Hausman Test:** H₀: RE is consistent
• If p < 0.05 → Fixed Effects preferred
• If p ≥ 0.05 → Random Effects preferred

**Model:** Y_it = α + β₁GDP + β₂Inflation + β₃Interest + β₄Exchange + β₅M3 + β₆IIP + μᵢ + εᵢₜ""","source":"fallback"}
    
    return {"success":True,"response":f"""Welcome to **FinSight AI** 🤖

I can help you with:
• 📊 Stock price predictions (ask: *"Predict TCS movement"*)
• 🌐 Macroeconomic analysis (ask: *"Explain GDP impact on banks"*)
• 📐 Econometric models (ask: *"Explain Fixed vs Random Effects"*)
• 💡 Investment insights (ask: *"Should I buy pharma stocks?"*)
• 🔍 Sector analysis (ask: *"How does inflation affect FMCG?"*)

Current macro environment: GDP {latest['GDP_Growth']}% | Inflation {latest['Inflation']}% | Repo {latest['Interest_Rate']}%""","source":"fallback"}

def predict_stock_movement(symbol):
    if not symbol: return None
    try:
        df = _generate_price_series(symbol).dropna(subset=['Close']+MACRO_VARS)
        y = np.log(df['Close'].values); X_raw = df[MACRO_VARS].values
        n = len(y); X = np.column_stack([np.ones(n),X_raw])
        betas = np.linalg.pinv(X.T@X)@X.T@y
        y_hat = X@betas; resid = y-y_hat
        r2 = 1-(resid**2).sum()/max(((y-y.mean())**2).sum(),1e-10)
        latest_macro = list(MACRO_DATA[2024].values())
        X_pred = np.array([1.0]+latest_macro)
        pred_log = float(X_pred@betas); pred_price = np.exp(pred_log)
        current_price = float(df['Close'].iloc[-1])
        expected_change = (pred_price-current_price)/current_price*100
        momentum = (df['Close'].tail(6).iloc[-1]/df['Close'].tail(6).iloc[0]-1)*100
        trend = "UP 📈" if expected_change>0 else "DOWN 📉"
        mom_label = "Bullish 🟢" if momentum>0 else "Bearish 🔴"
        abs_change = abs(expected_change)
        confidence = "HIGH" if abs_change>5 and r2>0.7 else "MODERATE" if abs_change>2 and r2>0.5 else "LOW"
        if expected_change>3 and momentum>0: suggestion,color = "BUY","green"
        elif expected_change<-3 or momentum<-5: suggestion,color = "CAUTION","red"
        else: suggestion,color = "HOLD","orange"
        sector = SECTOR_MAP.get(symbol,"Technology")
        sensitivities = SECTOR_SENSITIVITIES.get(sector,SECTOR_SENSITIVITIES["Technology"])
        return {"symbol":symbol,"sector":sector,"current_price":round(current_price,2),
                "predicted_price":round(pred_price,2),"expected_change_pct":round(expected_change,2),
                "trend":trend,"momentum":mom_label,"momentum_pct":round(momentum,2),
                "confidence":confidence,"suggestion":suggestion,"suggestion_color":color,
                "r_squared":round(r2,4),"macro_reasons":_macro_reasons(sensitivities,MACRO_DATA[2024]),
                "coefficients":dict(zip(MACRO_VARS,[round(b,4) for b in betas[1:]])),
                "disclaimer":"⚠️ For educational purposes only. Not financial advice."}
    except Exception as e:
        return {"symbol":symbol,"error":str(e)}

def _macro_reasons(sensitivities, macro):
    reasons = []
    impacts = {"high_positive":("strongly positive","🟢"),"moderate_positive":("positive","🟡"),
               "low_positive":("mildly positive","🟡"),"high_negative":("strongly negative","🔴"),
               "moderate_negative":("negative","🔴"),"low_negative":("mildly negative","🟠")}
    for var, sens in sensitivities.items():
        val = macro.get(var,0)
        label = {"GDP_Growth":"GDP","Inflation":"Inflation","Interest_Rate":"Interest Rate",
                 "Exchange_Rate":"Exchange Rate","M3_Growth":"M3 Money Supply","IIP":"IIP"}.get(var,var)
        desc, icon = impacts.get(sens,("neutral","⚪"))
        reasons.append(f"{icon} {label} ({val}{'%' if var!='Exchange_Rate' else '₹/USD'}): {desc} impact")
    return reasons
