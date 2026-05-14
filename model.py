"""
model.py — Econometric Models for FinSight Analytics
Fixed Effects, Random Effects, Hausman, OLS, Diagnostics, Manual Prediction
"""
import numpy as np
import pandas as pd
from scipy import stats
from data_fetch import get_panel_dataframe, _generate_price_series, MACRO_DATA, MACRO_VARS, ALL_SYMBOLS

# ─── PANEL REGRESSION ────────────────────────────────────────────────────────────
def run_panel_regression(symbols, effect='both'):
    if len(symbols) < 2:
        return {"error": "Panel regression requires at least 2 stocks."}
    df = get_panel_dataframe(symbols).dropna(subset=['Close']+MACRO_VARS)
    df['log_price'] = np.log(df['Close'])
    fe = _fixed_effects(df)
    re = _random_effects(df)
    hausman = _hausman_test(fe, re)
    return {'fixed_effects':fe,'random_effects':re,'hausman':hausman,
            'n_stocks':len(symbols),'n_obs':len(df),'symbols':symbols,
            'model_comparison':_compare_models(fe,re,hausman)}

def _fixed_effects(df):
    df2 = df.copy()
    for col in ['log_price']+MACRO_VARS:
        gmeans = df2.groupby('Symbol')[col].transform('mean')
        df2[col+'_dm'] = df2[col] - gmeans
    y = df2['log_price_dm'].values
    X = df2[[c+'_dm' for c in MACRO_VARS]].values
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T@X)
    betas = XtX_inv@X.T@y
    resid = y - X@betas
    df_resid = n - k - len(df2['Symbol'].unique())
    s2 = (resid@resid)/max(df_resid,1)
    se = np.sqrt(np.diag(s2*XtX_inv))
    t_stats = betas/(se+1e-10)
    p_values = [2*(1-stats.t.cdf(abs(t),df=max(df_resid,1))) for t in t_stats]
    ss_res = resid@resid; ss_tot = ((y-y.mean())**2).sum()
    r2 = 1-ss_res/max(ss_tot,1e-10)
    return {"name":"Fixed Effects (Within)","coefficients":dict(zip(MACRO_VARS,[round(b,6) for b in betas])),
            "std_errors":dict(zip(MACRO_VARS,[round(s,6) for s in se])),
            "t_stats":dict(zip(MACRO_VARS,[round(t,4) for t in t_stats])),
            "p_values":dict(zip(MACRO_VARS,[round(p,4) for p in p_values])),
            "r_squared":round(r2,4),"n_obs":n,"betas_array":betas.tolist(),"vcov":(s2*XtX_inv).tolist()}

def _random_effects(df):
    df2 = df.copy()
    y = df2['log_price'].values; X = df2[MACRO_VARS].values
    n, k = X.shape
    X_aug = np.column_stack([np.ones(n),X])
    XtX_inv = np.linalg.pinv(X_aug.T@X_aug)
    betas_aug = XtX_inv@X_aug.T@y
    betas = betas_aug[1:]
    resid = y - X_aug@betas_aug
    df_resid = n-k-1; s2 = (resid@resid)/max(df_resid,1)
    se = np.sqrt(np.diag(s2*np.linalg.pinv(X.T@X)))
    t_stats = betas/(se+1e-10)
    p_values = [2*(1-stats.t.cdf(abs(t),df=max(df_resid,1))) for t in t_stats]
    ss_res = resid@resid; ss_tot = ((y-y.mean())**2).sum()
    r2 = 1-ss_res/max(ss_tot,1e-10)
    return {"name":"Random Effects (GLS)","coefficients":dict(zip(MACRO_VARS,[round(b,6) for b in betas])),
            "std_errors":dict(zip(MACRO_VARS,[round(s,6) for s in se])),
            "t_stats":dict(zip(MACRO_VARS,[round(t,4) for t in t_stats])),
            "p_values":dict(zip(MACRO_VARS,[round(p,4) for p in p_values])),
            "r_squared":round(r2,4),"n_obs":n,"betas_array":betas.tolist(),"vcov":(s2*np.linalg.pinv(X.T@X)).tolist()}

def _hausman_test(fe, re):
    try:
        b_fe = np.array(fe['betas_array']); b_re = np.array(re['betas_array'])
        V_fe = np.array(fe['vcov']); V_re = np.array(re['vcov'])
        diff = b_fe - b_re; V_diff = V_fe - V_re
        try: V_inv = np.linalg.pinv(V_diff)
        except: V_inv = np.eye(len(diff))
        H = float(diff@V_inv@diff); df_h = len(diff)
        p_val = 1-stats.chi2.cdf(H,df=df_h)
        decision = "Fixed Effects preferred" if p_val<0.05 else "Random Effects preferred"
        return {"statistic":round(H,4),"p_value":round(p_val,4),"df":df_h,"decision":decision,
                "reason":"Systematic difference — endogeneity detected" if p_val<0.05 else "No systematic difference — RE is efficient",
                "preferred_model":"Fixed Effects" if p_val<0.05 else "Random Effects"}
    except Exception as e:
        return {"error":str(e),"decision":"Inconclusive","preferred_model":"Fixed Effects"}

def _compare_models(fe, re, hausman):
    winner = hausman.get("preferred_model","Fixed Effects")
    return {"winner":winner,"fe_r2":fe["r_squared"],"re_r2":re["r_squared"],
            "recommendation":"Use Fixed Effects — entity effects correlated with regressors." if winner=="Fixed Effects"
            else "Use Random Effects — entity effects uncorrelated with regressors."}

# ─── OLS SINGLE STOCK ────────────────────────────────────────────────────────────
def run_ols_regression(symbol):
    df = _generate_price_series(symbol).dropna(subset=['Close']+MACRO_VARS)
    df['log_price'] = np.log(df['Close'])
    y = df['log_price'].values; X_raw = df[MACRO_VARS].values
    n, k = X_raw.shape; X = np.column_stack([np.ones(n),X_raw])
    XtX_inv = np.linalg.pinv(X.T@X); betas = XtX_inv@X.T@y
    resid = y - X@betas; df_resid = n-k-1; s2 = (resid@resid)/max(df_resid,1)
    se = np.sqrt(np.diag(s2*XtX_inv)); t_stats = betas/(se+1e-10)
    p_values = [2*(1-stats.t.cdf(abs(t),df=max(df_resid,1))) for t in t_stats]
    ss_res = resid@resid; ss_tot = ((y-y.mean())**2).sum()
    r2 = 1-ss_res/max(ss_tot,1e-10); adj_r2 = 1-(1-r2)*(n-1)/max(df_resid,1)
    ss_reg = ss_tot-ss_res; f_stat = (ss_reg/k)/max(s2,1e-10)
    f_pval = 1-stats.f.cdf(f_stat,k,df_resid)
    corr_matrix = df[['log_price']+MACRO_VARS].corr().round(4).to_dict()
    tail_df = df.tail(24)
    return {"symbol":symbol,"n_obs":n,
            "coefficients":dict(zip(['Intercept']+MACRO_VARS,[round(b,6) for b in betas])),
            "std_errors":dict(zip(['Intercept']+MACRO_VARS,[round(s,6) for s in se])),
            "t_stats":dict(zip(['Intercept']+MACRO_VARS,[round(t,4) for t in t_stats])),
            "p_values":dict(zip(['Intercept']+MACRO_VARS,[round(p,4) for p in p_values])),
            "r_squared":round(r2,4),"adj_r_squared":round(adj_r2,4),
            "f_statistic":round(f_stat,4),"f_pvalue":round(f_pval,4),
            "aic":round(n*np.log(ss_res/n)+2*(k+1),4),"bic":round(n*np.log(ss_res/n)+np.log(n)*(k+1),4),
            "corr_matrix":corr_matrix,"dates_tail":tail_df['Date'].dt.strftime('%Y-%m').tolist(),
            "actual":[round(v,4) for v in tail_df['log_price'].tolist()],
            "fitted":[round(v,4) for v in (X@betas)[-24:].tolist()],
            "residuals":[round(v,4) for v in resid[-24:].tolist()]}

# ─── DIAGNOSTIC TESTS ────────────────────────────────────────────────────────────
def run_diagnostic_tests(symbol):
    df = _generate_price_series(symbol).dropna(subset=['Close']+MACRO_VARS)
    df['log_price'] = np.log(df['Close'])
    results = {'adf':_adf_test(df['log_price'].values),'vif':_vif(df[MACRO_VARS].values,MACRO_VARS)}
    y = df['log_price'].values; X = np.column_stack([np.ones(len(y)),df[MACRO_VARS].values])
    betas = np.linalg.pinv(X.T@X)@X.T@y; resid = y-X@betas
    results['durbin_watson'] = _durbin_watson(resid)
    results['breusch_pagan'] = _breusch_pagan(resid,X)
    return results

def _adf_test(series):
    n=len(series); dy=np.diff(series); y_lag=series[:-1]
    X=np.column_stack([np.ones(n-1),y_lag]); b=np.linalg.pinv(X.T@X)@X.T@dy
    res=dy-X@b; s2=(res@res)/(n-3); se_b1=np.sqrt(s2*np.linalg.pinv(X.T@X)[1,1])
    tau=b[1]/max(se_b1,1e-10); stationary=tau<-2.86
    return {"statistic":round(tau,4),"critical_5pct":-2.86,"stationary":bool(stationary),
            "interpretation":"Series is stationary (reject unit root)" if stationary else "Series has unit root (non-stationary)"}

def _vif(X_array, names):
    result={}; k=X_array.shape[1]
    for i,name in enumerate(names):
        y=X_array[:,i]; X_other=np.column_stack([np.ones(len(y)),X_array[:,[j for j in range(k) if j!=i]]])
        b=np.linalg.pinv(X_other.T@X_other)@X_other.T@y; y_hat=X_other@b
        ss_res=((y-y_hat)**2).sum(); ss_tot=((y-y.mean())**2).sum()
        r2=1-ss_res/max(ss_tot,1e-10); vif=1/max(1-r2,1e-10)
        result[name]=round(min(vif,999.9),2)
    return result

def _durbin_watson(resid):
    dw=sum((resid[i]-resid[i-1])**2 for i in range(1,len(resid)))/max((resid**2).sum(),1e-10)
    if dw<1.5: interp="Positive autocorrelation detected"
    elif dw>2.5: interp="Negative autocorrelation detected"
    else: interp="No significant autocorrelation (1.5–2.5)"
    return {"statistic":round(dw,4),"interpretation":interp}

def _breusch_pagan(resid, X):
    n=len(resid); resid2=resid**2; b=np.linalg.pinv(X.T@X)@X.T@resid2; fitted=X@b
    ss_reg=((fitted-resid2.mean())**2).sum(); ss_tot=((resid2-resid2.mean())**2).sum()
    r2=ss_reg/max(ss_tot,1e-10); lm=n*r2; k=X.shape[1]-1
    p_val=1-stats.chi2.cdf(lm,df=k); homosked=p_val>0.05
    return {"lm_statistic":round(lm,4),"p_value":round(p_val,4),"homoskedastic":bool(homosked),
            "interpretation":"Homoskedastic residuals" if homosked else "Heteroskedasticity detected — use robust SEs"}

# ─── MANUAL PREDICTION (Panel Regression Based) ──────────────────────────────────
def manual_predict(symbol, year, stock_price, selected_vars, custom_values=None):
    """
    Predict stock price using trained panel regression model.
    Uses all panel stocks to train, then predicts for specific stock+year+macro combo.
    """
    # Train panel model on all available stocks
    train_symbols = [s for s in ALL_SYMBOLS if s != symbol][:20]  # 20 training stocks
    if symbol in ALL_SYMBOLS:
        train_symbols.append(symbol)
    
    df_panel = get_panel_dataframe(train_symbols).dropna(subset=['Close']+MACRO_VARS)
    df_panel['log_price'] = np.log(df_panel['Close'])
    
    vars_to_use = [v for v in selected_vars if v in MACRO_VARS]
    if not vars_to_use:
        vars_to_use = MACRO_VARS
    
    # Fixed effects (within estimator)
    df2 = df_panel.copy()
    for col in ['log_price']+vars_to_use:
        gmeans = df2.groupby('Symbol')[col].transform('mean')
        df2[col+'_dm'] = df2[col] - gmeans
    
    y = df2['log_price_dm'].values
    X = df2[[c+'_dm' for c in vars_to_use]].values
    n_obs, k = X.shape
    XtX_inv = np.linalg.pinv(X.T@X)
    betas = XtX_inv@X.T@y
    resid = y - X@betas
    df_resid = n_obs - k - len(df2['Symbol'].unique())
    s2 = (resid@resid)/max(df_resid,1)
    se = np.sqrt(np.diag(s2*XtX_inv))
    t_stats = betas/(se+1e-10)
    p_values = [2*(1-stats.t.cdf(abs(t),df=max(df_resid,1))) for t in t_stats]
    ss_res = resid@resid; ss_tot = ((y-y.mean())**2).sum()
    r2 = 1-ss_res/max(ss_tot,1e-10)
    
    # Entity mean for target symbol (intercept proxy)
    sym_data = df_panel[df_panel['Symbol']==symbol] if symbol in df_panel['Symbol'].values else df_panel
    entity_mean_log = sym_data['log_price'].mean() if len(sym_data)>0 else np.log(stock_price)
    
    # Get macro values for prediction
    if custom_values:
        macro_vals = [custom_values.get(v, MACRO_DATA.get(int(year),{}).get(v, 0)) for v in vars_to_use]
    else:
        year_macro = MACRO_DATA.get(int(year), MACRO_DATA[max(MACRO_DATA.keys())])
        macro_vals = [year_macro.get(v, 0) for v in vars_to_use]
    
    # Global macro means from training data
    macro_means = [df_panel[v].mean() for v in vars_to_use]
    macro_demeaned = np.array([macro_vals[i] - macro_means[i] for i in range(len(vars_to_use))])
    
    # Prediction: entity_mean + beta * (X - X_bar)
    pred_log = entity_mean_log + float(betas @ macro_demeaned)
    predicted_price = float(np.exp(pred_log))
    current_price = float(stock_price)
    change_pct = (predicted_price - current_price) / current_price * 100
    
    # Impact analysis per variable
    impact_analysis = {}
    for i, var in enumerate(vars_to_use):
        beta = betas[i]; macro_val = macro_vals[i]
        direction = "Positive" if beta > 0 else "Negative"
        strength = abs(beta * macro_demeaned[i])
        impact_analysis[var] = {
            "coefficient": round(float(beta), 6),
            "std_error": round(float(se[i]), 6),
            "t_stat": round(float(t_stats[i]), 4),
            "p_value": round(float(p_values[i]), 4),
            "macro_value": round(macro_val, 4),
            "direction": direction,
            "significant": bool(p_values[i] < 0.05),
            "impact_strength": round(strength, 6),
        }
    
    trend = "UP" if change_pct > 0 else "DOWN"
    confidence = "HIGH" if abs(change_pct) > 5 and r2 > 0.6 else "MODERATE" if abs(change_pct) > 2 else "LOW"
    
    return {
        "symbol": symbol,
        "year": year,
        "input_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "change_pct": round(change_pct, 2),
        "trend": trend,
        "confidence": confidence,
        "r_squared": round(r2, 4),
        "n_obs": n_obs,
        "n_stocks_in_panel": len(train_symbols),
        "vars_used": vars_to_use,
        "impact_analysis": impact_analysis,
        "macro_values_used": dict(zip(vars_to_use, [round(v, 4) for v in macro_vals])),
        "model_type": "Fixed Effects Panel Regression",
        "model_equation": f"log(Price)_it = α + " + " + ".join([f"β{i+1}·{v}" for i,v in enumerate(vars_to_use)]) + " + μ_i + ε_it",
        "economic_interpretation": _economic_interpret(impact_analysis, trend),
    }

def _economic_interpret(impact, trend):
    msgs = []
    for var, info in impact.items():
        sign = "+" if info['direction']=="Positive" else "-"
        sig = "significant" if info['significant'] else "not significant"
        label = {"GDP_Growth":"GDP Growth","Inflation":"Inflation","Interest_Rate":"Interest Rate",
                 "Exchange_Rate":"Exchange Rate","M3_Growth":"Money Supply (M3)","IIP":"IIP"}.get(var,var)
        msgs.append(f"{label}: β={info['coefficient']:.4f} ({sign}impact, {sig} at 5%)")
    trend_msg = f"Overall trend: {trend} — macroeconomic conditions {'support' if trend=='UP' else 'suppress'} stock appreciation."
    return msgs + [trend_msg]
