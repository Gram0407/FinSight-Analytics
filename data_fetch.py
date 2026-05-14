"""
data_fetch.py — FinSight Analytics Data Provider
"""
import pandas as pd
import numpy as np
import os

# ─── WATCHLIST CSV LOADER ──────────────────────────────────────────────────────
def load_watchlist_csv():
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset', 'MC_WATCHLIST_DATA__1_.csv')
    if not os.path.exists(csv_path):
        return []
    stocks = []
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            try:
                name = str(row.get('Stocks/F&O', '')).strip()
                def safe_float(val):
                    try: return float(str(val).replace(',','').replace('+','').strip())
                    except: return 0.0
                price = safe_float(row.get('Live Price', 0))
                change = safe_float(row.get('Change', 0))
                pct = safe_float(row.get('Change %', 0))
                high = safe_float(row.get("Today's High", price))
                low = safe_float(row.get("Today's low", price))
                low52 = safe_float(row.get('52 Week Low', price))
                high52 = safe_float(row.get('52 Week High', price))
                sector = str(row.get('Sector', 'Unknown')).strip()
                mcap = str(row.get('MCAP', 'Large Cap')).strip()
                prev_close = safe_float(row.get('Previous Close', price))
                term = str(row.get('ShortTerm / LongTerm', '')).strip()
                if not name or price == 0: continue
                stocks.append({'symbol': name,'name': name,'price': round(price,2),
                    'change': round(change,2),'pct': round(pct,2),'today_high': round(high,2),
                    'today_low': round(low,2),'low_52w': round(low52,2),'high_52w': round(high52,2),
                    'sector': sector,'mcap': mcap,'prev_close': round(prev_close,2),
                    'term': term or 'Long-term','exchange': 'NSE','fno': mcap=='Large Cap'})
            except: continue
    except Exception as e:
        print(f"CSV error: {e}")
    return stocks

CSV_STOCKS = load_watchlist_csv()

# ─── STOCK UNIVERSE ──────────────────────────────────────────────────────────────
STOCK_UNIVERSE = {
    "Technology": ["TCS","INFY","WIPRO","HCLTECH","TECHM","MPHASIS","LTIM","COFORGE"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","INDUSINDBK","BANDHANBNK"],
    "Energy": ["RELIANCE","ONGC","BPCL","IOC","NTPC","POWERGRID","TATAPOWER"],
    "FMCG": ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","EMAMILTD"],
    "Auto": ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT"],
    "Pharma": ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","BIOCON","AUROPHARMA"],
    "Infrastructure": ["LT","ADANIPORTS","ADANIGREEN","ULTRACEMCO","GRASIM"],
    "Finance": ["BAJFINANCE","BAJAJFINSV","HDFC","LICHSGFIN","MUTHOOTFIN"],
    "Metals": ["TATASTEEL","HINDALCO","JSWSTEEL","VEDL","COALINDIA"],
}
ALL_SYMBOLS = [s for stocks in STOCK_UNIVERSE.values() for s in stocks]
SECTOR_MAP = {s: sec for sec, stocks in STOCK_UNIVERSE.items() for s in stocks}

BASE_PRICES = {
    "TCS":2500,"INFY":1100,"WIPRO":550,"HCLTECH":900,"TECHM":600,"MPHASIS":400,"LTIM":1200,"COFORGE":500,
    "HDFCBANK":900,"ICICIBANK":280,"SBIN":230,"AXISBANK":420,"KOTAKBANK":700,"INDUSINDBK":750,"BANDHANBNK":300,
    "RELIANCE":900,"ONGC":200,"BPCL":300,"IOC":330,"NTPC":130,"POWERGRID":120,"TATAPOWER":75,
    "HINDUNILVR":800,"ITC":200,"NESTLEIND":5000,"BRITANNIA":1500,"DABUR":250,"MARICO":200,"EMAMILTD":450,
    "MARUTI":3500,"TATAMOTORS":350,"M&M":600,"BAJAJ-AUTO":2000,"HEROMOTOCO":2500,"EICHERMOT":1500,
    "SUNPHARMA":700,"DRREDDY":2500,"CIPLA":450,"DIVISLAB":1200,"BIOCON":300,"AUROPHARMA":500,
    "LT":1200,"ADANIPORTS":300,"ADANIGREEN":200,"ULTRACEMCO":2500,"GRASIM":900,
    "BAJFINANCE":2000,"BAJAJFINSV":700,"HDFC":1000,"LICHSGFIN":300,"MUTHOOTFIN":400,
    "TATASTEEL":250,"HINDALCO":120,"JSWSTEEL":150,"VEDL":150,"COALINDIA":350,
}

MACRO_DATA = {
    2015:{"GDP_Growth":8.0,"Inflation":4.9,"Interest_Rate":7.75,"Exchange_Rate":65.5,"M3_Growth":11.3,"IIP":3.4},
    2016:{"GDP_Growth":8.3,"Inflation":4.5,"Interest_Rate":6.75,"Exchange_Rate":67.2,"M3_Growth":10.1,"IIP":4.6},
    2017:{"GDP_Growth":6.8,"Inflation":3.3,"Interest_Rate":6.25,"Exchange_Rate":65.1,"M3_Growth":9.8,"IIP":4.4},
    2018:{"GDP_Growth":7.2,"Inflation":3.9,"Interest_Rate":6.5,"Exchange_Rate":69.8,"M3_Growth":10.5,"IIP":3.8},
    2019:{"GDP_Growth":6.1,"Inflation":4.8,"Interest_Rate":5.15,"Exchange_Rate":71.3,"M3_Growth":9.6,"IIP":0.1},
    2020:{"GDP_Growth":-6.6,"Inflation":6.2,"Interest_Rate":4.0,"Exchange_Rate":74.1,"M3_Growth":12.2,"IIP":-8.4},
    2021:{"GDP_Growth":8.7,"Inflation":5.1,"Interest_Rate":4.0,"Exchange_Rate":73.9,"M3_Growth":13.1,"IIP":13.7},
    2022:{"GDP_Growth":7.0,"Inflation":6.7,"Interest_Rate":6.5,"Exchange_Rate":82.7,"M3_Growth":8.9,"IIP":5.1},
    2023:{"GDP_Growth":8.2,"Inflation":5.4,"Interest_Rate":6.5,"Exchange_Rate":83.1,"M3_Growth":8.5,"IIP":5.9},
    2024:{"GDP_Growth":8.2,"Inflation":4.7,"Interest_Rate":6.5,"Exchange_Rate":84.0,"M3_Growth":9.2,"IIP":4.2},
}
MACRO_VARS = ['GDP_Growth','Inflation','Interest_Rate','Exchange_Rate','M3_Growth','IIP']

def _generate_price_series(symbol, seed=None):
    if seed is None: seed = sum(ord(c) for c in symbol)
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2015-01-01','2024-12-31',freq='ME')
    base = BASE_PRICES.get(symbol, 500)
    sector = SECTOR_MAP.get(symbol,"Technology")
    drift_map = {"Technology":0.012,"Banking":0.009,"Energy":0.006,"FMCG":0.008,
                 "Auto":0.007,"Pharma":0.010,"Infrastructure":0.008,"Finance":0.011,"Metals":0.005}
    drift = drift_map.get(sector,0.008)
    vol = rng.uniform(0.04,0.08)
    prices = [base]
    for _ in range(len(dates)-1):
        ret = drift + vol*rng.standard_normal()
        prices.append(prices[-1]*(1+ret))
    df = pd.DataFrame({'Date':dates,'Close':prices})
    df['Open'] = df['Close']*(1+rng.uniform(-0.01,0.01,len(df)))
    df['High'] = df[['Open','Close']].max(axis=1)*(1+rng.uniform(0,0.02,len(df)))
    df['Low']  = df[['Open','Close']].min(axis=1)*(1-rng.uniform(0,0.02,len(df)))
    df['Volume'] = rng.integers(1_000_000,50_000_000,len(df))
    df['Symbol'] = symbol; df['Sector'] = sector; df['Year'] = df['Date'].dt.year
    macro_df = pd.DataFrame([{**{'Year':y},**vals} for y,vals in MACRO_DATA.items()])
    df = df.merge(macro_df,on='Year',how='left')
    return df

def get_stock_data(symbol):
    df = _generate_price_series(symbol)
    curr = df['Close'].iloc[-1]; prev = df['Close'].iloc[-2]
    latest = df.tail(12)
    return {"symbol":symbol,"sector":SECTOR_MAP.get(symbol,"Unknown"),
            "current_price":round(curr,2),"price_change":round(curr-prev,2),
            "pct_change":round((curr/prev-1)*100,2),"high_52w":round(df['Close'].tail(12).max(),2),
            "low_52w":round(df['Close'].tail(12).min(),2),"today_high":round(curr*1.015,2),
            "today_low":round(curr*0.985,2),"prev_close":round(prev,2),
            "dates":df['Date'].dt.strftime('%Y-%m').tolist(),
            "prices":[round(p,2) for p in df['Close'].tolist()],
            "volumes":df['Volume'].tolist(),
            "macro":{"dates":latest['Date'].dt.strftime('%Y-%m').tolist(),
                     "gdp":latest['GDP_Growth'].tolist(),"inflation":latest['Inflation'].tolist(),
                     "interest":latest['Interest_Rate'].tolist(),"exchange":latest['Exchange_Rate'].tolist(),
                     "m3":latest['M3_Growth'].tolist(),"iip":latest['IIP'].tolist()}}

def get_macro_data():
    years = sorted(MACRO_DATA.keys())
    return {"years":years,"gdp":[MACRO_DATA[y]["GDP_Growth"] for y in years],
            "inflation":[MACRO_DATA[y]["Inflation"] for y in years],
            "interest":[MACRO_DATA[y]["Interest_Rate"] for y in years],
            "exchange":[MACRO_DATA[y]["Exchange_Rate"] for y in years],
            "m3":[MACRO_DATA[y]["M3_Growth"] for y in years],
            "iip":[MACRO_DATA[y]["IIP"] for y in years]}

def get_all_stocks():
    result = []; seen = set()
    for s in CSV_STOCKS:
        seen.add(s['symbol'].upper()); result.append(s)
    for symbol in ALL_SYMBOLS:
        if symbol.upper() in seen: continue
        df = _generate_price_series(symbol)
        curr = df['Close'].iloc[-1]; prev = df['Close'].iloc[-2]
        result.append({'symbol':symbol,'name':symbol,'price':round(curr,2),
            'change':round(curr-prev,2),'pct':round((curr/prev-1)*100,2),
            'today_high':round(curr*1.015,2),'today_low':round(curr*0.985,2),
            'high_52w':round(df['Close'].tail(12).max(),2),'low_52w':round(df['Close'].tail(12).min(),2),
            'sector':SECTOR_MAP.get(symbol,'Unknown'),'mcap':'Large Cap','prev_close':round(prev,2),
            'term':'Long-term','exchange':'NSE','fno':True})
    return result

def get_panel_dataframe(symbols):
    return pd.concat([_generate_price_series(s) for s in symbols], ignore_index=True)

def get_macro_for_year(year):
    year = int(year)
    return MACRO_DATA.get(year, MACRO_DATA[max(MACRO_DATA.keys())])

# ─── DB-AWARE DATA FUNCTIONS ──────────────────────────────────────────────────────

def get_macro_data_db():
    """Return macro data from DB (admin-editable), fallback to static."""
    try:
        import sqlite3, os
        db_path = os.path.join(os.path.dirname(__file__), 'database.db')
        if not os.path.exists(db_path):
            return get_macro_data()
        conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
        # Get latest entry per year
        rows = conn.execute('''SELECT year, GDP_Growth, Inflation, Interest_Rate,
            Exchange_Rate, M3_Growth, IIP FROM macro_data
            WHERE id IN (SELECT MAX(id) FROM macro_data GROUP BY year)
            ORDER BY year''').fetchall()
        conn.close()
        if not rows:
            return get_macro_data()
        years = [r['year'] for r in rows]
        return {"years": years,
                "gdp":      [r['GDP_Growth']    for r in rows],
                "inflation":[r['Inflation']     for r in rows],
                "interest": [r['Interest_Rate'] for r in rows],
                "exchange": [r['Exchange_Rate'] for r in rows],
                "m3":       [r['M3_Growth']     for r in rows],
                "iip":      [r['IIP']           for r in rows]}
    except Exception as e:
        print(f"DB macro error: {e}")
        return get_macro_data()

def get_all_stocks_db():
    """Return all stocks: CSV + synthetic + admin-added custom stocks."""
    result = []
    seen = set()
    # 1. CSV stocks
    for s in CSV_STOCKS:
        seen.add(s['symbol'].upper()); result.append(s)
    # 2. Admin custom stocks from DB
    try:
        import sqlite3, os
        db_path = os.path.join(os.path.dirname(__file__), 'database.db')
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM custom_stocks WHERE is_active=1").fetchall()
            conn.close()
            for r in rows:
                sym = r['symbol'].upper()
                if sym in seen: continue
                seen.add(sym)
                result.append({'symbol':r['symbol'],'name':r['name'] or r['symbol'],
                    'price':r['price'],'change':r['change_val'],'pct':r['pct_change'],
                    'today_high':r['today_high'],'today_low':r['today_low'],
                    'high_52w':r['high_52w'],'low_52w':r['low_52w'],
                    'sector':r['sector'],'mcap':r['mcap'],'prev_close':r['prev_close'],
                    'term':r['term'],'exchange':r['exchange'],'fno':bool(r['fno'])})
    except Exception as e:
        print(f"DB stocks error: {e}")
    # 3. Synthetic panel stocks
    for symbol in ALL_SYMBOLS:
        if symbol.upper() in seen: continue
        df = _generate_price_series(symbol)
        curr = df['Close'].iloc[-1]; prev = df['Close'].iloc[-2]
        result.append({'symbol':symbol,'name':symbol,'price':round(curr,2),
            'change':round(curr-prev,2),'pct':round((curr/prev-1)*100,2),
            'today_high':round(curr*1.015,2),'today_low':round(curr*0.985,2),
            'high_52w':round(df['Close'].tail(12).max(),2),'low_52w':round(df['Close'].tail(12).min(),2),
            'sector':SECTOR_MAP.get(symbol,'Unknown'),'mcap':'Large Cap','prev_close':round(prev,2),
            'term':'Long-term','exchange':'NSE','fno':True})
    return result
