from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_bcrypt import Bcrypt
import sqlite3, os, json
from datetime import datetime
from model import run_panel_regression, run_ols_regression, run_diagnostic_tests, manual_predict
from ai_assistant import get_ai_response, predict_stock_movement
from data_fetch import get_stock_data, get_macro_data_db, get_all_stocks_db, MACRO_VARS, MACRO_DATA as MACRO_FALLBACK

app = Flask(__name__)
app.secret_key = 'finsight_dual_session_2025'
bcrypt = Bcrypt(app)

UPLOAD_FOLDER = 'static/images/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DEFAULT_ADMIN_EMAIL    = 'admin@finsight.com'
DEFAULT_ADMIN_PASSWORD = 'Gram@0417'
DEFAULT_ADMIN_USERNAME = 'admin'

# ─── DB ─────────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db(); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL, phone TEXT, broker TEXT DEFAULT 'Zerodha',
        profile_image TEXT DEFAULT 'default.png', theme TEXT DEFAULT 'dark',
        is_admin INTEGER DEFAULT 0, status TEXT DEFAULT 'pending',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    for col, typ, defval in [('is_admin','INTEGER','0'),('status','TEXT',"'pending'"),('theme','TEXT',"'dark'")]:
        try: c.execute(f"ALTER TABLE users ADD COLUMN {col} {typ} DEFAULT {defval}")
        except: pass

    c.execute('''CREATE TABLE IF NOT EXISTS macro_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT, year INTEGER NOT NULL,
        entry_date TEXT NOT NULL, GDP_Growth REAL, Inflation REAL,
        Interest_Rate REAL, Exchange_Rate REAL, M3_Growth REAL, IIP REAL,
        notes TEXT DEFAULT '', updated_by INTEGER,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS custom_stocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, name TEXT,
        sector TEXT DEFAULT 'Other', mcap TEXT DEFAULT 'Large Cap',
        exchange TEXT DEFAULT 'NSE', price REAL DEFAULT 0,
        change_val REAL DEFAULT 0, pct_change REAL DEFAULT 0,
        today_high REAL DEFAULT 0, today_low REAL DEFAULT 0,
        high_52w REAL DEFAULT 0, low_52w REAL DEFAULT 0,
        prev_close REAL DEFAULT 0, term TEXT DEFAULT 'Long-term',
        fno INTEGER DEFAULT 0, is_active INTEGER DEFAULT 1,
        added_by INTEGER, added_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS admin_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT, setting_key TEXT UNIQUE NOT NULL,
        setting_value TEXT NOT NULL DEFAULT '1',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, symbol TEXT,
        added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        symbol TEXT, input_price REAL, predicted_price REAL,
        trend TEXT, confidence TEXT, year_used INTEGER,
        macro_snapshot TEXT, model_type TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id))''')

    defaults = [
        ('show_live_stocks','1'),('show_macro','1'),('show_regression','1'),
        ('show_prediction','1'),('show_history','1'),('show_ai','1'),
        ('show_gdp','1'),('show_inflation','1'),('show_interest','1'),
        ('show_exchange','1'),('show_m3','1'),('show_iip','1'),
        ('market_status','open'),('site_notice',''),
    ]
    for k,v in defaults:
        c.execute('INSERT OR IGNORE INTO admin_settings (setting_key,setting_value) VALUES (?,?)',(k,v))

    # Seed macro data
    if c.execute('SELECT COUNT(*) FROM macro_data').fetchone()[0] == 0:
        for yr, vals in MACRO_FALLBACK.items():
            c.execute('''INSERT INTO macro_data (year,entry_date,GDP_Growth,Inflation,
                Interest_Rate,Exchange_Rate,M3_Growth,IIP,notes,updated_by,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                (yr, f'{yr}-12-31', vals['GDP_Growth'], vals['Inflation'],
                 vals['Interest_Rate'], vals['Exchange_Rate'], vals['M3_Growth'],
                 vals['IIP'], 'Initial seed', 1, f'{yr}-12-31 00:00:00'))

    # Admin account
    existing_admin = c.execute('SELECT id,password FROM users WHERE is_admin=1').fetchone()
    if not existing_admin:
        hashed = bcrypt.generate_password_hash(DEFAULT_ADMIN_PASSWORD).decode('utf-8')
        c.execute("INSERT OR IGNORE INTO users (username,email,password,is_admin,status) VALUES (?,?,?,1,'approved')",
                  (DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_EMAIL, hashed))
    else:
        try:
            if bcrypt.check_password_hash(existing_admin['password'], 'admin123'):
                new_hashed = bcrypt.generate_password_hash(DEFAULT_ADMIN_PASSWORD).decode('utf-8')
                c.execute('UPDATE users SET password=? WHERE id=?',(new_hashed,existing_admin['id']))
        except: pass

    conn.commit(); conn.close()

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
def get_setting(key, default='1'):
    conn = get_db()
    row = conn.execute('SELECT setting_value FROM admin_settings WHERE setting_key=?',(key,)).fetchone()
    conn.close()
    return row['setting_value'] if row else default

def get_latest_macro_from_db():
    """Returns the latest macro entry per year (most recent update_at), plus last_updated timestamp."""
    conn = get_db()
    # Latest entry per year
    rows = conn.execute('''
        SELECT m.* FROM macro_data m
        INNER JOIN (SELECT year, MAX(updated_at) AS max_ua FROM macro_data GROUP BY year) latest
        ON m.year = latest.year AND m.updated_at = latest.max_ua
        ORDER BY m.year DESC''').fetchall()
    # Overall last update
    last_row = conn.execute(
        'SELECT updated_at, entry_date, year FROM macro_data ORDER BY updated_at DESC LIMIT 1').fetchone()
    conn.close()
    macro_by_year = {r['year']: dict(r) for r in rows}
    last_updated = dict(last_row) if last_row else {}
    return macro_by_year, last_updated

def get_settings_dict():
    conn = get_db()
    rows = conn.execute('SELECT setting_key,setting_value FROM admin_settings').fetchall()
    conn.close()
    return {r['setting_key']: r['setting_value'] for r in rows}

# ─── SESSION HELPERS (DUAL LOGIN SUPPORT) ────────────────────────────────────────
# Admin session: session['admin_id'], session['admin_username']
# User session:  session['user_id'],  session['username']
# Both can coexist — admin can have /admin open, user has /dashboard open in same browser

def require_login(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def require_admin(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check admin session (separate from user session)
        if 'admin_id' not in session:
            return jsonify({'success': False, 'message': 'Admin session required. Please log in at /login as admin.'}), 401
        conn = get_db()
        user = conn.execute('SELECT is_admin FROM users WHERE id=?',(session['admin_id'],)).fetchone()
        conn.close()
        if not user or not user['is_admin']:
            return jsonify({'success': False, 'message': 'Admin access required.'}), 403
        return f(*args, **kwargs)
    return decorated

# ─── AUTH ────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session: return redirect(url_for('dashboard'))
    if 'admin_id' in session: return redirect(url_for('admin_panel'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email','').strip()
        password = request.form.get('password','')
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone()
        conn.close()
        if not user:
            return jsonify({'success':False,'message':'No account found with this email'})
        if not bcrypt.check_password_hash(user['password'], password):
            return jsonify({'success':False,'message':'Incorrect password'})

        if user['is_admin']:
            # Admin login — set ADMIN session (preserve any existing user session)
            session['admin_id'] = user['id']
            session['admin_username'] = user['username']
            # If admin also wants to use user dashboard, set user session too
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['theme'] = user['theme'] or 'dark'
            return jsonify({'success':True,'redirect':url_for('admin_panel'),'is_admin':True})

        # Regular user checks
        if user['status'] == 'pending':
            return jsonify({'success':False,'message':'⏳ Account pending admin approval.','pending':True})
        if user['status'] == 'blocked':
            return jsonify({'success':False,'message':'🚫 Account blocked. Contact administrator.','blocked':True})
        if user['status'] != 'approved':
            return jsonify({'success':False,'message':'Account not approved yet'})

        # User login — set USER session (preserve any existing admin session)
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['theme'] = user['theme'] or 'dark'
        session['is_admin'] = False
        return jsonify({'success':True,'redirect':url_for('dashboard'),'is_admin':False})
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        email = request.form.get('email','').strip()
        password = request.form.get('password','')
        phone = request.form.get('phone','').strip()
        if not all([username,email,password]):
            return jsonify({'success':False,'message':'All fields required'})
        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        try:
            conn = get_db()
            conn.execute('INSERT INTO users (username,email,password,phone,status) VALUES (?,?,?,?,?)',
                         (username,email,hashed,phone,'pending'))
            conn.commit(); conn.close()
            return jsonify({'success':True,'redirect':url_for('login')+'?registered=1'})
        except sqlite3.IntegrityError:
            return jsonify({'success':False,'message':'Email or username already exists'})
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout current user session only (preserve admin session if active)."""
    session.pop('user_id', None); session.pop('username', None)
    session.pop('theme', None); session.pop('is_admin', None)
    return redirect(url_for('login'))

@app.route('/admin/logout')
def admin_logout():
    """Logout admin session only (preserve user session if active)."""
    session.pop('admin_id', None); session.pop('admin_username', None)
    return redirect(url_for('login'))

@app.route('/logout/all')
def logout_all():
    session.clear()
    return redirect(url_for('login'))

# ─── USER DASHBOARD ──────────────────────────────────────────────────────────────
@app.route('/dashboard')
@require_login
def dashboard():
    conn = get_db()
    watchlist = conn.execute('SELECT symbol FROM watchlist WHERE user_id=?',(session['user_id'],)).fetchall()
    conn.close()
    settings = get_settings_dict()
    stocks = get_all_stocks_db()
    macro_by_year, last_updated = get_latest_macro_from_db()

    # Get most recent year's data for the stat cards
    latest_year = max(macro_by_year.keys()) if macro_by_year else 2024
    latest_macro = macro_by_year.get(latest_year, {})

    # Format last_updated nicely
    last_updated_str = ''
    if last_updated:
        try:
            ua = last_updated.get('updated_at','')
            dt = datetime.fromisoformat(ua.replace('Z','')) if ua else None
            last_updated_str = dt.strftime('%d %b %Y, %I:%M %p') if dt else ua[:16]
        except: last_updated_str = str(last_updated.get('updated_at',''))[:16]

    return render_template('dashboard.html',
        username=session['username'],
        theme=session.get('theme','dark'),
        is_admin=bool(session.get('admin_id')),
        watchlist=[w['symbol'] for w in watchlist],
        stocks=stocks,
        macro_years=sorted(macro_by_year.keys()),
        macro_vars=MACRO_VARS,
        settings=settings,
        latest_macro=latest_macro,
        latest_year=latest_year,
        last_updated=last_updated_str,
        macro_by_year_json=json.dumps({str(k): v for k,v in macro_by_year.items()}))

@app.route('/profile', methods=['GET','POST'])
@require_login
def profile():
    conn = get_db()
    if request.method == 'POST':
        phone = request.form.get('phone',''); broker = request.form.get('broker','Zerodha')
        theme = request.form.get('theme','dark'); img_path = None
        if 'profile_image' in request.files:
            f = request.files['profile_image']
            if f.filename:
                fname = f'user_{session["user_id"]}_{f.filename}'
                f.save(os.path.join(app.config['UPLOAD_FOLDER'],fname)); img_path = fname
        if img_path:
            conn.execute('UPDATE users SET phone=?,broker=?,profile_image=?,theme=? WHERE id=?',
                         (phone,broker,img_path,theme,session['user_id']))
        else:
            conn.execute('UPDATE users SET phone=?,broker=?,theme=? WHERE id=?',
                         (phone,broker,theme,session['user_id']))
        conn.commit(); session['theme'] = theme
        return jsonify({'success':True,'message':'Profile updated'})
    user = conn.execute('SELECT * FROM users WHERE id=?',(session['user_id'],)).fetchone()
    watchlist = conn.execute('SELECT symbol FROM watchlist WHERE user_id=?',(session['user_id'],)).fetchall()
    conn.close()
    return render_template('profile.html',user=dict(user),watchlist=[w['symbol'] for w in watchlist])

# ─── ADMIN PANEL ─────────────────────────────────────────────────────────────────
@app.route('/admin')
def admin_panel():
    if 'admin_id' not in session: return redirect(url_for('login'))
    conn = get_db()
    users = conn.execute('SELECT id,username,email,phone,broker,status,is_admin,created_at FROM users ORDER BY created_at DESC').fetchall()
    macro_rows = conn.execute('SELECT * FROM macro_data ORDER BY year DESC, updated_at DESC').fetchall()
    custom_stocks = conn.execute('SELECT * FROM custom_stocks ORDER BY added_at DESC').fetchall()
    settings_rows = conn.execute('SELECT * FROM admin_settings ORDER BY setting_key').fetchall()
    pending_count = conn.execute("SELECT COUNT(*) FROM users WHERE status='pending'").fetchone()[0]
    total_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_admin=0").fetchone()[0]
    predictions_count = conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
    conn.close()
    return render_template('admin.html',
        username=session.get('admin_username','Admin'),
        users=[dict(u) for u in users],
        macro_rows=[dict(m) for m in macro_rows],
        custom_stocks=[dict(s) for s in custom_stocks],
        settings={r['setting_key']:r['setting_value'] for r in settings_rows},
        pending_count=pending_count, total_users=total_users,
        predictions_count=predictions_count)

# ─── ADMIN API: USER MANAGEMENT ──────────────────────────────────────────────────
@app.route('/admin/user/<int:uid>/approve', methods=['POST'])
@require_admin
def admin_approve_user(uid):
    conn = get_db()
    conn.execute("UPDATE users SET status='approved' WHERE id=? AND is_admin=0",(uid,))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':'User approved successfully'})

@app.route('/admin/user/<int:uid>/block', methods=['POST'])
@require_admin
def admin_block_user(uid):
    conn = get_db()
    conn.execute("UPDATE users SET status='blocked' WHERE id=? AND is_admin=0",(uid,))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':'User blocked'})

@app.route('/admin/user/<int:uid>/delete', methods=['DELETE'])
@require_admin
def admin_delete_user(uid):
    conn = get_db()
    conn.execute('DELETE FROM users WHERE id=? AND is_admin=0',(uid,))
    conn.execute('DELETE FROM watchlist WHERE user_id=?',(uid,))
    conn.execute('DELETE FROM prediction_history WHERE user_id=?',(uid,))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':'User deleted'})

@app.route('/admin/user/add', methods=['POST'])
@require_admin
def admin_add_user():
    data = request.json
    username=data.get('username','').strip(); email=data.get('email','').strip()
    password=data.get('password',''); is_admin=int(data.get('is_admin',0))
    if not all([username,email,password]):
        return jsonify({'success':False,'message':'All fields required'})
    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    try:
        conn = get_db()
        conn.execute("INSERT INTO users (username,email,password,is_admin,status) VALUES (?,?,?,?,'approved')",
                     (username,email,hashed,is_admin))
        conn.commit(); conn.close()
        return jsonify({'success':True,'message':'User added and approved'})
    except sqlite3.IntegrityError:
        return jsonify({'success':False,'message':'Email or username already exists'})

@app.route('/admin/users/list')
@require_admin
def admin_list_users():
    conn = get_db()
    users = conn.execute('SELECT id,username,email,phone,broker,status,is_admin,created_at FROM users ORDER BY created_at DESC').fetchall()
    conn.close()
    return jsonify([dict(u) for u in users])

# ─── ADMIN API: MACRO DATA ────────────────────────────────────────────────────────
@app.route('/admin/macro/add', methods=['POST'])
@require_admin
def admin_add_macro():
    data = request.json
    year = int(data.get('year', datetime.now().year))
    entry_date = data.get('entry_date', datetime.now().strftime('%Y-%m-%d'))
    now = datetime.now().isoformat()
    conn = get_db()
    conn.execute('''INSERT INTO macro_data
        (year,entry_date,GDP_Growth,Inflation,Interest_Rate,Exchange_Rate,M3_Growth,IIP,notes,updated_by,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
        (year, entry_date, float(data.get('GDP_Growth',0)), float(data.get('Inflation',0)),
         float(data.get('Interest_Rate',0)), float(data.get('Exchange_Rate',0)),
         float(data.get('M3_Growth',0)), float(data.get('IIP',0)),
         data.get('notes',''), session['admin_id'], now))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':f'Macro data for {year} saved on {entry_date}'})

@app.route('/admin/macro/<int:mid>/update', methods=['POST'])
@require_admin
def admin_update_macro(mid):
    data = request.json
    now = datetime.now().isoformat()
    conn = get_db()
    conn.execute('''UPDATE macro_data SET year=?,entry_date=?,GDP_Growth=?,Inflation=?,
        Interest_Rate=?,Exchange_Rate=?,M3_Growth=?,IIP=?,notes=?,updated_by=?,updated_at=?
        WHERE id=?''',
        (int(data.get('year',2024)), data.get('entry_date',''),
         float(data.get('GDP_Growth',0)), float(data.get('Inflation',0)),
         float(data.get('Interest_Rate',0)), float(data.get('Exchange_Rate',0)),
         float(data.get('M3_Growth',0)), float(data.get('IIP',0)),
         data.get('notes',''), session['admin_id'], now, mid))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':'Macro entry updated'})

@app.route('/admin/macro/<int:mid>/delete', methods=['DELETE'])
@require_admin
def admin_delete_macro(mid):
    conn = get_db()
    conn.execute('DELETE FROM macro_data WHERE id=?',(mid,))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':'Macro entry deleted'})

@app.route('/admin/macro/list')
@require_admin
def admin_list_macro():
    conn = get_db()
    rows = conn.execute('SELECT * FROM macro_data ORDER BY year DESC, updated_at DESC').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ─── ADMIN API: STOCKS ───────────────────────────────────────────────────────────
@app.route('/admin/stock/add', methods=['POST'])
@require_admin
def admin_add_stock():
    data = request.json; symbol = data.get('symbol','').strip().upper()
    if not symbol: return jsonify({'success':False,'message':'Symbol required'})
    conn = get_db()
    if conn.execute('SELECT id FROM custom_stocks WHERE symbol=?',(symbol,)).fetchone():
        conn.close(); return jsonify({'success':False,'message':'Stock already exists'})
    price = float(data.get('price',0))
    conn.execute('''INSERT INTO custom_stocks
        (symbol,name,sector,mcap,exchange,price,change_val,pct_change,today_high,today_low,
         high_52w,low_52w,prev_close,term,fno,is_active,added_by)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)''',
        (symbol,data.get('name',symbol),data.get('sector','Other'),data.get('mcap','Large Cap'),
         data.get('exchange','NSE'),price,float(data.get('change_val',0)),float(data.get('pct_change',0)),
         float(data.get('today_high',price)),float(data.get('today_low',price)),
         float(data.get('high_52w',price)),float(data.get('low_52w',price)),
         float(data.get('prev_close',price)),data.get('term','Long-term'),
         int(data.get('fno',0)),session['admin_id']))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':f'Stock {symbol} added'})

@app.route('/admin/stock/<int:sid>/update', methods=['POST'])
@require_admin
def admin_update_stock(sid):
    data = request.json; price = float(data.get('price',0))
    conn = get_db()
    conn.execute('''UPDATE custom_stocks SET name=?,sector=?,mcap=?,exchange=?,price=?,change_val=?,
        pct_change=?,today_high=?,today_low=?,high_52w=?,low_52w=?,prev_close=?,term=?,fno=?,is_active=?
        WHERE id=?''',
        (data.get('name',''),data.get('sector','Other'),data.get('mcap','Large Cap'),
         data.get('exchange','NSE'),price,float(data.get('change_val',0)),
         float(data.get('pct_change',0)),float(data.get('today_high',price)),
         float(data.get('today_low',price)),float(data.get('high_52w',price)),
         float(data.get('low_52w',price)),float(data.get('prev_close',price)),
         data.get('term','Long-term'),int(data.get('fno',0)),int(data.get('is_active',1)),sid))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':'Stock updated'})

@app.route('/admin/stock/<int:sid>/delete', methods=['DELETE'])
@require_admin
def admin_delete_stock(sid):
    conn = get_db()
    conn.execute('DELETE FROM custom_stocks WHERE id=?',(sid,))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':'Stock removed'})

@app.route('/admin/stocks/list')
@require_admin
def admin_list_stocks():
    conn = get_db()
    rows = conn.execute('SELECT * FROM custom_stocks ORDER BY added_at DESC').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ─── ADMIN API: SETTINGS ─────────────────────────────────────────────────────────
@app.route('/admin/settings/update', methods=['POST'])
@require_admin
def admin_update_settings():
    data = request.json
    conn = get_db()
    for key,value in data.items():
        conn.execute('INSERT OR REPLACE INTO admin_settings (setting_key,setting_value,updated_at) VALUES (?,?,?)',
                     (key,str(value),datetime.now().isoformat()))
    conn.commit(); conn.close()
    return jsonify({'success':True,'message':'Settings saved'})

@app.route('/admin/settings/list')
@require_admin
def admin_list_settings():
    conn = get_db()
    rows = conn.execute('SELECT setting_key,setting_value FROM admin_settings').fetchall()
    conn.close()
    return jsonify({r['setting_key']:r['setting_value'] for r in rows})

# ─── PUBLIC APIs ─────────────────────────────────────────────────────────────────
@app.route('/api/stock/<symbol>')
def api_stock(symbol):
    return jsonify(get_stock_data(symbol))

@app.route('/api/macro')
def api_macro():
    return jsonify(get_macro_data_db())

@app.route('/api/macro/by-year')
def api_macro_by_year():
    """Return latest macro values per year + last_updated — used by Manual Prediction."""
    macro_by_year, last_updated = get_latest_macro_from_db()
    return jsonify({'by_year': {str(k): v for k,v in macro_by_year.items()},
                    'last_updated': last_updated})

@app.route('/api/stocks')
def api_stocks():
    return jsonify(get_all_stocks_db())

@app.route('/api/regression/panel', methods=['POST'])
@require_login
def api_panel_regression():
    data = request.json; symbols = data.get('symbols',[]); effect = data.get('effect','both')
    result = run_panel_regression(symbols, effect)
    return jsonify(result)

@app.route('/api/regression/ols', methods=['POST'])
@require_login
def api_ols_regression():
    return jsonify(run_ols_regression(request.json.get('symbol','RELIANCE')))

@app.route('/api/diagnostics', methods=['POST'])
@require_login
def api_diagnostics():
    return jsonify(run_diagnostic_tests(request.json.get('symbol','RELIANCE')))

@app.route('/api/predict/manual', methods=['POST'])
@require_login
def api_manual_predict():
    data = request.json
    symbol = data.get('symbol','RELIANCE'); year = data.get('year',2024)
    stock_price = float(data.get('stock_price',0)) or 1000.0
    selected_vars = data.get('selected_vars', MACRO_VARS)
    custom_values = data.get('custom_values', None)

    # Use DB macro values if no custom override
    if not custom_values:
        macro_by_year, _ = get_latest_macro_from_db()
        yr_data = macro_by_year.get(int(year), {})
        if yr_data:
            custom_values = {v: yr_data.get(v, 0) for v in selected_vars if v in yr_data}

    result = manual_predict(symbol, year, stock_price, selected_vars, custom_values)
    if 'error' not in result:
        conn = get_db()
        conn.execute('''INSERT INTO prediction_history
            (user_id,symbol,input_price,predicted_price,trend,confidence,year_used,macro_snapshot,model_type)
            VALUES (?,?,?,?,?,?,?,?,?)''',
            (session['user_id'], symbol, stock_price, result['predicted_price'],
             result['trend'], result['confidence'], year,
             json.dumps(result.get('macro_values_used',{})), 'Fixed Effects Panel Regression'))
        conn.commit(); conn.close()
    return jsonify(result)

@app.route('/api/history')
@require_login
def api_history():
    conn = get_db()
    rows = conn.execute('SELECT * FROM prediction_history WHERE user_id=? ORDER BY created_at DESC LIMIT 100',(session['user_id'],)).fetchall()
    conn.close()
    history = []
    for r in rows:
        macro = {}
        try: macro = json.loads(r['macro_snapshot']) if r['macro_snapshot'] else {}
        except: pass
        history.append({'id':r['id'],'symbol':r['symbol'],'input_price':r['input_price'],
                        'predicted_price':r['predicted_price'],'trend':r['trend'],
                        'confidence':r['confidence'],'year_used':r['year_used'],
                        'macro_snapshot':macro,'model_type':r['model_type'],'created_at':r['created_at']})
    return jsonify(history)

@app.route('/api/history/delete/<int:hid>', methods=['DELETE'])
@require_login
def api_history_delete(hid):
    conn = get_db()
    conn.execute('DELETE FROM prediction_history WHERE id=? AND user_id=?',(hid,session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'success':True})

@app.route('/api/ai/chat', methods=['POST'])
@require_login
def api_ai_chat():
    data = request.json
    return jsonify(get_ai_response(data.get('query',''), data.get('context',{})))

@app.route('/api/ai/predict', methods=['POST'])
@require_login
def api_ai_predict():
    return jsonify(predict_stock_movement(request.json.get('symbol','RELIANCE')))

@app.route('/api/watchlist/add', methods=['POST'])
@require_login
def add_watchlist():
    symbol = request.json.get('symbol')
    conn = get_db()
    if not conn.execute('SELECT id FROM watchlist WHERE user_id=? AND symbol=?',(session['user_id'],symbol)).fetchone():
        conn.execute('INSERT INTO watchlist (user_id,symbol) VALUES (?,?)',(session['user_id'],symbol))
        conn.commit()
    conn.close(); return jsonify({'success':True})

@app.route('/api/watchlist/remove', methods=['POST'])
@require_login
def remove_watchlist():
    symbol = request.json.get('symbol')
    conn = get_db()
    conn.execute('DELETE FROM watchlist WHERE user_id=? AND symbol=?',(session['user_id'],symbol))
    conn.commit(); conn.close(); return jsonify({'success':True})

@app.route('/api/theme', methods=['POST'])
@require_login
def api_set_theme():
    theme = request.json.get('theme','dark')
    session['theme'] = theme
    conn = get_db()
    conn.execute('UPDATE users SET theme=? WHERE id=?',(theme,session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'success':True,'theme':theme})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
