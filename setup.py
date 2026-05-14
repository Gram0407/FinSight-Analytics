"""
FinSight Analytics — Setup & Run Script
"""
import subprocess, sys, os

def install_deps():
    print("Installing dependencies...")
    subprocess.check_call([sys.executable,'-m','pip','install','-r','requirements.txt','--break-system-packages','-q'])
    print("✅ Dependencies installed")

def run_app():
    os.environ['FLASK_ENV'] = 'development'
    from app import app, init_db
    init_db()
    print("\n🚀 FinSight Analytics running at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    install_deps()
    run_app()
