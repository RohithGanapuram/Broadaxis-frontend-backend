#!/usr/bin/env python3
"""
Startup script to run the complete BroadAxis RFP/RFQ Management Platform
"""
import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def run_backend():
    """Run the FastAPI backend server"""
    try:
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        print("🚀 Starting FastAPI Backend...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except Exception as e:
        print(f"❌ Backend error: {e}")

def run_frontend():
    """Run the React frontend"""
    try:
        frontend_dir = Path(__file__).parent / "frontend"
        os.chdir(frontend_dir)
        
        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ npm not found. Please install Node.js from https://nodejs.org/")
            print("💡 Alternative: Run backend only with: cd backend && python run_backend.py")
            return
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        print("🚀 Starting React Frontend...")
        subprocess.run(["npm", "run", "dev"])
    except Exception as e:
        print(f"❌ Frontend error: {e}")
        print("💡 You can still use the backend API at http://localhost:8000/docs")

def main():
    """Main startup function"""
    print("🤖 BroadAxis RFP/RFQ Management Platform")
    print("=" * 50)
    print("🔧 Starting all services...")
    print()
    
    # Check if Node.js is available
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        node_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        node_available = False
        print("⚠️  Node.js/npm not found. Starting backend only.")
        print("💡 Install Node.js from https://nodejs.org/ to use the React frontend")
        print("📚 Backend API docs: http://localhost:8000/docs")
        print()
    
    if node_available:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Wait a bit for backend to start
        time.sleep(3)
        
        # Start frontend in main thread
        try:
            run_frontend()
        except KeyboardInterrupt:
            print("\n👋 Shutting down all services...")
            sys.exit(0)
    else:
        # Run backend only
        try:
            run_backend()
        except KeyboardInterrupt:
            print("\n👋 Shutting down backend...")
            sys.exit(0)

if __name__ == "__main__":
    main()