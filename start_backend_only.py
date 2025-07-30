#!/usr/bin/env python3
"""
Start only the FastAPI backend server
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the FastAPI backend"""
    try:
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        print("🤖 BroadAxis FastAPI Backend")
        print("=" * 30)
        print("🚀 Starting backend server...")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔗 Health Check: http://localhost:8000/health")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down backend...")
    except Exception as e:
        print(f"❌ Backend error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()