#!/usr/bin/env python3
"""
Startup script for the FastAPI backend server
"""
import subprocess
import sys
import os

def main():
    """Launch the FastAPI backend server"""
    try:
        # Change to the backend directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Run uvicorn with the FastAPI app
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ]
        
        print("Starting BroadAxis FastAPI Backend...")
        print("API will be available at: http://localhost:8000")
        print("API Documentation: http://localhost:8000/docs")
        print("WebSocket endpoint: ws://localhost:8000/ws/chat")
        print("Press Ctrl+C to stop the server")
        print("-" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nShutting down the backend server...")
    except Exception as e:
        print(f"Error starting backend server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
