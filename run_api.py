#!/usr/bin/env python3
"""
Entry point for Flask REST API server
Run: python run_api.py
"""

import sys
from pathlib import Path

# Add interfaces to path
sys.path.insert(0, str(Path(__file__).parent / "interfaces"))

# Import and run API server
from api_server import app, initialize_chatbot

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Medical Chatbot REST API Server"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )

    args = parser.parse_args()

    # Initialize chatbot
    if initialize_chatbot():
        print(f"Starting API server on http://{args.host}:{args.port}")
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
        )
    else:
        print("Failed to start server: Model loading failed")
        sys.exit(1)
