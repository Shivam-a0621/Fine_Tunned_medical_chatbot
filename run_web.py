#!/usr/bin/env python3
"""
Entry point for Streamlit web interface
Run: streamlit run run_web.py
"""

import sys
from pathlib import Path

# Add interfaces to path
sys.path.insert(0, str(Path(__file__).parent / "interfaces"))

# Import and run streamlit app
from web_chatbot import main

if __name__ == "__main__":
    main()
