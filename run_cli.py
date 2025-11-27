#!/usr/bin/env python3
"""
Entry point for CLI chatbot
Run the medical chatbot in terminal mode
"""

import sys
from pathlib import Path

# Add interfaces to path
sys.path.insert(0, str(Path(__file__).parent / "interfaces"))

from cli_chatbot import main

if __name__ == "__main__":
    main()
