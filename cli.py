#!/usr/bin/env python3
"""
CaptchaKraken CLI - Simple entry point
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from captchakraken.cli import main

if __name__ == "__main__":
    main()
