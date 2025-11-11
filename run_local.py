#!/usr/bin/env python
"""
Local development runner for Flask app
This script sets up the correct paths and runs the app
"""
import os
import sys

# Get project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add src to path
sys.path.insert(0, os.path.join(project_root, 'src'))

# Change to project root directory (so models load from here)
os.chdir(project_root)

# Set environment variable to tell Flask where templates are
os.environ['FLASK_APP'] = os.path.join(project_root, 'src', 'app.py')

# Import and run the app
from app import app

if __name__ == '__main__':
    print("Starting Flask server for local development...")
    print(f"Project root: {project_root}")
    print(f"Templates: {os.path.join(project_root, 'templates')}")
    print("Server running on http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    app.run(debug=True, host='127.0.0.1', port=5000)

