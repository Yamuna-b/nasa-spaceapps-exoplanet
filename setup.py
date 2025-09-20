#!/usr/bin/env python3
"""
Setup script for Exoplanet AI application.
This script initializes the database and creates necessary directories.
"""

import os
import sys
import sqlite3
from pathlib import Path

def create_directories():
    """Create necessary directories for the application."""
    directories = [
        'logs',
        'logs/issue_reports',
        'logs/contact_inquiries',
        'artifacts',
        'data',
        'auth/__pycache__',
        'utils/__pycache__'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def initialize_database():
    """Initialize the SQLite database."""
    from auth.database import DatabaseManager
    
    print("Initializing database...")
    db = DatabaseManager()
    print("✓ Database initialized successfully")

def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✓ Created .env file from .env.example")
            print("⚠️  Please edit .env file with your actual configuration values")
        else:
            print("❌ .env.example file not found")
    else:
        print("✓ .env file already exists")

def check_requirements():
    """Check if all required packages are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import plotly
        import bcrypt
        import jwt
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Exoplanet AI application...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python version: {sys.version}")
    
    # Create directories
    create_directories()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Initialize database
    try:
        initialize_database()
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file with your configuration")
    print("2. Run: streamlit run app.py")
    print("3. Open your browser to http://localhost:8501")

if __name__ == "__main__":
    main()
