# conftest.py
import sys
import os

# Add src/ to Python path so 'from ingestion.x import' works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))