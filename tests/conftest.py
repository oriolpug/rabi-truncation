"""Shared sys.path setup so all test files can import from src/."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
