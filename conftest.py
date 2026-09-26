import os
import sys

# Ensure the inner VisonAssist package is on the PYTHONPATH for pytest imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "VisonAssist"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
