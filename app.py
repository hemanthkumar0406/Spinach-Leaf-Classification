"""Root entry point for Hugging Face Spaces / Render."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "webapp"))

from gradio_spinach_app import app

if __name__ == "__main__":
    app.launch()