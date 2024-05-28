# gunicorn_entry.py
from waitress import serve  # Import the Waitress server

from app.main import app

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)