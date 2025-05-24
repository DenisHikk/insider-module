import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / os.getenv('TEMP_DIR', 'temp')

TEMP_DIR.mkdir(exist_ok=True)

INSIDER_API_URL = os.getenv('INSIDER_API_URL')
INSIDER_API_KEY = os.getenv('INSIDER_API_KEY')

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///screenshots.db')

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 20))
SCHEDULE_INTERVAL = int(os.getenv('SCHEDULE_INTERVAL', 60)) 

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'app.log'

if not all([INSIDER_API_URL, INSIDER_API_KEY]):
    raise ValueError("Необходимо указать INSIDER_API_URL и INSIDER_API_KEY в .env файле")

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
STATIC_FOLDER = "static"
FAVICON_PATH = "static/favicon.png"
FONT_PATH = "arial.ttf"

HOST = "0.0.0.0"
PORT = 5000
RELOAD = False  

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}