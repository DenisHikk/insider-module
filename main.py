import schedule
import time
import sys
from loguru import logger
from db.database import init_db
from screenshot_processor import ScreenshotProcessor

# Настройка логирования
logger.add("screenshot_processor.log", rotation="1 day", retention="7 days", level="INFO")

if __name__ == "__main__":
    try:
        init_db()
        processor = ScreenshotProcessor()
        processor.run()
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске программы: {e}")
        sys.exit(1)
