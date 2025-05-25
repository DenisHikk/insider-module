import os
import signal
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from loguru import logger
from insider_service import InsiderService
from image_processor import ImageProcessor
from config import SCHEDULE_INTERVAL, TEMP_DIR
from db.database import SessionLocal, init_db
from models import Employee, Screenshot

class ScreenshotProcessor:
    def __init__(self):
        self.running = True
        self.session = SessionLocal()
        
        self.insider_service = InsiderService(self.session)
        self.image_processor = ImageProcessor()
        
        os.makedirs(TEMP_DIR, exist_ok=True)

        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Обработка сигналов завершения"""
        logger.info("Получен сигнал завершения. Завершаем работу...")
        self.running = False

    def process_employee_screenshots(self, employee: Employee):
        """Обработка скриншотов для одного сотрудника"""
        try:
            logger.info(f"Начало обработки скриншотов для сотрудника {employee.insider_id}")
            
            # Получение скриншотов
            screenshots = self.insider_service.get_screenshots(
                employee.insider_id,
                start_date=datetime.now() - timedelta(days=1)
            )
            
            if not screenshots:
                logger.warning(f"Не найдено скриншотов для сотрудника {employee.insider_id}")
                return
                
            logger.info(f"Получено {len(screenshots)} скриншотов для обработки")
            processed_count = 0
            
            for screenshot_data in screenshots:
                if not self.running:
                    break
                    
                try:
                    file_id = screenshot_data['data']['file']
                    logger.debug(f"Обработка скриншота {file_id}")
                    
                    # Проверка на дубликаты по insider_id
                    existing = self.session.query(Screenshot).filter_by(
                        insider_id=file_id
                    ).first()
                    if existing:
                        logger.debug(f"Скриншот {file_id} уже существует в базе")
                        continue
                    
                    # Скачивание скриншота
                    temp_path = os.path.join(TEMP_DIR, f"{file_id}.jpg")
                    if not self.insider_service.download_screenshot(file_id, temp_path):
                        logger.warning(f"Не удалось скачать скриншот {file_id}")
                        continue
                    
                    # Вычисление хеша изображения
                    image_hash = self.image_processor.calculate_image_hash(temp_path)
                    if not image_hash:
                        logger.warning(f"Не удалось вычислить хеш для скриншота {file_id}")
                        continue
                        
                    # Проверка на дубликаты по хешу
                    duplicate = self.session.query(Screenshot).filter_by(
                        image_hash=image_hash
                    ).first()
                    if duplicate:
                        logger.debug(f"Скриншот {file_id} является дубликатом {duplicate.insider_id}")
                        os.remove(temp_path)
                        continue
                    
                    # Обработка изображения
                    processed_text = self.image_processor.process_image(temp_path)
                    if not processed_text:
                        logger.warning(f"Не удалось обработать скриншот {file_id}")
                        os.remove(temp_path)
                        continue
                    logger.debug("Before get size")
                    # Получение размера файла
                    file_size = os.path.getsize(temp_path)
                    logger.debug("Before save to DB")
                    # Сохранение в базу
                    screenshot = Screenshot(
                        insider_id=file_id,
                        employee_id=employee.id,
                        file_path=temp_path,
                        processed_text=processed_text,
                        image_hash=image_hash,
                        file_size=file_size
                    )
                    self.session.add(screenshot)
                    self.session.commit()
                    
                    # Удаление временного файла
                    try:
                        os.remove(temp_path)
                    except OSError as e:
                        logger.warning(f"Не удалось удалить временный файл {temp_path}: {e}")
                    
                    processed_count += 1
                    logger.debug(f"Успешно обработан скриншот {file_id}")
                    
                except KeyError as e:
                    logger.error(f"Ошибка в структуре данных скриншота: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Ошибка при обработке скриншота: {e}")
                    continue
                
            logger.info(f"Обработано {processed_count} скриншотов для сотрудника {employee.insider_id}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке скриншотов сотрудника {employee.insider_id}: {e}")
            self.session.rollback()

    def process_all_employees(self):
        """Обработка скриншотов для всех сотрудников"""
        try:
            logger.info("Начало обработки всех сотрудников")
            # Синхронизация списка сотрудников
            self.insider_service.sync_employees()
            
            # Получение всех сотрудников
            employees = self.session.query(Employee).all()
            logger.info(f"Найдено {len(employees)} сотрудников для обработки")
            
            for employee in employees:
                if not self.running:
                    break
                self.process_employee_screenshots(employee)
            
            # Очистка старых скриншотов
            self.insider_service.cleanup_old_screenshots()
            
            # Очистка временных файлов
            self.image_processor.cleanup()
            
            logger.info("Завершена обработка всех сотрудников")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке всех сотрудников: {e}")

    def run(self):
        """Запуск обработчика скриншотов"""
        logger.info("Запуск обработчика скриншотов")
        self.process_all_employees()
        self.insider_service.cleanup_old_screenshots()
        logger.info("Завершена обработка всех сотрудников")
        
        # Вывод результатов обработки скриншотов
        self.print_processed_screenshots()

    def print_processed_screenshots(self):
        """Вывод результатов обработки скриншотов"""
        try:
            screenshots = self.session.query(Screenshot).all()
            if not screenshots:
                logger.info("Нет обработанных скриншотов в базе данных.")
                return
            logger.info("Результаты обработки скриншотов:")
            for screenshot in screenshots:
                logger.info(f"ID: {screenshot.insider_id}, Текст: {screenshot.processed_text}, Дата обработки: {screenshot.processed_at}")
        except Exception as e:
            logger.error(f"Ошибка при выводе результатов обработки скриншотов: {e}")