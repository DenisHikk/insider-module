# -сервис для работы с API ИНСАЙДЕР
import requests
from datetime import datetime, timedelta
from loguru import logger
from config import INSIDER_API_URL, INSIDER_API_KEY, BATCH_SIZE
from models import Employee, Screenshot
from sqlalchemy.orm import Session
import os

class InsiderService:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.base_url = INSIDER_API_URL
        self.api_key = INSIDER_API_KEY
        
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Выполняет запрос к API ИНСАЙДЕР"""
        if params is None:
            params = {}
        params['key'] = self.api_key
        
        try:
    
            if endpoint == '/api/activities/find':
                response = requests.post(f"{self.base_url}{endpoint}", params=params)
            else:
                response = requests.get(f"{self.base_url}{endpoint}", params=params)
            
            response.raise_for_status()
            result = response.json()
            logger.debug(f"API Response for {endpoint}: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к API {endpoint}: {e}")
            return None

    def get_employees(self) -> list:
        """Получает список сотрудников"""
        response = self._make_request('/api/users/find')
        if not response or 'data' not in response:
            return []
        
        employees = []
        for emp_data in response['data']:
            employee = Employee(
                insider_id=str(emp_data['id']),
                email=emp_data.get('email'),
                department=emp_data.get('department'),
                position=emp_data.get('position')
            )
            employees.append(employee)
        
        return employees

    def get_screenshots(self, user_id: str, start_date: datetime = None, end_date: datetime = None) -> list:
        """Получает список скриншотов для сотрудника"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=1)
        if not end_date:
            end_date = datetime.now()
            
        params = {
            'usersId': user_id,
            'startDate': start_date.strftime('%Y-%m-%d %H:%M:%S'),
            'endDate': end_date.strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'screen',
            'limit': BATCH_SIZE
        }
        
        
        logger.info(f"Запрос скриншотов для пользователя {user_id} с параметрами: {params}")
        response = self._make_request('/api/activities/find', params)
        
        if not response:
            logger.error("Получен пустой ответ от API")
            return []
            
        if 'data' not in response:
            logger.error(f"Неожиданный формат ответа API: {response}")
            return []
            
        screenshots = response['data']
        logger.info(f"Получено {len(screenshots)} скриншотов для пользователя {user_id}")
        return screenshots

    def download_screenshot(self, file_id: str, save_path: str) -> bool:
        """Скачивает скриншот по его ID"""
        try:
            response = requests.get(
                f"{self.base_url}/api/resources/get",
                params={'id': file_id, 'key': self.api_key},
                stream=True
            )
            response.raise_for_status()
            folder_path = os.path.join("image", f"screenshot_{file_id}.jpg")
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Ошибка при скачивании скриншота {file_id}: {e}")
            return False

    def sync_employees(self):
        """Синхронизирует список сотрудников с базой данных"""
        employees = self.get_employees()
        for emp in employees:
            existing = self.db.query(Employee).filter_by(insider_id=emp.insider_id).first()
            if existing:
                existing.email = emp.email
                existing.department = emp.department
                existing.position = emp.position
            else:
                self.db.add(emp)
        self.db.commit()
        logger.info(f"Синхронизировано {len(employees)} сотрудников")

    def cleanup_old_screenshots(self):
        """Удаляет устаревшие скриншоты из базы данных"""
        old_date = datetime.now() - timedelta(days=30)
        old_screenshots = self.db.query(Screenshot).filter(
            Screenshot.created_at < old_date
        ).all()
        
        for screenshot in old_screenshots:
            if screenshot.file_path and os.path.exists(screenshot.file_path):
                os.remove(screenshot.file_path)
            self.db.delete(screenshot)
        
        self.db.commit()
        logger.info(f"Удалено {len(old_screenshots)} устаревших скриншотов") 