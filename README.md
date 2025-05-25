# Сервис обработки скриншотов ИНСАЙДЕР

Сервис для автоматического сбора, обработки и анализа текстовой информации со скриншотов рабочих компьютеров сотрудников из системы ИНСАЙДЕР.

## Функциональность

- Автоматический сбор скриншотов из системы ИНСАЙДЕР
- Распознавание текста с помощью EasyOCR
- Обработка и нормализация текста с помощью Natasha
- Сохранение результатов в базу данных SQLite
- Автоматическая синхронизация данных
- Очистка устаревших данных
- Логирование всех операций

## Требования

- Python 3.11+
- pip
- SQLite3
- build-essential
- python3-dev
- libhunspell-dev

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Создайте виртуальное окружение:
```bash
python3 -m venv venv
```

3. Активируйте виртуальное окружение:
- Linux/MacOS:
```bash
source venv/bin/activate
```

4. Установка зависимостей:
```bash
sudo apt install build-essential python3-dev libhunspell-dev
sudo apt install python3-hunspell 
sudo apt-get install python3-opencv
# Если выбираете tesseract
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-rus
sudo apt install tesseract-ocr-eng
```

5. Установите зависимости:
```bash
pip install -r requirements.txt
```

6. Создайте файл .env в корневой директории проекта:
```env
INSIDER_API_URL=URL
INSIDER_API_KEY=API_KEY
DATABASE_URL=sqlite:///screenshots.db
BATCH_SIZE=20
SCHEDULE_INTERVAL=60
LOG_LEVEL=INFO
TEMP_DIR=temp
```

## Запуск

1. Активируйте виртуальное окружение (если еще не активировано)

2. Запустите сервис:
```bash
python main.py
```

Сервис начнет работу и будет:
- Синхронизировать список сотрудников
- Собирать и обрабатывать скриншоты
- Сохранять результаты в базу данных
- Очищать устаревшие данные

## Структура проекта

- `main.py` - основной файл приложения
- `config.py` - конфигурация приложения
- `/models` - модели базы данных
- `insider_service.py` - сервис для работы с API ИНСАЙДЕР
- `image_processor.py` - сервис для обработки изображений
- `requirements.txt` - зависимости проекта
- `.env` - файл с переменными окружения
- `temp/` - директория для временных файлов
- `screenshots.db` - база данных SQLite
- `show_table_data.py` - сервис для просмотра БД
- `screenshot_processor.py` - основной процесс приложения

## Просмотр базы данных
`vicorn show_table_data:app --host 0.0.0.0 --port 8000 --reload`

## Логирование

Логи сохраняются в файл `app.log` в корневой директории проекта. Уровень логирования можно настроить в файле `.env`.
- `screenshot_processor.log` - все ошибки, возникающие в процессе работы программы, должны фиксироваться в лог-файле

## База данных

База данных SQLite создается автоматически при первом запуске. Структура базы данных:
- Таблица `employees` - информация о сотрудниках
- Таблица `screenshots` - информация о скриншотах и распознанном тексте