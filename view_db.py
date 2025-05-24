from app.db.database import SessionLocal, Image

def view_database():
    session = SessionLocal()
    try:
        # Получаем все записи
        images = session.query(Image).all()
        
        print("\n=== Содержимое базы данных ===")
        print(f"Всего записей: {len(images)}\n")
        
        for img in images:
            print(f"ID: {img.id}")
            print(f"Файл: {img.filename}")
            print(f"Объекты: {img.objects}")
            print(f"Распознанный текст: {img.recognized_text}")
            print(f"Дата обработки: {img.processed_at}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Ошибка при чтении базы данных: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    view_database() 