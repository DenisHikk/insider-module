# -модели для базы данных
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from config import DATABASE_URL
from loguru import logger

Base = declarative_base()
engine = create_engine(DATABASE_URL)

class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    insider_id = Column(String(50), unique=True, nullable=False)
    email = Column(String(100))
    department = Column(String(100))
    position = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    screenshots = relationship("Screenshot", back_populates="employee", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_employee_insider_id', 'insider_id'),
    )

class Screenshot(Base):
    __tablename__ = 'screenshots'
    
    id = Column(Integer, primary_key=True)
    insider_id = Column(String(50), unique=True, nullable=False)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    file_path = Column(String(255))
    processed_text = Column(Text)
    image_hash = Column(String(64))  
    file_size = Column(Integer) 
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    employee = relationship("Employee", back_populates="screenshots")
    
    __table_args__ = (
        Index('idx_screenshot_insider_id', 'insider_id'),
        Index('idx_screenshot_employee_id', 'employee_id'),
        Index('idx_screenshot_processed_at', 'processed_at'),
        Index('idx_screenshot_image_hash', 'image_hash'),
    )

def init_db():
    """Инициализация базы данных"""
    try:
        Base.metadata.create_all(engine)
        logger.info("База данных успешно инициализирована")
    except Exception as e:
        logger.error(f"Ошибка при инициализации базы данных: {e}")
        raise 