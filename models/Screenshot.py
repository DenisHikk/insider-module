from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base


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
