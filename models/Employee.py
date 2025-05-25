from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

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