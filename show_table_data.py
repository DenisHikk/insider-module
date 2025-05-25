from sqlalchemy import create_engine
from config import DATABASE_URL, INSIDER_API_URL, INSIDER_API_KEY
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, create_engine
from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from models import Employee, Screenshot
from db.database import SessionLocal

### Для просмотра результата БД ### 
app = FastAPI()
templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_screenshots(request: Request, db: Session = Depends(get_db)):
    screenshots = db.query(Screenshot).order_by(Screenshot.id.desc()).limit(100).all()
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "data": screenshots, 
        "url": INSIDER_API_URL,
        "key": INSIDER_API_KEY
        })

