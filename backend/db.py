from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Load DB credentials from environment variables or config
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'kacem123')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'plates_project')

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PlateRecord(Base):
    __tablename__ = 'plate_records'
    id = Column(Integer, primary_key=True, index=True)
    plate_text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PlateRecord(id={self.id}, plate_text='{self.plate_text}', timestamp='{self.timestamp}')>"

# New table for car info
class CarInfo(Base):
    __tablename__ = 'car_info'
    id = Column(Integer, primary_key=True, index=True)
    plate_text = Column(String, nullable=False, unique=True)
    car_model = Column(String, nullable=False)
    car_type = Column(String, nullable=False)
    car_class = Column(String, nullable=True)  # New column for car class

    def __repr__(self):
        return f"<CarInfo(id={self.id}, plate_text='{self.plate_text}', car_model='{self.car_model}', car_type='{self.car_type}', car_class='{self.car_class}')>"

def init_db():
    Base.metadata.create_all(bind=engine)

# Helper for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
