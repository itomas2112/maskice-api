#%% Imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

#%%
load_dotenv(r'.env')

username = os.getenv("usern")
password = os.getenv("password")
host = os.getenv("host")
port = os.getenv("port")
database = os.getenv("database")

connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}?sslmode=require'
engine = create_engine(connection_string,
                       pool_size=5,
                       max_overflow=5,
                       pool_timeout=30,
                       pool_recycle=1800)

### Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()  # ✅ Properly returns connection to pool