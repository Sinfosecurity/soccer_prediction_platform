import os
from fastapi import FastAPI
from app.api import endpoints
from app.models.database import engine, Base

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

def add(a, b):
    return a + b

def new_feature_function():
    return "new feature"

if os.getenv('TESTING') != 'true':
    Base.metadata.create_all(bind=engine)
