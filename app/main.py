import os
from app.api import endpoints
from app.models.database import engine, Base

def add(a, b):
    return a + b

def new_feature_function():
    return "new feature"

if os.getenv('TESTING') != 'true':
    Base.metadata.create_all(bind=engine)
