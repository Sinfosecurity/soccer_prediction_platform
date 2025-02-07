from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from app.models.database import Base

class Match(Base):
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(String, unique=True, index=True)
    home_team = Column(String, index=True)
    away_team = Column(String, index=True)
    start_time = Column(DateTime, index=True)
    odds_data = Column(JSON)
    prediction = Column(Integer)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "start_time": self.start_time.isoformat(),
            "prediction": self.prediction,
            "confidence": self.confidence,
            "odds_data": self.odds_data
        }

    class Config:
        orm_mode = True 