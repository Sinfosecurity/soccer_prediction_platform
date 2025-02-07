import httpx
from typing import List, Dict
from datetime import datetime, timedelta
import asyncio
from app.core.config import get_settings
from app.core.logging_config import logger
from app.models.match import Match
from sqlalchemy.orm import Session

settings = get_settings()

class OddsService:
    def __init__(self):
        self.api_key = settings.ODDS_API_KEY
        self.base_url = settings.ODDS_API_URL
        self.request_count = 0
        self.last_request_time = datetime.min

    async def fetch_odds(self, sport: str = "soccer") -> List[Dict]:
        """Fetch latest odds from the API"""
        try:
            # Rate limiting
            if (datetime.now() - self.last_request_time).seconds < 1:
                await asyncio.sleep(1)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/sports/{sport}/odds",
                    params={
                        "apiKey": self.api_key,
                        "regions": "eu",
                        "markets": "h2h"
                    }
                )

                self.last_request_time = datetime.now()
                self.request_count += 1

                if response.status_code != 200:
                    logger.error(f"API request failed: {response.text}")
                    return []

                return response.json()

        except Exception as e:
            logger.error(f"Error fetching odds: {str(e)}")
            return []

    async def update_matches(self, db: Session):
        """Update matches in database with latest odds"""
        try:
            matches_data = await self.fetch_odds()

            for match_data in matches_data:
                match_id = match_data.get('id')
                existing_match = db.query(Match).filter(
                    Match.match_id == match_id
                ).first()

                match_info = {
                    'match_id': match_id,
                    'home_team': match_data.get('home_team'),
                    'away_team': match_data.get('away_team'),
                    'start_time': datetime.fromisoformat(match_data.get('commence_time')),
                    'odds_data': match_data.get('bookmakers', [{}])[0]
                }

                if existing_match:
                    for key, value in match_info.items():
                        setattr(existing_match, key, value)
                else:
                    db.add(Match(**match_info))

            db.commit()
            logger.info(f"Updated {len(matches_data)} matches")

        except Exception as e:
            logger.error(f"Error updating matches: {str(e)}")
            db.rollback()
