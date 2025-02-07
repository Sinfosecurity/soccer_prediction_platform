import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from datetime import datetime
from app.core.logging_config import logger

class FeaturePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'home_win_odds', 'draw_odds', 'away_win_odds',
            'home_form', 'away_form',
            'home_goals_scored', 'home_goals_conceded',
            'away_goals_scored', 'away_goals_conceded'
        ]

    def extract_odds(self, odds_data: Dict) -> Tuple[float, float, float]:
        """Extract odds from raw API data"""
        try:
            home_win_odds = odds_data.get('h2h', [{}])[0].get('home_win', 2.0)
            draw_odds = odds_data.get('h2h', [{}])[0].get('draw', 3.0)
            away_win_odds = odds_data.get('h2h', [{}])[0].get('away_win', 2.0)
            
            return home_win_odds, draw_odds, away_win_odds
        except Exception as e:
            logger.error(f"Error extracting odds: {str(e)}")
            return 2.0, 3.0, 2.0  # Default odds if extraction fails

    def preprocess_match_data(
        self,
        odds_data: Dict,
        historical_matches: List[Dict] = None
    ) -> pd.DataFrame:
        """Preprocess match data for prediction"""
        try:
            # Extract odds
            home_win_odds, draw_odds, away_win_odds = self.extract_odds(odds_data)
            
            features = {
                'home_win_odds': home_win_odds,
                'draw_odds': draw_odds,
                'away_win_odds': away_win_odds
            }
            
            # Add historical performance features if available
            if historical_matches:
                home_team = odds_data.get('home_team')
                away_team = odds_data.get('away_team')
                
                features.update(self._calculate_historical_features(
                    home_team, away_team, historical_matches
                ))
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error preprocessing match data: {str(e)}")
            raise

    def _calculate_historical_features(
        self,
        home_team: str,
        away_team: str,
        historical_matches: List[Dict]
    ) -> Dict:
        """Calculate features from historical matches"""
        try:
            recent_matches = [
                m for m in historical_matches
                if (datetime.utcnow() - datetime.fromisoformat(m['start_time'])).days <= 90
            ]
            
            home_form = self._calculate_team_form(home_team, recent_matches)
            away_form = self._calculate_team_form(away_team, recent_matches)
            
            home_goals = self._calculate_goal_stats(home_team, recent_matches)
            away_goals = self._calculate_goal_stats(away_team, recent_matches)
            
            return {
                'home_form': home_form,
                'away_form': away_form,
                'home_goals_scored': home_goals[0],
                'home_goals_conceded': home_goals[1],
                'away_goals_scored': away_goals[0],
                'away_goals_conceded': away_goals[1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating historical features: {str(e)}")
            return {
                'home_form': 0.5,
                'away_form': 0.5,
                'home_goals_scored': 1.5,
                'home_goals_conceded': 1.5,
                'away_goals_scored': 1.5,
                'away_goals_conceded': 1.5
            }

    def _calculate_team_form(self, team: str, matches: List[Dict]) -> float:
        """Calculate team's recent form"""
        team_matches = [
            m for m in matches
            if m['home_team'] == team or m['away_team'] == team
        ]
        
        if not team_matches:
            return 0.5
            
        points = []
        for match in team_matches[-5:]:  # Last 5 matches
            if match['winner'] == team:
                points.append(1.0)
            elif match['winner'] is None:
                points.append(0.5)
            else:
                points.append(0.0)
                
        return sum(points) / len(points) if points else 0.5

    def _calculate_goal_stats(self, team: str, matches: List[Dict]) -> Tuple[float, float]:
        """Calculate team's goal scoring and conceding averages"""
        team_matches = [
            m for m in matches
            if m['home_team'] == team or m['away_team'] == team
        ]
        
        if not team_matches:
            return 1.5, 1.5
            
        goals_scored = []
        goals_conceded = []
        
        for match in team_matches:
            if match['home_team'] == team:
                goals_scored.append(match['home_score'])
                goals_conceded.append(match['away_score'])
            else:
                goals_scored.append(match['away_score'])
                goals_conceded.append(match['home_score'])
                
        return (
            sum(goals_scored) / len(goals_scored) if goals_scored else 1.5,
            sum(goals_conceded) / len(goals_conceded) if goals_conceded else 1.5
        )

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(X[self.feature_columns])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X[self.feature_columns]) 