import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from app.core.logging_config import logger

class FeatureEngineer:
    def __init__(self):
        self.historical_data = {}  # Cache for historical performance data
        
    def calculate_team_form(self, team: str, matches: List[Dict], reference_date: datetime) -> float:
        """Calculate team form based on recent matches"""
        try:
            recent_matches = [
                m for m in matches 
                if (reference_date - datetime.fromisoformat(m['start_time'])).days <= 90  # Last 90 days
            ]
            
            if not recent_matches:
                return 0.0
                
            form_score = 0
            total_weight = 0
            
            for idx, match in enumerate(recent_matches):
                weight = 1 / (idx + 1)  # More recent matches have higher weight
                if match['winner'] == team:
                    form_score += 3 * weight
                elif match['winner'] == 'draw':
                    form_score += 1 * weight
                total_weight += weight
                
            return form_score / total_weight if total_weight > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating form for {team}: {str(e)}")
            return 0.0
            
    def calculate_goal_stats(self, team: str, matches: List[Dict], is_home: bool) -> tuple:
        """Calculate average goals scored and conceded"""
        try:
            if not matches:
                return 0.0, 0.0
                
            goals_scored = []
            goals_conceded = []
            
            for match in matches:
                team_score = match['home_score'] if is_home else match['away_score']
                opponent_score = match['away_score'] if is_home else match['home_score']
                
                goals_scored.append(team_score)
                goals_conceded.append(opponent_score)
                
            avg_scored = np.mean(goals_scored) if goals_scored else 0
            avg_conceded = np.mean(goals_conceded) if goals_conceded else 0
            
            return avg_scored, avg_conceded
            
        except Exception as e:
            logger.error(f"Error calculating goal stats for {team}: {str(e)}")
            return 0.0, 0.0
            
    def calculate_h2h_dominance(self, home_team: str, away_team: str, matches: List[Dict]) -> float:
        """Calculate head-to-head dominance score"""
        try:
            h2h_matches = [
                m for m in matches 
                if (m['home_team'] == home_team and m['away_team'] == away_team) or
                   (m['home_team'] == away_team and m['away_team'] == home_team)
            ]
            
            if not h2h_matches:
                return 0.0
                
            home_wins = sum(1 for m in h2h_matches if m['winner'] == home_team)
            away_wins = sum(1 for m in h2h_matches if m['winner'] == away_team)
            
            return (home_wins - away_wins) / len(h2h_matches)
            
        except Exception as e:
            logger.error(f"Error calculating H2H dominance: {str(e)}")
            return 0.0
            
    def engineer_features(self, match_data: Dict, historical_matches: List[Dict]) -> pd.DataFrame:
        """Generate all features for a match"""
        try:
            home_team = match_data['home_team']
            away_team = match_data['away_team']
            match_date = datetime.fromisoformat(match_data['commence_time'])
            
            features = {
                'home_win_odds': float(match_data['h2h'][0]),
                'draw_odds': float(match_data['h2h'][1]),
                'away_win_odds': float(match_data['h2h'][2]),
                
                'home_form': self.calculate_team_form(home_team, historical_matches, match_date),
                'away_form': self.calculate_team_form(away_team, historical_matches, match_date),
                
                'h2h_dominance': self.calculate_h2h_dominance(home_team, away_team, historical_matches)
            }
            
            # Add goal statistics
            home_scored, home_conceded = self.calculate_goal_stats(home_team, historical_matches, True)
            away_scored, away_conceded = self.calculate_goal_stats(away_team, historical_matches, False)
            
            features.update({
                'home_goals_scored': home_scored,
                'home_goals_conceded': home_conceded,
                'away_goals_scored': away_scored,
                'away_goals_conceded': away_conceded,
                
                'goal_difference_ratio': (home_scored - away_scored) / (home_scored + away_scored + 1),
                'defense_ratio': (home_conceded - away_conceded) / (home_conceded + away_conceded + 1)
            })
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            raise 