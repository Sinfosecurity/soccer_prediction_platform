import pytest
from unittest.mock import patch, AsyncMock
from app.main import add

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

@pytest.fixture
def mock_db():
    with patch('app.models.database.Session') as mock:
        yield mock

@pytest.mark.asyncio
async def test_update_matches(mock_db):
    from app.services.odds_service import OddsService
    service = OddsService()
    # Mock the fetch_odds method to return sample data
    with patch.object(service, 'fetch_odds', return_value=AsyncMock(return_value=[{
        'id': 'match1',
        'home_team': 'Team A',
        'away_team': 'Team B',
        'commence_time': '2025-02-06T12:00:00Z',
        'bookmakers': [{'markets': [{'outcomes': [{'name': 'Team A', 'price': 1.5}, {'name': 'Team B', 'price': 2.5}]}]}]
    }])):
        await service.update_matches(mock_db)
        assert mock_db.commit.called
