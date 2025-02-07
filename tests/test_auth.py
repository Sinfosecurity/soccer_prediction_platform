```python
# filepath: /Users/tahirah/soccer_prediction_platform/backend/tests/test_auth.py

import pytest
from app.auth import authenticate_user

def test_authenticate_user():
    user = authenticate_user("username", "password")
    assert user is not None
    assert user["username"] == "username"
