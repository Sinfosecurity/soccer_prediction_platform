import pytest
from app.main import new_feature_function

def test_new_feature():
    result = new_feature_function()
    assert result == "new feature"
