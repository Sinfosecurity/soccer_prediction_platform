import pytest
from app.services import some_service_function

def test_some_service_function():
    result = some_service_function()
    assert result == "expected result"
