import pytest
from app.ml import predict

def test_predict():
    result = predict({"feature1": 1.0, "feature2": 2.0})
    assert result == {"prediction": "class1"}
