import pytest
from unittest.mock import MagicMock, patch
from aioresponses import aioresponses
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
import joblib
from myapp import app, load_data, load_model
from fastapi.testclient import TestClient
from httpx import AsyncClient 
from unittest.mock import patch
import pickle



@pytest.mark.asyncio
async def test_load_model():
    mock_model_data = {"status": "success", "message": "Model loaded successfully"}
    mock_serialized_model_data = pickle.dumps(mock_model_data)

    with patch('myapp.load_model', return_value=mock_serialized_model_data):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_model")
            assert response.status_code == 200

            # Десериализуем ответ для проверки
            response_data = pickle.loads(response.content)
            
            # Добавляем проверку типа и вывод десериализованных данных для отладки
            assert isinstance(response_data, dict), "Response data is not a dictionary"
            print("Deserialized response data:", response_data)

            assert response_data == mock_model_data, f"Expected {mock_model_data}, got {response_data}"
