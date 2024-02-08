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



@pytest.mark.asyncio
async def test_load_data():
    # Создаем мок DataFrame
    mock_df = pd.DataFrame({
        'column1': ['value1', 'value2']
    })

    # Мокируем функцию `load_data` чтобы возвращала mock_df
    with patch('myapp.load_data', return_value=mock_df):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_data?offset=0&limit=10")
            assert response.status_code == 200
            assert response.json() == mock_df.to_dict(orient='records')
            

@pytest.mark.asyncio
async def test_load_model():
    # Создаем мокированные данные модели
    mock_model_data = {"status": "success", "message": "Model loaded successfully"}
    # Сериализуем данные
    mock_serialized_model_data = pickle.dumps(mock_model_data)

    # Мокируем функцию `load_model` чтобы возвращала сериализованные данные
    with patch('myapp.load_model', return_value=mock_serialized_model_data):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_model")
            assert response.status_code == 200
            # Десериализуем ответ для проверки
            response_data = pickle.loads(response.content)
            assert response_data == mock_model_data
