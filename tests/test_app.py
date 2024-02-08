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
    mock_data = [{'column1': 'value1'}, {'column1': 'value2'}]  # Пример мокированных данных
    with patch('path.to.your.load_data_function', return_value=mock_data):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_data?offset=0&limit=10")
            assert response.status_code == 200
            assert response.json() == mock_data

@pytest.mark.asyncio
async def test_load_model():
    # Создаем мок модели, который может быть любым объектом
    mock_model = MagicMock(name='MockModel')
    
    # Сериализуем мок модели в байты, чтобы имитировать реальный ответ
    model_bytes = BytesIO()
    joblib.dump(mock_model, model_bytes)
    model_bytes.seek(0)
    
    # Мокируем функцию load_model, чтобы она возвращала сериализованную мок модель
    with patch('path.to.your.load_model_function', return_value=model_bytes.getvalue()):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_model")
            assert response.status_code == 200
