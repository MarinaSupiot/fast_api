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
    with patch('myapp.load_data', return_value=mock_data):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_data?offset=0&limit=10")
            assert response.status_code == 200
            assert response.json() == mock_data

@pytest.mark.asyncio
async def test_load_model():
    # Предопределяем ответ, который будет возвращен мокированной функцией
    # В этом случае, мы используем простой словарь в качестве примера
    mock_model_response = {"status": "success", "message": "Model loaded successfully"}

    # Мокируем функцию load_model, чтобы она возвращала mock_model_response
    with patch('myapp.load_model', return_value=mock_model_response):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_model")
            assert response.status_code == 200
            # Проверяем, что ответ содержит данные, которые мы ожидаем
            assert response.json() == mock_model_response
