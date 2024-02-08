import pytest
from unittest.mock import MagicMock, patch
from aioresponses import aioresponses
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
import joblib
from myapp import app, load_data, load_model
from fastapi.testclient import TestClient




@pytest.mark.asyncio
async def test_load_data():
    # Мокируем запрос к внешнему URL для получения данных
    mock_url = "https://github.com/MarinaSupiot/fast_api/raw/main/test_preprocess_reduit.csv.zip"
    with aioresponses.aioresponses() as m:
        m.get(mock_url, status=200, body=b'mock zip content')

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_data?offset=0&limit=10")
        assert response.status_code == 200
        # Проверьте структуру ответа, если нужно

@pytest.mark.asyncio
async def test_load_model():
    # Мокируем запрос к внешнему URL для получения модели
    mock_url = "https://raw.githubusercontent.com/MarinaSupiot/fast_api/main/model_su04.pkl"
    with aioresponses.aioresponses() as m:
        m.get(mock_url, status=200, body=b'mock model content')

        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_model")
        assert response.status_code == 200
        # Проверьте содержимое ответа, если это необходимо

