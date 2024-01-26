import pytest
from httpx import AsyncClient
from app import app  # Импортируйте ваш FastAPI приложение
from aioresponses import aioresponses
import pandas as pd
import numpy as np
from io import BytesIO
import joblib

# Фикстуры для тестирования
@pytest.fixture
def test_app():
    return app

@pytest.fixture
async def test_client(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client

# Юнит-тесты
@pytest.mark.asyncio
async def test_load_data_success():
    with aioresponses() as m:
        mock_url = "https://github.com/MarinaSupiot/fast_api/raw/main/test_preprocess_reduit.csv.zip"
        m.get(mock_url, status=200, body=b'your_csv_data_here')

        response = await load_data(0, 8000)
        assert isinstance(response, pd.DataFrame)

@pytest.mark.asyncio
async def test_load_model_success():
    with aioresponses() as m:
        mock_url = "https://raw.githubusercontent.com/MarinaSupiot/fast_api/main/model_su04.pkl"
        m.get(mock_url, status=200, body=joblib.dump(np.array([1, 2, 3]), BytesIO()))

        model = await load_model()
        assert model is not None

# Интеграционные тесты
@pytest.mark.asyncio
async def test_load_data_endpoint(test_client):
    response = await test_client.get("/load_data?offset=0&limit=100")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@pytest.mark.asyncio
async def test_load_model_endpoint(test_client):
    response = await test_client.get("/load_model")
    assert response.status_code == 200

