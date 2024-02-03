import pytest
from httpx import AsyncClient
from myapp import app  # Предполагается, что myapp - это ваш модуль FastAPI
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from myapp import load_data, load_model
import aioresponses

# Фикстуры для тестирования
@pytest.fixture
def test_app():
    return app

@pytest.fixture
async def test_client(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client

# Юнит-тесты с использованием mocker
@pytest.mark.asyncio
async def test_load_data_function():
    with aioresponses() as m:
        mock_url = "https://github.com/MarinaSupiot/fast_api/raw/main/test_preprocess_reduit.csv.zip"
        m.get(mock_url, status=200, body=b'fake_zip_bytes')
        
        data = await load_data(0, 100)
        assert isinstance(data, pd.DataFrame)
        
@pytest.mark.asyncio
async def test_load_model_function():
    with aioresponses() as m:
        mock_url = "https://raw.githubusercontent.com/MarinaSupiot/fast_api/main/model_su04.pkl"
        m.get(mock_url, status=200, body=b'fake_model_bytes')
        
        model = await load_model()
        assert model is not None

# Интеграционные тесты
@pytest.mark.asyncio
async def test_get_load_data_endpoint(test_client):
    response = await test_client.get("/load_data?offset=0&limit=100")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_get_load_model_endpoint(test_client):
    response = await test_client.get("/load_model")
    assert response.status_code == 200
