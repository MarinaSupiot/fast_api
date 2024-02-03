import pytest
from httpx import AsyncClient
from myapp import app  # Предполагается, что myapp - это ваш модуль FastAPI
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

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
    # Использование patch и MagicMock для мокирования aiohttp сессии
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_get.return_value.__aenter__.return_value.read = MagicMock(return_value=b'fake_zip_bytes')
        data = await load_data(0, 100)
        assert isinstance(data, pd.DataFrame)

@pytest.mark.asyncio
async def test_load_model_function():
    # То же самое для load_model
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_get.return_value.__aenter__.return_value.read = MagicMock(return_value=b'fake_model_bytes')
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
