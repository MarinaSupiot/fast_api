import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from unittest.mock import AsyncMock
import pandas as pd
from myapp import app  # Импортируйте ваш FastAPI приложение
import numpy as np


@pytest.fixture
def test_app():
    return app

@pytest.fixture
async def test_client(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_load_data_function(mocker):
    # Мокирование aiohttp.ClientSession().get().read() для возврата байтов zip-файла
    mock_get = mocker.patch('aiohttp.ClientSession.get', new_callable=AsyncMock)
    mock_read = AsyncMock(return_value=b'fake_zip_bytes')
    mock_get.return_value.__aenter__.return_value.read = mock_read
    
    # Вызов функции
    data = await load_data(0, 100)
    
    # Проверка, что возвращается DataFrame
    assert isinstance(data, pd.DataFrame)
    # Дополнительные проверки можно добавить в зависимости от структуры данных

@pytest.mark.asyncio
async def test_load_model_function(mocker):
    # Мокирование aiohttp.ClientSession().get().read() для возврата байтов модели
    mock_get = mocker.patch('aiohttp.ClientSession.get', new_callable=AsyncMock)
    mock_read = AsyncMock(return_value=b'fake_model_bytes')
    mock_get.return_value.__aenter__.return_value.read = mock_read
    
    # Вызов функции
    model = await load_model()
    
    # Проверка, что модель загружена (можно проверять тип или ключевые свойства модели)
    assert model is not None


@pytest.mark.asyncio
async def test_get_load_data_endpoint(test_client):
    response = await test_client.get("/load_data?offset=0&limit=100")
    assert response.status_code == 200
    data = response.json()
    # Проверка структуры ответа
    assert isinstance(data, list)  # Предполагаем, что ответ в формате списка словарей

@pytest.mark.asyncio
async def test_get_load_model_endpoint(test_client):
    response = await test_client.get("/load_model")
    assert response.status_code == 200
    # Проверка заголовков ответа для подтверждения отправки файла
    content_disposition = response.headers.get("Content-Disposition")
    assert content_disposition == "attachment; filename=model.pkl"
