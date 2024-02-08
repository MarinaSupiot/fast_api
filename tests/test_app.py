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
    mock_data = pd.DataFrame({'column1': ['value1', 'value2']})
    with patch('myapp.load_data', return_value=mock_data):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_data?offset=0&limit=10")
            assert response.status_code == 200
            assert response.json() == mock_data.to_dict(orient='records')


@pytest.mark.asyncio
async def test_load_model():
    # Предположим, что функция возвращает сериализованный объект.
    # Мокируем функцию так, чтобы возвращать байты.
    mock_model_response = b"fake_serialized_model_data"
    with patch('myapp.load_model', return_value=mock_model_response):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/load_model")
            assert response.status_code == 200
            # Дополнительно проверяем, соответствует ли тело ответа мокированному
            assert response.content == mock_model_response
