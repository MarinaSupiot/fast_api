import pytest
from unittest.mock import MagicMock, patch
from aioresponses import aioresponses
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
import joblib
from myapp import app, load_data, load_model
from fastapi.testclient import TestClient


@pytest.fixture
def test_app():
    return app

@pytest.fixture
async def test_client(test_app):
    async with TestClient(app) as client:
        yield client

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

