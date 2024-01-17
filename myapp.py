from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import pickle
import joblib
from io import BytesIO
from zipfile import ZipFile
import aiohttp
from aiohttp import ClientSession

app = FastAPI()

async def load_data(start_row=0, chunk_size=5000):
    try:
        zip_url = "https://github.com/MarinaSupiot/fast_api/raw/main/test_preprocess_reduit.csv.zip"
        
        async with ClientSession() as session:
            async with session.get(zip_url) as response:
                content = await response.read()

        with ZipFile(BytesIO(content), 'r') as zip_file:
            csv_file_name = 'test_preprocess_reduit.csv'
            with zip_file.open(csv_file_name) as file:
                # Читаем только часть данных с использованием параметров skiprows и nrows
                df_test = pd.read_csv(file, skiprows=range(1, start_row), nrows=chunk_size)

        return df_test
    except Exception as e:
        raise ValueError(f"Error loading DataFrame: {str(e)}")


async def load_model():
    try:
        model_url = "https://raw.githubusercontent.com/MarinaSupiot/fast_api/main/model_su04.pkl"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(model_url) as response:
                model_content = await response.read()
        
        model = joblib.load(BytesIO(model_content))
        
        return model
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")


@app.get("/load_data")
async def get_load_data(start_row: int = 0, chunk_size: int = 5000):
    df_test = await load_data(start_row=start_row, chunk_size=chunk_size)
    return df_test.to_dict(orient='records')


@app.get("/load_model")
async def get_load_model():
    model = await load_model()
    return {"model": str(model)}

