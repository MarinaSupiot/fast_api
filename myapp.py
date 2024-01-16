from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import pickle
import joblib
from io import BytesIO
from zipfile import ZipFile
import requests

app = FastAPI()

#def load_data():
#    try:
#        df_test = pd.read_csv("/content/drive/MyDrive/Projet 7/test_preprocess_reduit.csv")
#        return df_test
#    except Exception as e:
#        raise ValueError(f"Error loading DataFrame: {str(e)}")



def load_data():
    try:
        # URL к архиву ZIP
        zip_url = "https://github.com/MarinaSupiot/fast_api/raw/main/test_preprocess_reduit.csv.zip"
        
        # Чтение ZIP-архива и извлечение файла
        with ZipFile(BytesIO(requests.get(zip_url).content), 'r') as zip_file:
            # Извлечение файла CSV из архива
            csv_file_name = 'test_preprocess_reduit.csv'
            with zip_file.open(csv_file_name) as file:
                # Чтение CSV файла в DataFrame
                df_test = pd.read_csv(file)

        return df_test
    except Exception as e:
        raise ValueError(f"Error loading DataFrame: {str(e)}")



#def load_model():
#    try:
#        with open('/content/drive/MyDrive/Projet 7/model_su04.pkl', 'rb') as file:
#            model = joblib.load(file)
#        return model
#    except Exception as e:
#        raise ValueError(f"Error loading model: {str(e)}")

def load_model():
    try:
        # URL к вашему файлу модели на GitHub
        model_url = "https://raw.githubusercontent.com/MarinaSupiot/fast_api/main/model_su04.pkl"
        
        # Получение содержимого файла модели по URL
        model_content = requests.get(model_url).content
        
        # Загрузка модели из байтового содержимого
        model = joblib.load(BytesIO(model_content))
        
        return model
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")



@app.get("/load_data")
def get_load_data():
    df_test = load_data()
    return df_test.to_dict(orient='records')


@app.get("/load_model")
def get_load_model():
    model = load_model()
    return {"model": str(model)}





