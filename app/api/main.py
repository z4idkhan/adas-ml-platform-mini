from fastapi import FastAPI, UploadFile, File
import shutil
import os

from app.model.inference import Predictor

app = FastAPI(title="ADAS ML Platform Mini API")
predictor = Predictor()


@app.get("/")
def root():
    return {"message": "ADAS ML Platform Mini API is running"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    temp_dir = "data/processed"
    os.makedirs(temp_dir, exist_ok=True)

    temp_path = os.path.join(temp_dir, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predictor.predict(temp_path)

    return {
        "filename": file.filename,
        "prediction": prediction
    }