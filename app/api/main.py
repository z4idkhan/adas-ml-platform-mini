from fastapi import FastAPI, UploadFile, File
import shutil
import os

from app.model.inference import Predictor
from fastapi.responses import RedirectResponse

app = FastAPI(title="ADAS ML Platform Mini API")
predictor = Predictor()



@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")


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