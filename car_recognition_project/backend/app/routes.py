from fastapi import FastAPI, UploadFile, File
import shutil
from app.predict import predict_car

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict_car(file_path)
    return {"prediction": prediction}
