from fastapi import FastAPI, UploadFile, File
import shutil
from app.predict import predict_car
from pathlib import Path

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = Path(f"temp/{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict_car(file_path)
    if isinstance(prediction, str):
        # return {"prediction": []}
        # #jsonify the prediction
        return jsonify({"prediction": prediction})
    

    prediction = [{"label": label, "probability": float(probability)} for label, probability in prediction]
    return {"prediction": prediction}
