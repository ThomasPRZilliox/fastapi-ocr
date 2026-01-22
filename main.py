from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from core import ocr_pipeline, load_models

digit_model,operator_model = load_models()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve the HTML file
@app.get("/")
async def read_root():
    return FileResponse("frontend.html")

# Your OCR endpoint
@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)

    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    result = ocr_pipeline(img, digit_model=digit_model,operator_model=operator_model)
    return {"text": str(result)}
