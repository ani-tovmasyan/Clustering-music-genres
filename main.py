from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from song_clustering import ClusterMaker
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the model and encoder
encoder_path = "./encoder"
model_path = "./model"

matcher = ClusterMaker(encoder_path=encoder_path, model_path=model_path, build_hpcp=False, read_database = False)

matcher.load_model()
matcher.load_encoder_and_covers()

model_metadata = {
    "algorithm_name": "K-Nearest Neighbors",
    "related_papers": [
        "https://arxiv.org/pdf/2004.04523",
    ],
    "version": "1.0",
    "training_date": "2024-05-19",
    "dataset_used": "Covers80",
    "dataset_url": "http://labrosa.ee.columbia.edu/projects/coversongs/covers80/"  # Adjust URL as appropriate
}

# Endpoint for the main page
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint for model details
@app.get("/model-details", response_class=HTMLResponse)
def get_model_details(request: Request):
    return templates.TemplateResponse("model_details.html", {"request": request, "details": model_metadata})

# Form for prediction
@app.get("/predict-form", response_class=HTMLResponse)
def predict_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

# Endpoint for making predictions
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, audio_file: UploadFile = File(...)):
    contents = await audio_file.read()
    temp_file_path = "temp_audio_file.mp3"  # Ensure this is the intended format and handling
    with open(temp_file_path, "wb") as f:
        f.write(contents)
    prediction = matcher(temp_file_path)  # Assuming matcher now takes raw audio data and sample rate

    import os
    os.remove(temp_file_path)
    os.remove(temp_file_path.replace("mp3", "npy"))

    return templates.TemplateResponse("prediction_result.html", {"request": request, "prediction": prediction})

# Additional setup for running the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
