import os
import random
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from song_clustering import ClusterMaker

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the model and encoder
encoder_path = "./encoder"
model_path = "./model"

matcher = ClusterMaker(encoder_path=encoder_path, model_path=model_path, build_hpcp=False, read_database=False)

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
    "dataset_url": "http://labrosa.ee.columbia.edu/projects/coversongs/covers80/",
    "generated_image_path": "/static/picture.jpg",
    "dataset_description": """
        As described on the main coversongs page, we have been researching automatic detection 
        of "cover songs" i.e., alternative performances of the same basic musical piece by 
        different artists, typically with large stylistic and/or harmonic changes. The 
        covers80 dataset is a collection of 80 songs, each performed by 2 artists.
    """
}

app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Function to get a random song path from a class directory
def get_random_song_path(class_name):
    songs_dir = os.path.join("static/covers32k", class_name)
    print(songs_dir)
    print(os.path.exists(songs_dir))
    print('***************************')
    if os.path.exists(songs_dir) and os.path.isdir(songs_dir):
        songs = [f for f in os.listdir(songs_dir) if os.path.isfile(os.path.join(songs_dir, f))]
        if songs:
            return os.path.join("static/covers32k", class_name, random.choice(songs))
    return None

# Endpoint for making predictions
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, audio_file: UploadFile = File(...)):
    contents = await audio_file.read()
    temp_file_path = "temp_audio_file.mp3"  # Save the uploaded file temporarily
    with open(temp_file_path, "wb") as f:
        f.write(contents)
    
    # Ensure matcher returns a list of class names sorted by similarity
    similar_classes = matcher(temp_file_path)  # This should now return a list of names
    
    # Clean up temporary files
    os.remove(temp_file_path)
    if os.path.exists(temp_file_path.replace("mp3", "npy")):
        os.remove(temp_file_path.replace("mp3", "npy"))

    # Generate the list of class names and their corresponding song paths
    similar_classes_with_paths = []
    for class_name in similar_classes:
        song_path = get_random_song_path(class_name)
        if song_path:
            similar_classes_with_paths.append((class_name, song_path))
    
    print(similar_classes_with_paths)
    print(similar_classes)
    print('================================================')
    # Pass the list of similar classes and song paths to the template
    return templates.TemplateResponse("prediction_result.html", {
        "request": request,
        "similar_classes": similar_classes_with_paths
    })

# Additional setup for running the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
