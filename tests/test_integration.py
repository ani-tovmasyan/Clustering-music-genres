from fastapi.testclient import TestClient

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from main import app

client = TestClient(app)

def test_predict_endpoint_with_audio_file():
    audio_file_path = "./covers80_test/coversongs/covers32k/Addicted_To_Love/robert_palmer+Riptide+03-Addicted_To_Love.mp3"

    response = client.post("/predict", files={"audio_file": open(audio_file_path, "rb")})

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]