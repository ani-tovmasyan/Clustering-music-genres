import pytest
from fastapi.testclient import TestClient
import time
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from main import app

client = TestClient(app)

def test_predict_endpoint_time():
    audio_file_path = "./test_data/Addicted_To_Love/robert_palmer+Riptide+03-Addicted_To_Love.mp3"

    start_time = time.time()
    response = client.post("/predict", files={"audio_file": open(audio_file_path, "rb")})
    end_time = time.time()


    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time} seconds")

    assert response.status_code == 200

    maximum = 5.0
    assert inference_time < maximum, f"Inference time {inference_time} exceeds {maximum} seconds"