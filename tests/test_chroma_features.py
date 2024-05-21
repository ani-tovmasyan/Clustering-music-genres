import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.chroma_features import ChromaFeatures

def test_chroma_hpcp():
    dataset_path = "./test_data"

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".mp3"):
                audio_file = os.path.join(root, file)
                print("Processing:", audio_file)

                chroma = ChromaFeatures(audio_file)
                hpcp_features = chroma.chroma_hpcp()

                assert isinstance(hpcp_features, np.ndarray)
                assert hpcp_features.shape[1] == 12  # Assuming HPCP output has 12 bins