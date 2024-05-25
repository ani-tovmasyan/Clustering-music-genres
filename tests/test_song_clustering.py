import pytest
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from song_clustering import ClusterMaker, get_hpcp

@pytest.fixture
def audio_path():
    return "./test_audio_files/robert_palmer+Riptide+03-Addicted_To_Love.mp3"

@pytest.fixture
def covers_database_path():
    return "./test_audio_files/covers32k/"

@pytest.fixture
def encoder_path():
    return "./encoder"

@pytest.fixture
def model_path():
    return "./model"

def test_get_hpcp(audio_path):
    hpcp = get_hpcp(audio_path, save=False)
    assert hpcp.shape[0] > 0

def test_cluster_maker(covers_database_path, encoder_path, model_path, audio_path):
    cluster_maker = ClusterMaker(encoder_path, model_path, covers_database_path)

    # Test if ClusterMaker instance is created successfully
    assert isinstance(cluster_maker, ClusterMaker)

    # Test if building HPCP features works
    cluster_maker.build_hpcp()
    files = os.listdir(covers_database_path)
    
    # Filter out only .mp3 and .wav files
    files = [file for file in files if file.endswith('.mp3') or file.endswith('.wav')]
    for path in files:
        assert os.path.exists(audio_path.replace('.wav', '.npy').replace('.mp3', '.npy'))

    # Test reading database
    cluster_maker.read_database()
    assert len(cluster_maker.covers['data']) > 0

    # Test saving and loading encoder and covers
    cluster_maker.save_encoder_and_covers()
    assert os.path.exists(encoder_path)
    cluster_maker.load_encoder_and_covers()
    assert isinstance(cluster_maker.encoder, LabelEncoder)
    assert len(cluster_maker.covers['data']) > 0

    # Test saving and loading model
    cluster_maker.save_model()
    assert os.path.exists(model_path)
    cluster_maker.load_model()
    assert isinstance(cluster_maker.knn, KNeighborsClassifier)

    # Test querying
    most_similar_classes = cluster_maker(audio_path)
    assert isinstance(most_similar_classes, list)
    assert len(most_similar_classes) == 5
    assert len(set(most_similar_classes)) == 5  # Test that all returned classes are unique
