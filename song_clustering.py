import os
import sys
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from utils.similarity_measures import * 
from sklearn.pipeline import make_pipeline
from utils.chroma_features import ChromaFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder 

THRESHOLD = 0.08

def get_hpcp(audio_path, save=True):
    """
    Get hpcp feature of audio file

    Args:
    audio_path(str): The path to audio file.
    save(bool): Saves features in .npy file in cas of True

    Returns:
    np.array: hpcp featues
    """
    audio = ChromaFeatures(audio_file=audio_path, mono=True, sample_rate=44100)
    c_hpcp = audio.chroma_hpcp()
    if save:
        path = audio_path.replace('.wav', '.npy')
        np.save(path, c_hpcp)
    return c_hpcp



def qmax(hpcp1, hpcp2):
    # Compute cross recurrent plot from two chroma audio feature vectors
    sim_matrix = cross_recurrent_plot(hpcp1, hpcp2)
    #Computing qmax cover song audio similarity measure (distance)
    qmax, _ = qmax_measure(sim_matrix)

    return qmax


class ClusterMaker:
    def __init__(self, covers_database, n_neighbors=20, build_hpcp=True):
        self.covers_database_path = covers_database
        self.n_neighbors = n_neighbors
        if build_hpcp:
            self.build_hpcp()
        self.read_database()

    def build_hpcp(self):
        for dirpath, dirnames, filenames in os.walk(self.covers_database_path):
            for file in filenames:
                if file.endswith('.wav'):
                    filepath = os.path.join(dirpath, file)
                    if  not os.path.exists(filepath.replace('.wav', '.npy')):
                        get_hpcp(filepath) 

    def read_database(self):
        self.covers = {'data': [],
                        'y' : [],
                        'path':[]}

        for dirpath, dirnames, filenames in os.walk(self.covers_database_path):
            for file in filenames:
                if file.endswith('.npy') and 'reduced' not in file:
                    # Extract class_id from the directory name
                    class_id = os.path.basename(dirpath)

                    embedding = np.load(os.path.join(dirpath, file))
                    if embedding.shape[0]<2585:
                        print(f'skipping {file}')
                        continue
                    embedding = embedding[:2585].flatten()
                    self.covers['data'].append(embedding)
                    self.covers['y'].append(class_id)
                    self.covers['path'].append(os.path.join(dirpath, file))

        # "Fit" on the training labels; this is really just specifying our vocabulary
        encoder = LabelEncoder()
        self.covers['y'] = encoder.fit_transform(self.covers['y'])
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)#, metric='cosine')
        self.knn.fit(self.covers['data'], self.covers['y'])

    def get_cluster(self):
        probabilities = self.knn.predict_proba(self.query['hpcp'].reshape(1, -1))
        indexes = np.argsort(probabilities[0])[::-1][-self.n_neighbors:]

        return indexes

    def __call__(self, audio_path):
        hpcp_path = audio_path.replace('.wav', '.npy').replace('.mp3', '.wav')
        self.query = {'hpcp':None}

        if not os.path.exists(hpcp_path):
            self.query['hpcp'] = get_hpcp(audio_path)
        else:
            self.query['hpcp'] = np.load(hpcp_path)
        if self.query['hpcp'].shape[0]<2585:
            print(f'skipping {audio_path}')
            return -1
        self.query['hpcp'] = self.query['hpcp'][:2585].flatten()

        cluster = self.get_cluster()
        
        return cluster 



if __name__ == '__main__':
    #argv1 - dataset directory
    #argv2 - query song 
    matcher = ClusterMaker(sys.argv[1])
    now = datetime.now()
    print(matcher(sys.argv[2]))
    print(f'Time: {datetime.now()-now}')
