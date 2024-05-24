import os
import sys
import numpy as np
import joblib
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
        path = audio_path.replace('.wav', '.npy').replace('.mp3', '.npy')
        np.save(path, c_hpcp)
    return c_hpcp



def qmax(hpcp1, hpcp2):
    # Compute cross recurrent plot from two chroma audio feature vectors
    sim_matrix = cross_recurrent_plot(hpcp1, hpcp2)
    #Computing qmax cover song audio similarity measure (distance)
    qmax, _ = qmax_measure(sim_matrix)

    return qmax


class ClusterMaker:
    def __init__(self, encoder_path, model_path, covers_database= None, n_neighbors=20, metric='euclidean', build_hpcp=False, read_database = False):
        self.covers_database_path = covers_database
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.model_path = model_path
        self.encoder_path = encoder_path 

        if covers_database is not None:
            if build_hpcp:
                self.build_hpcp()
            self.read_database()
        else:
            self.load_encoder_and_covers()
        self.metric = metric
        if build_hpcp:
            self.build_hpcp()
        #self.read_database()

    def build_hpcp(self):
        for dirpath, dirnames, filenames in os.walk(self.covers_database_path):
            for file in filenames:
                if file.endswith('.wav') or file.endswith('.mp3')  :
                    filepath = os.path.join(dirpath, file)
                    if  not os.path.exists(filepath.replace('.wav', '.npy').replace('.mp3', '.npy')):
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
        self.encoder = encoder
        if self.metric=='qmax':
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=qmax)
        else:
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
        self.knn.fit(self.covers['data'], self.covers['y'])
        self.save_encoder_and_covers()
        self.save_model()

    def save_model(self):
        joblib.dump(self.knn, self.model_path)

    def load_model(self):
        self.knn = joblib.load(self.model_path)

    def save_encoder_and_covers(self):
        with open(self.encoder_path, 'wb') as f:
            
            joblib.dump({'encoder': self.encoder, 'covers': self.covers}, f)

    def load_encoder_and_covers(self):
        with open(self.encoder_path, 'rb') as f:
            data = joblib.load(f)
            self.encoder = data['encoder']
            self.covers = data['covers']

    def get_cluster(self):  
        probabilities = self.knn.predict_proba(self.query['hpcp'].reshape(1, -1))
        indexes = np.argsort(probabilities[0])[::-1][-self.n_neighbors:]

        return indexes

    def __call__(self, audio_path):
        hpcp_path = audio_path.replace('.wav', '.npy').replace('.mp3', '.npy')
        self.query = {'hpcp': None}

        if not os.path.exists(hpcp_path):
            self.query['hpcp'] = get_hpcp(audio_path)
        else:
            self.query['hpcp'] = np.load(hpcp_path)
        
        if self.query['hpcp'].shape[0] < 2585:
            return -1
        self.query['hpcp'] = self.query['hpcp'][:2585].flatten()

        # Get the indices of clusters and ensure you have enough to find 5 unique classes
        cluster_indices = self.get_cluster()
        unique_classes = []
        unique_indices = []
        
        for idx in cluster_indices:
            class_index = self.covers['y'][idx]
            if class_index not in unique_classes:
                unique_classes.append(class_index)
                unique_indices.append(idx)
            if len(unique_classes) == 5:
                break
        
        # Use the unique indices to fetch the class names
        most_similar_class_names = self.encoder.inverse_transform(unique_classes)

        return most_similar_class_names  




if __name__ == '__main__':
    #argv1 - dataset directory
    #argv2 - query song 
    matcher = ClusterMaker( sys.argv[2], sys.argv[3], sys.argv[1])
    now = datetime.now()
    print(matcher(sys.argv[4]))
    print(f'Time: {datetime.now()-now}')