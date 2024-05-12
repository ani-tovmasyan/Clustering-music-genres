# Clustering Music Genres

This project conducts experiments on clustering different music genres using data from the Covers80 dataset.

## Dataset

Access the dataset used in this project through the following link: [Covers80 Dataset](http://labrosa.ee.columbia.edu/projects/coversongs/covers80/).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have Git and Conda installed on your system to manage the repository and handle the project environment.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ani-tovmasyan/Clustering-music-genres.git
   cd Clustering-music-genres
   ```
2. **Install Dependencies**

    Make sure you are using ```python==3.9```

    ```bash
    pip install -r requirements.txt
    ```

### Usage
Run the script with the following command, substituting <dataset_path> and <song_path> with the paths to your dataset and song files:

```bash
python3 song_clustering.py <dataset_path> <song_path>
```

