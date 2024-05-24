import sys
from datetime import datetime
from song_clustering import ClusterMaker

def main( encoder_path, model_path, query_audio_path):
    matcher = ClusterMaker(encoder_path=encoder_path, model_path=model_path, build_hpcp=False, read_database = False)

    matcher.load_model()
    matcher.load_encoder_and_covers()

    now = datetime.now()
    most_similar_class_name = matcher(query_audio_path)
    print(f'Most similar class: {most_similar_class_name}')
    print(f'Time: {datetime.now() - now}')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python inference_script.py <encoder_path> <model_path> <query_audio_path>")
        sys.exit(1)

    encoder_path = sys.argv[1]
    model_path = sys.argv[2]
    query_audio_path = sys.argv[3]

    main(encoder_path, model_path, query_audio_path)
