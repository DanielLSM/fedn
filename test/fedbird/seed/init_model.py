import tempfile
from src.client1_new import TrainingProcess, Model, TrainDataReader

def create_seed_model():
    model = Model()
    data = TrainDataReader()
    start_process = TrainingProcess(data, model)

    return start_process.local_model


def save_model(outer_model, path='package'):
    import tarfile

    _, weights_path = tempfile.mkstemp(suffix='.h5')
    outer_model['model'].save_weights(weights_path)
    tar = tarfile.open(path, "w:gz")
    tar.add(weights_path,'weights.h5')
    tar.add('src', 'src')
    tar.close()

    return path


if __name__ == '__main__':
    outer_model = {}
    outer_model['model'] = create_seed_model()
    outfile_name = "birdcage"
    save_model(outer_model, outfile_name)
    print("seed model saved as: ", outfile_name)