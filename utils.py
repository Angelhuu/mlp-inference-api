import pickle
from mlp import MLP

def save_model(mlp, path='my_mlp.pkl'):
    data = {
        'layers': mlp.d.tolist(),
        'weights': mlp.W
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_model(path='my_mlp.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    mlp = MLP(data['layers'])
    mlp.W = data['weights']
    return mlp