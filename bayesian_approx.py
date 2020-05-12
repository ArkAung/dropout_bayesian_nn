from network import Network
from tensorflow.keras import backend as K
from tqdm import tqdm
import numpy as np


def run_bayesian_approx(net: Network, iterations: int, input_data, enable_dropout=True):
    model = net.model
    dropout_nn = K.function([model.layers[0].input, K.learning_phase()],
                            [model.layers[-1].output])
    pred_bayes_dist = []
    if enable_dropout:
        learning_phase = 1
    else:
        learning_phase = 0
    for _ in tqdm(range(0, iterations)):
        pred_bayes_dist.append(dropout_nn([input_data, learning_phase]))

    pred_bayes_dist = np.transpose(np.vstack(pred_bayes_dist), (1, 0, 2))
    return pred_bayes_dist
