import tensorflow.keras.backend as K
from tensorflow_core.python.keras.layers import merge
import numpy as np


class RandomWeightedAverage(merge._Merge):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def gradient_penalty_loss(y_true, y_pred, interpolated_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, interpolated_samples)[0]

    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def wasserstein(y_true, y_pred):
    return -K.mean(y_true * y_pred)


def set_trainable(m, val):
    m.trainable = val
    for l in m.layers:
        l.trainable = val
