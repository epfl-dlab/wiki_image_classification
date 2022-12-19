import tensorflow as tf
from keras import backend

epsilon = 1e-7


def get_custom_loss(alpha_weights):
    @tf.function
    def custom_loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.cast(y_pred, y_pred.dtype)

        bce = y_true * tf.math.log(y_pred + epsilon) * alpha_weights
        bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon)

        return -backend.mean(bce)
    return custom_loss
