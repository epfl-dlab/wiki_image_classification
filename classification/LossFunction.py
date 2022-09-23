import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import Loss
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops.losses import util as tf_losses_util
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import six

class LossFunctionWrapper(Loss):
  """Wraps a loss function in the `Loss` class."""

  def __init__(self,
               fn,
               name=None,
               **kwargs):
    
    super(LossFunctionWrapper, self).__init__(name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    Returns:
      Loss values per sample.
    """
    if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
      y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(
          y_pred, y_true)
    ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
    return ag_fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class BinaryFocalCrossentropy(LossFunctionWrapper):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, axis=-1):
        super().__init__(fn=binary_focal_crossentropy)
        self.alpha = alpha
        self.gamma = gamma
        self.axis = axis
        self.from_logits = from_logits

def binary_focal_crossentropy(y_true, y_pred, gamma=2, from_logits=True, axis=-1): # batch_size x nr_classes
    y_pred = ops.convert_to_tensor_v2(y_pred)
    if from_logits:
        # Transform logits to probabilities
        def sigmoid(x):
            return 1 / (1 + tf.math.exp(-x))
        y_pred = sigmoid(y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    y_true = math_ops.cast(y_true, y_pred.dtype)
    
    term_1 = y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
    term_0 = (1 - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
    focal_ce = -(term_1 + term_0)

    return K.mean(focal_ce, axis=axis)