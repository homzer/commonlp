import tensorflow as tf

from src.utils.tensor_util import get_activation, create_initializer


def Dense(input_tensor, units, activation="gelu", name=None):
    return tf.layers.dense(
        inputs=input_tensor, units=units,
        activation=get_activation(activation),
        kernel_initializer=create_initializer(),
        name=name)
