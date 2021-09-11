from math import log

import tensorflow as tf

from src.utils.tensor_util import gelu, create_initializer, loop_slice


def Conv1D(input_tensor, filter_shape: list, name):
    """
    Apply convolution layer.
    Activation using `gelu`, apply wide-convolution mechanism.
    :param input_tensor: 3-D `Tensor` with shape [batch, length, hidden_size], or
    4-D `Tensor` with shape [batch, length, hidden_size, in_channels]
    :param filter_shape: `List` [width of filter, in_channels, out_channels].
    :param name: scope name of this op.
    :return: `Tensor` with shape [batch, length, hidden_size, out_channels]
    """
    if input_tensor.shape.ndims == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=-1)
    assert input_tensor.shape.ndims == 4
    seq_length = int(input_tensor.shape[1])
    hidden_size = int(input_tensor.shape[2])
    filter_width = int(filter_shape[0])
    in_channels = int(filter_shape[1])
    out_channels = int(filter_shape[2])
    assert in_channels == int(input_tensor.shape[-1])

    # padding
    fw_pad = int((filter_width - 1) / 2)
    bw_pad = int((filter_width - 1) / 2)
    fw_pad += 1 if (filter_width - 1) % 2 != 0 else 0
    input_tensor = tf.pad(
        input_tensor, [[0, 0], [fw_pad, bw_pad], [0, 0], [0, 0]])

    channel_win_size = max(1, int(log(in_channels)))  # multi-channels window size
    channel_stride = max(1, channel_win_size - 1)  # multi-channels step size

    # `B` donates batch         `F` donates filter_width
    # `S` donates length        `H` donates hidden_size
    # `I` donates in_channels   `O` donates out_channels
    # `W` donates channel_win_size
    slice_contents = loop_slice(  # [B, S, F, H, I]
        input_tensor, stride=1, width=filter_width,
        num=seq_length, axis=1)
    slice_channels = loop_slice(  # [B, S, F, H, O, W]
        slice_contents, stride=channel_stride,
        width=channel_win_size, num=out_channels, axis=-1)
    filter_shape = [filter_width, hidden_size, out_channels, channel_win_size]
    filter_weights = tf.get_variable(  # [F, H, O, W]
        name=name+'_W', shape=filter_shape, initializer=create_initializer())
    bias_shape = [out_channels]
    filter_bias = tf.get_variable(  # [O]
        name=name+'_b', shape=bias_shape, initializer=create_initializer())
    filter_weights = tf.expand_dims(filter_weights, axis=0)
    filter_weights = tf.expand_dims(filter_weights, axis=0)  # [1, 1, F, H, O, W]
    result = tf.reduce_sum(  # [B, S, F, H, O]
        tf.multiply(filter_weights, slice_channels), axis=-1)
    result = tf.reduce_sum(result, axis=2)  # [B, S, H, O]
    result = gelu(tf.add(result, filter_bias))
    return result


def MaxPooling1D(input_tensor, pool_size: int, stride=None):
    """
    Max Pooling layer.
    Pooling 2-D `Tensor`. However, higher rank is also supported.
    Specifically, `Tensor` with shape [batch, length, hidden_size, channels]
    Require (length - pool_size) % stride == 0.
    For examples:
    ```python

    x = tf.constant(
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
         5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
        shape=[2, 4, 3])

    MaxPooling1D(x, 2, 2)  # [[[2, 2, 2], [4, 4, 4]],
                              [[6, 6, 6], [8, 8, 8]]]
    ```
    :param input_tensor: `Tensor` with shape [batch, length, ...].
    :param pool_size: `Integer` the width of pooling window.
    :param stride: `Integer` the step of pooling window moving.
    If None, stride = pool_size by default.
    :return: `Tensor` with shape [batch, num_pools, ...],
    where num_pools is the times of pooling, which can be calculated as follow:
    num_pools = (length - pool_size) / stride + 1.
    """
    length = int(input_tensor.shape[1])
    if stride is None:
        stride = pool_size
    assert (length - pool_size) % stride == 0
    num_pools = int((length - pool_size) / stride) + 1
    result = loop_slice(  # [batch, num_pools, pool_size, ...]
        input_tensor, stride=stride,
        width=pool_size, num=num_pools, axis=1)
    return tf.reduce_max(result, axis=2)


def Flatten1D(input_tensor):
    """
    Apply Flatten Layer after 1-D CNN layers,
    reshape the output of `CNN` to the input of `DNN`
    :param input_tensor: 4-D `Tensor` of shape [batch, length, hidden_size, channels]
    :return: [batch_size, length * hidden_size * channels]
    """
    length = input_tensor.shape[1]
    hidden = input_tensor.shape[2]
    channels = input_tensor.shape[3]
    last_dim = length * hidden * channels
    return tf.reshape(input_tensor, [-1, last_dim])
