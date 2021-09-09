from math import log

import tensorflow as tf

from src.utils.tensor_util import gelu, create_initializer, loop_slice


# def Conv1D_(input_tensor, filter_width, num_channels, name, padding='SAME'):
#     """
#     Apply convolution layer. With padding = 'SAME'.
#     :param input_tensor: `Tensor` with shape [batch, length, hidden]
#     :param filter_width: `Integer` indicates the width of window.
#     :param num_channels: `Integer` indicates the number of channels.
#     :param name: scope name.
#     :param padding: if padding = 'SAME', using `0` to pad.
#     :return: `Tensor` with shape [batch, num_windows, num_channels],
#     which `num_windows` represents the times of window moves,
#     `num_windows` = length - filter_width + 1 if not using padding,
#     and `num_windows` = length if using padding.
#     """
#     seq_length = int(input_tensor.shape[1])
#     hidden_size = int(input_tensor.shape[2])
#     if padding == 'SAME':
#         fw_pad = int((filter_width - 1) / 2)
#         bw_pad = int((filter_width - 1) / 2)
#         fw_pad += 1 if (filter_width - 1) % 2 != 0 else 0
#         input_tensor = tf.pad(input_tensor, [[0, 0], [fw_pad, bw_pad], [0, 0]])
#     filter_shape = [1, num_channels, filter_width, hidden_size]
#     # `m` donates `num_windows`, `n` donates `num_channels`
#     filter_weight = tf.get_variable(  # [1, n, w, h]
#         name=name + '_W', shape=filter_shape, initializer=create_initializer())
#     filter_bias = tf.get_variable(  # [1, n, h]
#         name=name + '_b', shape=[1, num_channels, hidden_size], initializer=create_initializer())
#     windows = tf.concat([tf.expand_dims(tf.slice(  # [b, m, w, h]
#         input_tensor, [0, i, 0], [-1, filter_width, -1]), axis=1)
#         for i in range(seq_length)], axis=1)
#     filter_weight = tf.expand_dims(filter_weight, axis=1)  # [1, 1, n, w, h]
#     windows = tf.expand_dims(windows, axis=2)  # [b, m, 1, w, h]
#     result = tf.multiply(filter_weight, windows)  # [b, m, n, w, h]
#     result = tf.squeeze(tf.reduce_sum(result, axis=-2))  # [b, m, n, h]
#     filter_bias = tf.expand_dims(filter_bias, axis=1)  # [1, 1, n, h]
#     result = tf.add(result, filter_bias)  # [b, m, n, h]
#     result = gelu(result)
#     return result


def Conv1D(input_tensor, filter_shape, name):
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
    slice_contents = tf.concat(  # [B, S, F, H, I]
        [tf.expand_dims(
            _tensor, axis=1) for _tensor in loop_slice(
            input_tensor, stride=1,
            width=filter_width, num=seq_length,
            axis=1)], axis=1)
    slice_channels = tf.concat(  # [B, S, F, H, O, W]
        [tf.expand_dims(
            _tensor, axis=-2) for _tensor in loop_slice(
            slice_contents, stride=channel_stride,
            width=channel_win_size,
            num=out_channels, axis=-1)], axis=-2)
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


def MaxPooling1D(input_tensor):
    """
    Max Pooling layer.
    For examples:
    ```python

    x = tf.constant(
    [[[1, 3, 5],
    [6, 9, 1],
    [3, 6, 6],
    [1, 0, 9],
    [4, 4, 4]]])  # num_windows = 5, num_channels = 3

    MaxPooling1D(x)  # [[6, 9, 9]]
    ```
    :param input_tensor: `Tensor` with shape [batch, num_windows, num_channels].
    :return: `Tensor` with shape [batch, num_channels]
    """
    input_tensor = tf.transpose(input_tensor, [0, 2, 1])
    return tf.reduce_max(input_tensor, axis=-1)


def MeanPooling1D(input_tensor):
    """
    Average Pooling Layer.
    For examples:
    ```python

    x = tf.constant(
    [[[3, 5, 5],
    [1, 3, 5],
    [1, 1, 5],
    [3, 2, 8],
    [2, 4, 2]]])  # num_windows = 5, num_channels = 3

    MaxPooling1D(x)  # [[2, 3, 5]]
    ```
    :param input_tensor: `Tensor` with shape [batch, num_windows, num_channels].
    :return: `Tensor` with shape [batch, num_channels]
    """
    input_tensor = tf.transpose(input_tensor, [0, 2, 1])
    return tf.reduce_mean(input_tensor, axis=-1)
