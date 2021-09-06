import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn, static_bidirectional_rnn
from tensorflow.contrib.rnn import LSTMCell


def Lstm(input_tensor):
    """
    Apply Lstm layer.
    :param input_tensor: `Tensor` with shape of [batch, time_steps, input_size]
    :return: `Tensor` with shape of [batch, input_size]
    """
    input_size = input_tensor.shape[2]
    time_steps = input_tensor.shape[1]
    input_tensor = tf.transpose(input_tensor, [1, 0, 2])
    input_tensor = tf.reshape(input_tensor, [-1, input_size])
    input_tensor = tf.split(axis=0, value=input_tensor, num_or_size_splits=time_steps)
    lstm_cell = LSTMCell(num_units=input_size)
    outputs, state = static_rnn(lstm_cell, input_tensor)  # outputs [time_steps, batch, input_size]
    output = tf.slice(outputs, [time_steps-1, 0, 0], [1, -1, -1])
    tf.squeeze(output)
    return outputs[-1]


def BiLstm(input_tensor):
    """
    Apply Bi-Direction Lstm Layer.
    :param input_tensor: `Tensor` with shape of [batch, time_steps, input_size]
    :return: `Tensor` with shape of [batch, input_size * 2]
    """
    input_size = input_tensor.shape[2]
    time_steps = input_tensor.shape[1]
    input_tensor = tf.transpose(input_tensor, [1, 0, 2])
    input_tensor = tf.reshape(input_tensor, [-1, input_size])
    input_tensor = tf.split(axis=0, value=input_tensor, num_or_size_splits=time_steps)
    lstm_fw_cell = LSTMCell(num_units=input_size)
    lstm_bw_cell = LSTMCell(num_units=input_size)
    outputs, fw_state, bw_state = static_bidirectional_rnn(
        lstm_fw_cell, lstm_bw_cell, input_tensor)
    return outputs[-1]



