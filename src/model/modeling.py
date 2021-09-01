import os

import numpy as np
import tensorflow as tf

from src.optimizer.adam import AdamOptimizer
from src.utils.CheckpointUtil import filter_compatible_params
from src.utils.variable_util import get_variable_name

__all__ = ['Model']


class Model(object):

    def __init__(self, features, labels, model_graph):
        """
        Building Model
        :param features: `tf.placeholder`, e.g. tf.placeholder("int32", [None, seq_length])
        :param labels: `tf.placeholder`, e.g. self.labels = tf.placeholder("int32", [None, ])
        :param model_graph: a function defining the graph of model,
        which should take `features` and `labels` as input,
        and take `predictions`, `loss`, `variables` as return, where `variables` can be optional,
        which can be set to `None`.
         updated_var_list: list of variables being updated during backward propagation.
        Default to `None`, represents to update all trainable variables.
        """
        self._features = features
        self._labels = labels
        self._model_graph = model_graph
        self._sess = tf.Session()
        self._predictions = None
        self._loss = None
        self._variables = None
        self._global_step = None
        self._optimizer = None
        self._update_var_list = None
        self._build_graph()

    def _build_graph(self):
        print("Building model graph......")
        self._predictions, self._loss, self._variables = self._model_graph(
            self._features, self._labels)
        print("Building model graph complete!")

    def compile(self):
        """ Compile model structure. """
        self._global_step = tf.Variable(
            initial_value=0, trainable=False, name='global_step', dtype=tf.int32)
        self._optimizer = AdamOptimizer(1e-5).minimize(
            self._loss, self._global_step, self._update_var_list)
        self._sess.run(tf.global_variables_initializer())

    def freeze(self, var_name_list):
        """
        Not apply back-propagation for var in var_list
        :param var_name_list: list or tuple of `Variable`'s name.
        for example:
        var_list = ['embeddings/word_embeddings', 'embeddings/position_embeddings']
        """
        tvars = tf.trainable_variables()
        self._update_var_list = []
        for tvar in tvars:
            tvar_name = get_variable_name(tvar)
            for frozen_name in var_name_list:
                if frozen_name in tvar_name:
                    print("Frozen variables: %s" % tvar_name)
                    continue
            self._update_var_list.append(tvar)

    def train(self, features, labels):
        """ Training model """
        self._sess.run(
            self._optimizer,
            feed_dict={
                self._features: features,
                self._labels: labels})
        global_step = self._sess.run(self._global_step)
        return global_step

    def evaluate(self, features, labels):
        """ Evaluate model """
        loss = self._sess.run(self._loss, feed_dict={self._features: features, self._labels: labels})
        predicts = self._sess.run(self._predictions, feed_dict={self._features: features, self._labels: labels})
        labels = np.array(labels)
        predicts = np.array(predicts)
        labels = np.reshape(labels, [-1, ])
        predicts = np.reshape(predicts, [-1, ])
        accuracy = np.mean([int(label) == int(predict) for label, predict in zip(labels, predicts)])
        return loss, accuracy

    def predict(self, features, labels):
        """ Prediction """
        predicts = self._sess.run(self._predictions, feed_dict={self._features: features, self._labels: labels})
        return predicts

    def validate(self, features, labels):
        """ Return the real value of some variables. """
        variables = None if self._variables is None else self._sess.run(
            self._variables, feed_dict={self._features: features, self._labels: labels})
        return variables

    def save(self, step=0, output_dir="result"):
        """ Save model parameters """
        tvars = tf.trainable_variables()
        saver = tf.train.Saver(tvars)
        save_path = os.path.join(output_dir, "model.ckpt-" + str(step))
        print("Saving model to %s......" % save_path)
        saver.save(self._sess, save_path)
        print("Saving model complete!")

    def restore(self, checkpoint_file="config/model.ckpt"):
        print("Restore model from %s......" % checkpoint_file)
        # Filtering compatible parameters
        rvars = filter_compatible_params(checkpoint_file)
        print("\n".join([str(var.name) for var in rvars]))
        saver = tf.train.Saver(rvars)
        saver.restore(self._sess, checkpoint_file)
        print("Restoring model complete!")

    def __delete__(self, instance):
        self._sess.close()
        print("Session closed!")
