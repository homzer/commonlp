import os

import numpy as np
import tensorflow as tf

from src.optimizer.adam import AdamOptimizer
from src.utils.checkpoint_util import filter_compatible_params
from src.utils.variable_util import get_variable_name

__all__ = ['Model']


class Model(object):

    def __init__(self, features, labels, model_graph):
        """
        Building Model.

        Before training or prediction or any other op., please run `compile` op first.

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
        returned = self._model_graph(self._features, self._labels)
        if len(returned) == 3:
            self._predictions = returned[0]
            self._loss = returned[1]
            self._variables = returned[2]
        elif len(returned) == 2:
            self._predictions = returned[0]
            self._loss = returned[1]
        else:
            raise ValueError("The return of `model_graph` op must be `list` or `tuple` "
                             "with content: [predicts, loss, variables] or "
                             "[predicts, loss]. Got return value:", returned)
        print("Building model graph complete!")

    def compile(self, learning_rate=1e-4):
        """ Compile model structure. """
        self._global_step = tf.Variable(
            initial_value=0, trainable=False, name='global_step', dtype=tf.int32)
        self._optimizer = AdamOptimizer(learning_rate).minimize(
            self._loss, self._global_step, self._update_var_list)
        self._sess.run(tf.global_variables_initializer())

    def freeze(self, var_name_list):
        """
        Not apply back-propagation for var in var_list
        :param var_name_list: list or tuple of `Variable`'s name, or just
        a part of it.
        for example:
        var_list = ['embeddings/word_embeddings', 'embeddings/position_embeddings']
        You can also write:
        var_list = ['embeddings'], which will freeze all variable names contain str `embeddings`.
        """
        def is_frozen(var):
            for frozen_name in var_name_list:
                if frozen_name in var:
                    return True
            return False

        tvars = tf.trainable_variables()
        self._update_var_list = []
        for tvar in tvars:
            tvar_name = get_variable_name(tvar)
            if is_frozen(tvar_name):
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
        print("\n".join([str(var.name) + ' ' + str(var.shape) for var in rvars]))
        saver = tf.train.Saver(rvars)
        saver.restore(self._sess, checkpoint_file)
        print("Restoring model complete!")

    def auto_train(
            self, train_generator, eval_generator, save_path,
            total_steps=None, eval_step=500, save_step=1000):
        """
        Training model automatically.
        :param train_generator: `DataGenerator` instance.
        :param eval_generator: `DataGenerator` instance.
        :param save_path: where the checkpoint file will be saved.
        :param total_steps: total training steps which you can specify.
        If is None, total steps will be automatically calculated.
        :param eval_step: how often to do the evaluation op.
        :param save_step: how often to save the model.
        """
        if total_steps is None:
            total_steps = train_generator.total_steps
        for step in range(total_steps):
            train_features, train_labels = train_generator.next_batch()
            self.train(train_features, train_labels)
            if step % 100 == 0:
                print("Global Step %d of %d" % (step, total_steps))
            if step % eval_step == 0:
                print("Evaluating......")
                all_train_loss = []
                all_train_acc = []
                all_eval_loss = []
                all_eval_acc = []
                for _ in range(100):
                    eval_features, eval_labels = eval_generator.next_batch()
                    train_features, train_labels = train_generator.next_batch()
                    eval_loss, eval_acc = self.evaluate(eval_features, eval_labels)
                    train_loss, train_acc = self.evaluate(train_features, train_labels)
                    all_eval_loss.append(eval_loss)
                    all_eval_acc.append(eval_acc)
                    all_train_loss.append(train_loss)
                    all_train_acc.append(train_acc)
                eval_features, eval_labels = eval_generator.next_batch()
                predicts = self.predict(eval_features, eval_labels)
                join_str = ' ' if len(str(predicts[0])) == 1 else '\n'
                eval_label_str = join_str.join([str(label) for label in eval_labels])
                pred_label_str = join_str.join([str(pred) for pred in predicts])
                print("************************************")
                print("Eval Loss: ", np.mean(all_eval_loss))
                print("Eval Accuracy: %.2f%%" % (np.mean(all_eval_acc) * 100))
                print("Train Loss: ", np.mean(all_train_loss))
                print("Train Accuracy: %.2f%% " % (np.mean(all_train_acc) * 100))
                print("Evaluate Labels:\t", eval_label_str)
                print("Evaluate Predicts:\t", pred_label_str)
                print("************************************")
            if step % save_step == 0 and step != 0:
                self.save(step, save_path)
        self.save(total_steps, save_path)

    def __del__(self):
        self._sess.close()
        print("Session closed!")
