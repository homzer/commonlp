import collections
import re

import tensorflow as tf


def filter_compatible_params(ckpt_dir):
    """
    Filter the same parameters both in `trainable params` and `checkpoint params`.
    For example:
        in trainable params:
        <tf.Variable 'A:0' shape=(21128, 768) dtype=float32_ref>
        <tf.Variable 'B:0' shape=(512, 768) dtype=float32_ref>
        <tf.Variable 'C:0' shape=(768,) dtype=float32_ref>
        in checkpoint params:
        ('F', [21128, 768])
        ('B', [512, 768])
        ('D', [512, 768])
        ('C', [768])
        result:
        [<tf.Variable 'B:0' shape=(512, 768) dtype=float32_ref>,
        <tf.Variable 'C:0' shape=(768,) dtype=float32_ref>]
        Because only `C` and `B` are both shared by `trainable params` and `checkpoint params`
    :param ckpt_dir: directory of checkpoint.
    :return: list of tf.Variable
    """
    ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
    reader = tf.train.NewCheckpointReader(ckpt_state.model_checkpoint_path)
    ckpt_vars = reader.get_variable_to_shape_map()
    ckpt_vars = sorted(ckpt_vars.items(), key=lambda x: x[0])
    train_vars = tf.trainable_variables()
    compatible_vars = []
    for ckpt_var in ckpt_vars:
        if type(ckpt_var) is str:
            ckpt_var_name = ckpt_var
        elif type(ckpt_var) is tuple:
            ckpt_var_name = ckpt_var[0]
        else:
            raise ValueError("Unknown checkpoint type: %s" % type(ckpt_var))
        for train_var in train_vars:
            train_var_name = re.match("^(.*):\\d+$", train_var.name).group(1)
            if train_var_name == ckpt_var_name:
                compatible_vars.append(train_var)
                break
    return compatible_vars


def print_param(name):
    """ Print the parameters in the graph. """
    with tf.Session() as sess:
        param = sess.graph.get_tensor_by_name(name)
        print(sess.run(param))


class CheckpointHelper:
    def __init__(self, ckpt_file):
        self.ckpt_file = ckpt_file

    def print_variables(self):
        reader = tf.train.NewCheckpointReader(self.ckpt_file)
        var_dict = reader.get_variable_to_shape_map()
        var_dict = sorted(var_dict.items(), key=lambda x: x[0])
        for item in var_dict:
            if 'adam' in item[0] or 'Adam' in item[0]:
                continue
            print(item)

    def init_from_checkpoint(self):
        """ Restore parameters from checkpoint. """
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = self.get_assignment_map_from_checkpoint(tvars)
        tf.train.init_from_checkpoint(self.ckpt_file, assignment_map)

    def get_assignment_map_from_checkpoint(self, tvars):
        """Compute the union of the current variables and checkpoint variables."""
        initialized_variable_names = {}

        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var

        init_vars = tf.train.list_variables(self.ckpt_file)

        assignment_map = collections.OrderedDict()
        for x in init_vars:
            (name, var) = (x[0], x[1])
            if name not in name_to_variable:
                continue
            assignment_map[name] = name
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1

        return assignment_map, initialized_variable_names

    def rename_variables(self, save_ckpt_file="./result/model.ckpt"):
        sess = tf.Session()
        imported_meta = tf.train.import_meta_graph(self.ckpt_file + '.meta')
        imported_meta.restore(sess, self.ckpt_file)
        name_to_variable = collections.OrderedDict()
        for var in tf.global_variables():
            new_name = var.name
            if 'self/' in new_name:
                new_name = new_name.replace("self/", '')
            if 'bert/' in new_name:
                new_name = new_name.replace('bert/', '')
            if ':0' in new_name:
                new_name = new_name.replace(':0', '')
            name_to_variable[new_name] = var
        saver = tf.train.Saver(name_to_variable)
        saver.save(sess, save_ckpt_file)
