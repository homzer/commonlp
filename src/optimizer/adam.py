import re

import tensorflow as tf

from tensorflow.train import Optimizer


class AdamOptimizer(Optimizer):
    # Values for gate_gradients.
    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 # num_train_steps=None,
                 use_locking=False,
                 name="Adam"):
        super(AdamOptimizer, self).__init__(use_locking, name)
        self._init_lr = learning_rate
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        # self._train_steps = num_train_steps
        # self._warmup_steps = int(num_train_steps * 0.1) if num_train_steps else None

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):

        def _get_variable_name(p_name):
            """Get the variable name from the tensor name."""
            match = re.match("^(.*):\\d+$", p_name)
            if match is not None:
                p_name = match.group(1)
            return p_name

        assignments = [global_step.assign(global_step + 1)] if global_step else []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue
            param_name = _get_variable_name(param.name)
            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            # Standard Adam update.
            next_m = (tf.multiply(self._beta1, m) + tf.multiply(1.0 - self._beta1, grad))
            next_v = (tf.multiply(self._beta2, v) + tf.multiply(1.0 - self._beta2, tf.square(grad)))
            update = next_m / (tf.sqrt(next_v) + self._epsilon)
            update_with_lr = self._lr * update
            next_param = param - update_with_lr
            assignments.extend([param.assign(next_param), m.assign(next_m), v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        learning_rate = tf.constant(value=self._init_lr, shape=[], dtype=tf.float32)
        # # Implements linear decay of the learning rate.
        # if self._train_steps and global_step:
        #     learning_rate = tf.train.polynomial_decay(
        #         learning_rate, global_step, self._train_steps,
        #         end_learning_rate=0.0, power=1.0, cycle=False)
        # # Implements linear warmup.
        # if self._warmup_steps and global_step:
        #     global_steps_int = tf.cast(global_step, tf.int32)
        #     warmup_steps_int = tf.constant(self._warmup_steps, dtype=tf.int32)
        #     global_steps_float = tf.cast(global_steps_int, tf.float32)
        #     warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        #     warmup_percent_done = global_steps_float / warmup_steps_float
        #     warmup_learning_rate = self._init_lr * warmup_percent_done
        #     is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        #     learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        self._lr = learning_rate
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        return self.apply_gradients(zip(grads, tvars), global_step=global_step)

    def _apply_dense(self, grad, var):
        pass

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass
