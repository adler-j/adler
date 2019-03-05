import demandimport
with demandimport.enabled():
    import tensorflow as tf

import numpy as np


__all__ = ('cosine_decay', 'ema_wrapper', 'EMAHelper')


def cosine_decay(learning_rate, global_step, maximum_steps,
                 name=None):
  """
  """
  from tensorflow.python.ops import math_ops
  from tensorflow.python.framework import ops

  if global_step is None:
    raise ValueError("global_step is required for cosine_decay.")
  with ops.name_scope(name, "CosineDecay",
                      [learning_rate, global_step, maximum_steps]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    maximum_steps = math_ops.cast(maximum_steps, dtype)

    p = tf.mod(global_step / maximum_steps, 1)

    return learning_rate * (0.5 + 0.5 * math_ops.cos(p * np.pi))


class EMAHelper(object):
    def __init__(self, decay=0.99, session=None):
        if session is None:
            self.session = tf.get_default_session()
        else:
            self.session = session

        self.all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.ema = tf.train.ExponentialMovingAverage(decay=decay)
        self.apply = self.ema.apply(self.all_vars)
        self.averages = [self.ema.average(var) for var in self.all_vars]

    def average_dict(self):
        ema_averages_results = self.session.run(self.averages)
        return {var: value for var, value in
                zip(self.all_vars, ema_averages_results)}

    def variables_to_restore(self):
        return self.ema.variables_to_restore(tf.moving_average_variables())


def ema_wrapper(is_training, decay=0.99, scope='ema_wrapper', reuse=False):
    """Use Exponential Moving Average of weights during testing.

    Parameters
    ----------
    is_training : bool or `tf.Tensor` of type bool
        Indicates if the EMA should be applied or not
    decay:

    Examples
    --------
    During training, the current value of a is used. During testing, the
    exponential moving average is applied instead.

    >>> @ema_wrapper(is_training)
    ... def function(x):
    ....    a = tf.get_variable('a', [], tf.float32)
    ...     return a * x
    """
    def function(fun):
        def fun_wrapper(*args, **kwargs):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                # Regular call
                with tf.variable_scope('function_call') as sc:
                    result_train = fun(*args, **kwargs)

                # Set up exponential moving average
                ema = tf.train.ExponentialMovingAverage(decay=decay)
                var_class = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              sc.name)
                ema_op = ema.apply(var_class)

                # Add to collection so they are updated
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

                # Getter for the variables with EMA applied
                def ema_getter(getter, name, *args, **kwargs):
                    var = getter(name, *args, **kwargs)
                    ema_var = ema.average(var)
                    return ema_var if ema_var else var

                # Call with EMA applied
                with tf.variable_scope('function_call',
                                       reuse=True,
                                       custom_getter=ema_getter):
                    result_test = fun(*args, **kwargs)

                # Return the correct version depending on if we're training or
                # not
                return tf.cond(is_training,
                               lambda: result_train, lambda: result_test)
        return fun_wrapper
    return function