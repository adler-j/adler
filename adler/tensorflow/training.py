import demandimport
with demandimport.enabled():
    import tensorflow as tf
    
import numpy as np


__all__ = ('cosine_decay',)

    
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