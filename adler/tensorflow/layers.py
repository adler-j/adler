import demandimport
with demandimport.enabled():
    import tensorflow as tf


def conv1d(x, W, stride=1, padding='SAME'):
    with tf.name_scope('conv1d'):
        return tf.nn.conv1d(x, W,
                            stride=stride,
                            padding=padding)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    with tf.name_scope('conv2d'):
        return tf.nn.conv2d(x, W,
                            strides=[1, stride[0], stride[1], 1],
                            padding=padding)


def conv2dtransp(x, W, stride=(1, 1), padding='SAME'):
    with tf.name_scope('conv2dtransp'):
        x_shape = tf.shape(x)
        W_shape = tf.shape(W)
        out_shape = tf.stack([x_shape[0],
                              stride[0] * x_shape[1],
                              stride[1] * x_shape[2],
                              W_shape[2]])
        return tf.nn.conv2d_transpose(x, W,
                                      output_shape=out_shape,
                                      strides=[1, stride[0], stride[1], 1],
                                      padding=padding)


def huber(values, max_grad=1.0):
    """Calculates the Huber function.

    Parameters
    ----------
    values: np.array, tf.Tensor
      Target value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    with tf.name_scope('huber'):
        err = tf.abs(values, name='abs')
        mg = tf.constant(max_grad, name='max_grad')
        lin = mg*(err-.5*mg)
        quad = .5*err*err
        return tf.where(err < mg, quad, lin)
