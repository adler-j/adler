import demandimport
with demandimport.enabled():
    import tensorflow as tf


def conv1d(x, W, stride=1, padding='SAME'):
    with tf.name_scope('conv1d'):
        return tf.nn.conv1d(x, W,
                            stride=stride,
                            padding=padding)


def conv1dtransp(x, W, stride=1, out_shape=None, padding='SAME'):
    with tf.name_scope('conv1dtransp'):
        x_shape = tf.shape(x)
        W_shape = tf.shape(W)
        if out_shape is None:
            out_shape = tf.stack([x_shape[0],
                                  1,
                                  stride * x_shape[1],
                                  W_shape[1]])
        else:
            out_shape = tf.stack([out_shape[0],
                                  1,
                                  out_shape[1],
                                  out_shape[2]])

        x_reshaped = tf.expand_dims(x, 1)
        W_reshaped = tf.expand_dims(W, 0)
        strides = [1, 1, stride, 1]

        result = tf.nn.conv2d_transpose(x_reshaped, W_reshaped,
                                        output_shape=out_shape,
                                        strides=strides,
                                        padding=padding)

        return tf.squeeze(result, axis=1)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    with tf.name_scope('conv2d'):
        strides = [1, stride[0], stride[1], 1]
        if padding in ('SAME', 'VALID'):
            return tf.nn.conv2d(x, W,
                                strides=strides, padding=padding)
        else:
            paddings = [[0, 0],
                        [1, 1],
                        [1, 1],
                        [0, 0]]
            x = tf.pad(x, paddings=paddings, mode=padding)

            return tf.nn.conv2d(x, W,
                                strides=strides, padding='VALID')


def conv2dtransp(x, W, stride=(1, 1), out_shape=None, padding='SAME'):
    with tf.name_scope('conv2dtransp'):
        x_shape = tf.shape(x)
        W_shape = tf.shape(W)
        if out_shape is None:
            out_shape = tf.stack([x_shape[0],
                                  stride[0] * x_shape[1],
                                  stride[1] * x_shape[2],
                                  W_shape[2]])

        return tf.nn.conv2d_transpose(x, W,
                                      output_shape=out_shape,
                                      strides=[1, stride[0], stride[1], 1],
                                      padding=padding)


def maxpool1d(x, stride=2, padding='SAME'):
    with tf.name_scope('maxpool1d'):
        ksize = [1, 1, stride, 1]
        strides = [1, 1, stride, 1]

        x_pad = tf.expand_dims(x, 1)
        result = tf.nn.max_pool(x_pad, ksize, strides, padding)
        return tf.squeeze(result, axis=1)


def maxpool2d(x, stride=(2, 2), padding='SAME'):
    with tf.name_scope('maxpool2d'):
        ksize = [1, stride[0], stride[1], 1]
        strides = [1, stride[0], stride[1], 1]
        return tf.nn.max_pool(x, ksize, strides, padding)


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
