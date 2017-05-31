import demandimport
with demandimport.enabled():
    import tensorflow as tf


def leaky_relu(_x, alpha=0.2):
    with tf.name_scope('leaky_relu'):
        alphas = tf.Variable(tf.constant(alpha,
                                         shape=[int(_x.get_shape()[-1])]),
                             trainable=False)

        pos = tf.nn.relu(_x)
        neg = alphas * (_x - tf.abs(_x)) * 0.5

        return pos + neg


def prelu(_x, init=0.0):
    with tf.name_scope('prelu'):
        alphas = tf.Variable(tf.constant(init,
                                         shape=[int(_x.get_shape()[-1])]),
                             trainable=True)

        pos = tf.nn.relu(_x)
        neg = alphas * (_x - tf.abs(_x)) * 0.5

        return pos + neg
