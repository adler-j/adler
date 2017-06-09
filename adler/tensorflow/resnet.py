import numpy as np
from adler.tensorflow.layers import conv2d, conv2dtransp, conv1d, conv1dtransp
from adler.tensorflow.activation import prelu, leaky_relu


import demandimport
with demandimport.enabled():
    import tensorflow as tf


class ResNet(object):
    def __init__(self,
                 channels,
                 ndim=2,
                 depth=3,
                 conv_layers=3,
                 reduce_layers=2,
                 features=32,
                 keep_prob=1.0,
                 use_batch_norm=True,
                 activation='prelu',
                 initialization='xavier',
                 is_training=True,
                 name='resunit'):
        self.channels = channels
        self.ndim = ndim
        self.depth = depth
        self.conv_layers = conv_layers
        self.reduce_layers = reduce_layers
        self.features = features

        self.keep_prob = keep_prob
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.initialization = initialization

        self.is_training = is_training

        self.name = str(name)

        with tf.name_scope('{}_variables'.format(name)):
            self.conv_weights = []
            self.conv_bias = []
            self.reduce_weights = []
            self.reduce_bias = []
            for d in range(self.depth):
                conv_weights = []
                conv_bias = []
                reduce_weights = []
                reduce_bias = []

                for i in range(self.conv_layers):
                    nin = self.channels if i == 0 else features
                    w, b = self.get_weight_bias(nin, features, size=3)
                    conv_weights.append(w)
                    conv_bias.append(b)

                for i in range(self.reduce_layers):
                    nout = self.channels if i == (self.reduce_layers - 1) else features
                    w, b = self.get_weight_bias(features, nout, size=1)
                    reduce_weights.append(w)
                    reduce_bias.append(b)

                self.conv_weights.append(conv_weights)
                self.conv_bias.append(conv_bias)
                self.reduce_weights.append(reduce_weights)
                self.reduce_bias.append(reduce_bias)

    def get_weight_bias(self, nin, nout, size=3, transpose=False):
        if self.ndim == 1:
            if self.initialization == 'xavier':
                stddev = np.sqrt(2.6 / (3 * (nin + nout)))
            elif self.initialization == 'he':
                stddev = np.sqrt(2.6 / (size * nin))

            if transpose:
                w = tf.Variable(tf.truncated_normal([size, nout, nin],
                                                    stddev=stddev))
            else:
                w = tf.Variable(tf.truncated_normal([size, nin, nout],
                                                    stddev=stddev))

            b = tf.Variable(tf.constant(0.0, shape=[1, 1, nout]))

            return w, b
        elif self.ndim == 2:
            if self.initialization == 'xavier':
                stddev = np.sqrt(2.6 / (3 * (nin + nout)))
            elif self.initialization == 'he':
                stddev = np.sqrt(2.6 / (size * nin))

            stddev = np.sqrt(2.6 / (size * size * nin))
            if transpose:
                w = tf.Variable(tf.truncated_normal([size, size, nout, nin],
                                                    stddev=stddev))
            else:
                w = tf.Variable(tf.truncated_normal([3, 3, nin, nout],
                                                    stddev=stddev))

            b = tf.Variable(tf.constant(0.0, shape=[1, 1, 1, nout]))

            return w, b
        else:
            raise ValueError('unknown ndim')

    def apply_activation(self, x):
        if self.activation == 'relu':
            return tf.nn.relu(x)
        elif self.activation == 'elu':
            return tf.nn.elu(x)
        elif self.activation == 'leaky_relu':
            return leaky_relu(x)
        elif self.activation == 'prelu':
            return prelu(x)
        else:
            raise RuntimeError('unknown activation')

    def apply_conv(self, x, w, b, stride=False,
                   disable_batch_norm=False,
                   disable_dropout=False,
                   disable_activation=False):
        if stride:
            if self.ndim == 1:
                stride = 2
            elif self.ndim == 2:
                stride = (2, 2)
        else:
            if self.ndim == 1:
                stride = 1
            elif self.ndim == 2:
                stride = (1, 1)

        with tf.name_scope('apply_conv'):
            if self.ndim == 1:
                x = conv1d(x, w, stride=stride) + b
            elif self.ndim == 2:
                x = conv2d(x, w, stride=stride) + b

            if self.use_batch_norm and not disable_batch_norm:
                x = tf.contrib.layers.batch_norm(x,
                                                 is_training=self.is_training)
            if self.keep_prob != 1.0 and not disable_dropout:
                x = tf.contrib.layers.dropout(x, keep_prob=self.keep_prob,
                                              is_training=self.is_training)

            if not disable_activation:
                x = self.apply_activation(x)

            return x

    def __call__(self, x):
        with tf.name_scope('{}_call'.format(self.name)):
            for d in range(self.depth):
                dx = x

                for i in range(self.conv_layers):
                    dx = self.apply_conv(dx,
                                         self.conv_weights[d][i],
                                         self.conv_bias[d][i])

                for i in range(self.reduce_layers):
                    last = (i == self.reduce_layers - 1)
                    dx = self.apply_conv(dx,
                                         self.reduce_weights[d][i],
                                         self.reduce_bias[d][i],
                                         disable_batch_norm=last,
                                         disable_dropout=last,
                                         disable_activation=last)

                x = x + dx

        return x
