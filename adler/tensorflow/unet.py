import demandimport
with demandimport.enabled():
    import tensorflow as tf

import numpy as np
from adler.tensorflow.layers import conv2d, conv2dtransp
from adler.tensorflow.activation import prelu, leaky_relu


class UNet(object):
    def __init__(self, nin, nout,
                 depth=4, layers_per_depth=2,
                 features=32, feature_increase=2,
                 keep_prob=1.0,
                 use_batch_norm=True,
                 activation='prelu',
                 is_training=True,
                 name='unet'):
        self.nin = nin
        self.nout = nout
        self.depth = depth
        self.layers_per_depth = layers_per_depth
        self.features = features
        self.feature_increase = feature_increase
        self.keep_prob = keep_prob

        self.use_batch_norm = use_batch_norm
        self.activation = activation

        self.is_training = is_training

        self.name = str(name)

        with tf.name_scope('{}_variables'.format(self.name)):
            self.w_in, self.b_in = self.get_weight_bias(nin, features)

            self.w_down, self.b_down = [], []
            self.w_up, self.b_up = [], []
            for i in range(self.depth):
                features_i = self.features_at(i)
                self.w_down.append([])
                self.b_down.append([])
                self.w_up.append([])
                self.b_up.append([])
                for j in range(self.layers_per_depth):
                    with tf.name_scope('down_{}_{}'.format(i, j)):
                        if i > 0 and j == 0:
                            w, b = self.get_weight_bias(self.features_at(i - 1), features_i)
                        else:
                            w, b = self.get_weight_bias(features_i, features_i)

                        self.w_down[i].append(w)
                        self.b_down[i].append(b)

                    with tf.name_scope('up_{}_{}'.format(i, j)):
                        if j == 0:
                            w, b = self.get_weight_bias(self.features_at(i + 1), features_i, transpose=True)
                        elif j == 1:
                            w, b = self.get_weight_bias(features_i * 2, features_i)
                        else:
                            w, b = self.get_weight_bias(features_i, features_i)

                        self.w_up[i].append(w)
                        self.b_up[i].append(b)

            features_i = self.features_at(self.depth)
            self.w_coarse, self.b_coarse = [], []
            for j in range(self.layers_per_depth):
                with tf.name_scope('coarse_{}'.format(j)):
                    if j == 0:
                        w, b = self.get_weight_bias(self.features_at(self.depth - 1), features_i)
                    else:
                        w, b = self.get_weight_bias(features_i, features_i)

                    self.w_coarse.append(w)
                    self.b_coarse.append(b)

            with tf.name_scope('out_{}'.format(j)):
                self.w_out, self.b_out = self.get_weight_bias(features, nout)

    def features_at(self, level):
        return int(self.features * self.feature_increase ** level)

    def get_weight_bias(self, nin, nout, transpose=False):
        # Xavier initialization
        stddev = np.sqrt(2.6 / (3 * 3 * (nin + nout)))
        if transpose:
            w = tf.Variable(tf.truncated_normal([3, 3, nout, nin], stddev=stddev))
        else:
            w = tf.Variable(tf.truncated_normal([3, 3, nin, nout], stddev=stddev))

        b = tf.Variable(tf.constant(0.0, shape=[1, 1, 1, nout]))

        return w, b

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

    def apply_conv(self, x, w, b, stride=(1, 1)):
        with tf.name_scope('apply_conv'):

            x = conv2d(x, w, stride=stride) + b

            if self.use_batch_norm:
                x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                                 is_training=self.is_training)
            if self.keep_prob != 1.0:
                x = tf.contrib.layers.dropout(x, keep_prob=self.keep_prob,
                                              is_training=self.is_training)

            return self.apply_activation(x)

    def apply_convtransp(self, x, w, b, stride=(1, 1), out_shape=None):
        with tf.name_scope('apply_convtransp'):

            x = conv2dtransp(x, w, stride=stride, out_shape=out_shape) + b

            if self.use_batch_norm:
                x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                                 is_training=self.is_training)
            if self.keep_prob != 1.0:
                x = tf.contrib.layers.dropout(x, keep_prob=self.keep_prob,
                                              is_training=self.is_training)

            return self.apply_activation(x)

    def __call__(self, x):

        finals = []

        with tf.name_scope('{}_call'.format(self.name)):
            with tf.name_scope('in'):
                current = self.apply_conv(x, self.w_in, self.b_in)

            # down layers
            for i in range(self.depth):
                with tf.name_scope('down_{}'.format(i)):
                    for j in range(self.layers_per_depth - 1):
                        current = self.apply_conv(current, self.w_down[i][j], self.b_down[i][j])

                    finals.append(current)

                    current = self.apply_conv(current, self.w_down[i][-1], self.b_down[i][-1], stride=(2, 2))

            with tf.name_scope('coarse'):
                for j in range(self.layers_per_depth):
                    current = self.apply_conv(current, self.w_coarse[j], self.b_coarse[j])

            # up layers
            for i in reversed(range(self.depth)):
                with tf.name_scope('up_{}'.format(i)):
                    x_shape = tf.shape(finals[i])
                    W_shape = tf.shape(self.w_up[i][0])
                    out_shape = tf.stack([x_shape[0],
                                          x_shape[1],
                                          x_shape[2],
                                          W_shape[2]])

                    current = self.apply_convtransp(current, self.w_up[i][0], self.b_up[i][0], stride=(2, 2), out_shape=out_shape)

                    # Skip connection
                    current = tf.concat([current, finals[i]], axis=-1)

                    for j in range(1, self.layers_per_depth):
                        current = self.apply_conv(current, self.w_up[i][j], self.b_up[i][j])

            with tf.name_scope('out'):
                current = conv2d(current, self.w_out) + self.b_out

        return current
