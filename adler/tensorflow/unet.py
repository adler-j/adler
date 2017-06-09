import demandimport
with demandimport.enabled():
    import tensorflow as tf

import numpy as np
from adler.tensorflow.layers import conv2d, conv2dtransp, conv1d, conv1dtransp, maxpool2d, maxpool1d
from adler.tensorflow.activation import prelu, leaky_relu


class UNet(object):
    def __init__(self, nin, nout,
                 ndim=2,
                 depth=4, layers_per_depth=2,
                 features=32, feature_increase=2,
                 residual=False,  # else skip
                 keep_prob=1.0,
                 use_batch_norm=True,
                 activation='prelu',
                 is_training=True,
                 name='unet'):
        self.nin = nin
        self.nout = nout
        self.ndim = ndim
        self.depth = depth
        self.layers_per_depth = layers_per_depth
        self.features = features
        self.feature_increase = feature_increase
        self.residual = residual
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
        if self.ndim == 1:
            # Xavier initialization
            # stddev = np.sqrt(2.6 / (3 * (nin + nout)))

            # He initialization
            stddev = np.sqrt(2.6 / (3 * nin))
            if transpose:
                w = tf.Variable(tf.truncated_normal([3, nout, nin], stddev=stddev))
            else:
                w = tf.Variable(tf.truncated_normal([3, nin, nout], stddev=stddev))

            b = tf.Variable(tf.constant(0.0, shape=[1, 1, nout]))

            return w, b
        elif self.ndim == 2:
            # Xavier initialization
            # stddev = np.sqrt(2.6 / (3 * 3 * (nin + nout)))

            # He initialization
            stddev = np.sqrt(2.6 / (3 * 3 * nin))
            if transpose:
                w = tf.Variable(tf.truncated_normal([3, 3, nout, nin], stddev=stddev))
            else:
                w = tf.Variable(tf.truncated_normal([3, 3, nin, nout], stddev=stddev))

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
                out = conv1d(x, w, stride=stride) + b
            elif self.ndim == 2:
                out = conv2d(x, w, stride=stride) + b

            if self.use_batch_norm and not disable_batch_norm:
                out = tf.contrib.layers.batch_norm(out,
                                                 is_training=self.is_training)
            if self.keep_prob != 1.0 and not disable_dropout:
                out = tf.contrib.layers.dropout(out, keep_prob=self.keep_prob,
                                              is_training=self.is_training)

            if not disable_activation:
                out = self.apply_activation(out)

            return out

    def apply_convtransp(self, x, w, b, stride=False, out_shape=None,
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

        with tf.name_scope('apply_convtransp'):
            if self.ndim == 1:
                out = conv1dtransp(x, w, stride=stride, out_shape=out_shape) + b
            elif self.ndim == 2:
                out = conv2dtransp(x, w, stride=stride, out_shape=out_shape) + b

            if self.use_batch_norm and not disable_batch_norm:
                out = tf.contrib.layers.batch_norm(out,
                                                   is_training=self.is_training)
            if self.keep_prob != 1.0 and not disable_dropout:
                out = tf.contrib.layers.dropout(out, keep_prob=self.keep_prob,
                                                is_training=self.is_training)

            if not disable_activation:
                out = self.apply_activation(out)

            return out

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

                    current = self.apply_conv(current, self.w_down[i][-1], self.b_down[i][-1], stride=True)

            with tf.name_scope('coarse'):
                for j in range(self.layers_per_depth):
                    current = self.apply_conv(current, self.w_coarse[j], self.b_coarse[j])

            # up layers
            for i in reversed(range(self.depth)):
                with tf.name_scope('up_{}'.format(i)):
                    with tf.name_scope('upsampling'):
                        x_shape = tf.shape(finals[i])
                        W_shape = tf.shape(self.w_up[i][0])
                        if self.ndim == 1:
                            out_shape = tf.stack([x_shape[0],
                                                  x_shape[1],
                                                  W_shape[1]])
                        elif self.ndim == 2:
                            out_shape = tf.stack([x_shape[0],
                                                  x_shape[1],
                                                  x_shape[2],
                                                  W_shape[2]])

                        current = self.apply_convtransp(current, self.w_up[i][0], self.b_up[i][0], stride=True, out_shape=out_shape)

                    with tf.name_scope('skip_connection'):
                        current = tf.concat([current, finals[i]], axis=-1)

                    for j in range(1, self.layers_per_depth):
                        current = self.apply_conv(current, self.w_up[i][j], self.b_up[i][j])

            with tf.name_scope('out'):
                current = self.apply_conv(current, self.w_out, self.b_out,
                                          disable_dropout=True,
                                          disable_batch_norm=True,
                                          disable_activation=True)

        return current


def reference_unet(x, nout,
                   ndim=2,
                   features=64,
                   keep_prob=1.0,
                   use_batch_norm=True,
                   activation='relu',
                   is_training=True,
                   init='xavier',
                   name='unet_original'):
    def get_weight_bias(nin, nout, transpose, size):
        if ndim == 1:
            if init == 'xavier':
                stddev = np.sqrt(2.6 / (size * (nin + nout)))
            elif init == 'he':
                stddev = np.sqrt(2.6 / (size * (nin)))

            if transpose:
                w = tf.Variable(tf.truncated_normal([size, nout, nin], stddev=stddev))
            else:
                w = tf.Variable(tf.truncated_normal([size, nin, nout], stddev=stddev))

            b = tf.Variable(tf.constant(0.0, shape=[1, 1, nout]))

            return w, b
        elif ndim == 2:
            if init == 'xavier':
                stddev = np.sqrt(2.6 / (size * size * (nin + nout)))
            elif init == 'he':
                stddev = np.sqrt(2.6 / (size * size * (nin)))

            if transpose:
                w = tf.Variable(tf.truncated_normal([size, size, nout, nin], stddev=stddev))
            else:
                w = tf.Variable(tf.truncated_normal([size, size, nin, nout], stddev=stddev))

            b = tf.Variable(tf.constant(0.0, shape=[1, 1, 1, nout]))

            return w, b
        else:
            raise ValueError('unknown ndim')

    def apply_activation(x):
        if activation == 'relu':
            return tf.nn.relu(x)
        elif activation == 'elu':
            return tf.nn.elu(x)
        elif activation == 'leaky_relu':
            return leaky_relu(x)
        elif activation == 'prelu':
            return prelu(x)
        else:
            raise RuntimeError('unknown activation')

    def apply_conv(x, nout,
                   stride=False,
                   size=3,
                   disable_batch_norm=False,
                   disable_dropout=False,
                   disable_activation=False):

        if stride:
            if ndim == 1:
                stride = 2
            elif ndim == 2:
                stride = (2, 2)
        else:
            if ndim == 1:
                stride = 1
            elif ndim == 2:
                stride = (1, 1)

        with tf.name_scope('apply_conv'):
            nin = int(x.get_shape()[-1])

            w, b = get_weight_bias(nin, nout, transpose=False, size=size)

            if ndim == 1:
                out = conv1d(x, w, stride=stride) + b
            elif ndim == 2:
                out = conv2d(x, w, stride=stride) + b

            if use_batch_norm and not disable_batch_norm:
                out = tf.contrib.layers.batch_norm(out,
                                                   is_training=is_training)
            if keep_prob != 1.0 and not disable_dropout:
                out = tf.contrib.layers.dropout(out, keep_prob=keep_prob,
                                                is_training=is_training)

            if not disable_activation:
                out = apply_activation(out)

            return out

    def apply_convtransp(x, nout,
                         stride=True, out_shape=None,
                         size=2,
                         disable_batch_norm=False,
                         disable_dropout=False,
                         disable_activation=False):

        if stride:
            if ndim == 1:
                stride = 2
            elif ndim == 2:
                stride = (2, 2)
        else:
            if ndim == 1:
                stride = 1
            elif ndim == 2:
                stride = (1, 1)

        with tf.name_scope('apply_convtransp'):
            nin = int(x.get_shape()[-1])

            w, b = get_weight_bias(nin, nout, transpose=True, size=size)

            if ndim == 1:
                out = conv1dtransp(x, w, stride=stride, out_shape=out_shape) + b
            elif ndim == 2:
                out = conv2dtransp(x, w, stride=stride, out_shape=out_shape) + b

            if use_batch_norm and not disable_batch_norm:
                out = tf.contrib.layers.batch_norm(out,
                                                   is_training=is_training)
            if keep_prob != 1.0 and not disable_dropout:
                out = tf.contrib.layers.dropout(out, keep_prob=keep_prob,
                                                is_training=is_training)

            if not disable_activation:
                out = apply_activation(out)

            return out

    def apply_maxpool(x):
        if ndim == 1:
            return maxpool1d(x)
        else:
            return maxpool2d(x)

    finals = []

    with tf.name_scope('{}_call'.format(name)):
        with tf.name_scope('in'):
            current = apply_conv(x, features)
            current = apply_conv(current, features)
            finals.append(current)

        with tf.name_scope('down_1'):
            current = apply_maxpool(current)
            current = apply_conv(current, features * 2)
            current = apply_conv(current, features * 2)
            finals.append(current)

        with tf.name_scope('down_2'):
            current = apply_maxpool(current)
            current = apply_conv(current, features * 4)
            current = apply_conv(current, features * 4)
            finals.append(current)

        with tf.name_scope('down_3'):
            current = apply_maxpool(current)
            current = apply_conv(current, features * 8)
            current = apply_conv(current, features * 8)
            finals.append(current)

        with tf.name_scope('coarse'):
            current = apply_maxpool(current)
            current = apply_conv(current, features * 16)
            current = apply_conv(current, features * 16)

        with tf.name_scope('up_3'):
            skip = finals.pop()
            current = apply_convtransp(current, features * 8,
                                       out_shape=skip.shape)
            current = tf.concat([current, skip], axis=-1)

            current = apply_conv(current, features * 8)
            current = apply_conv(current, features * 8)

        with tf.name_scope('up_2'):
            skip = finals.pop()
            current = apply_convtransp(current, features * 4,
                                       out_shape=skip.shape)
            current = tf.concat([current, skip], axis=-1)

            current = apply_conv(current, features * 4)
            current = apply_conv(current, features * 4)

        with tf.name_scope('up_1'):
            skip = finals.pop()
            current = apply_convtransp(current, features * 2,
                                       out_shape=skip.shape)
            current = tf.concat([current, skip], axis=-1)

            current = apply_conv(current, features * 2)
            current = apply_conv(current, features * 2)

        with tf.name_scope('out'):
            skip = finals.pop()
            current = apply_convtransp(current, features,
                                       out_shape=skip.shape)
            current = tf.concat([current, skip], axis=-1)

            current = apply_conv(current, features)
            current = apply_conv(current, features)

            current = apply_conv(current, nout,
                                 size=1,
                                 disable_activation=True,
                                 disable_batch_norm=True,
                                 disable_dropout=True)

    return current
