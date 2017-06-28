import demandimport
with demandimport.enabled():
    import tensorflow as tf


__all__ = ('log10', 'psnr')



def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(x_result, x_true, name='psnr'):
    with tf.variable_scope(name):
        maxval = tf.reduce_max(x_true) - tf.reduce_min(x_true)
        mse = tf.reduce_mean((x_result - x_true) ** 2)
        return 20 * log10(maxval) - 10 * log10(mse)