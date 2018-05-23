import demandimport
with demandimport.enabled():
    import tensorflow as tf

    
__all__ = ('image_grid', 'image_summary', 'scalars_summary')


def image_grid(x, size=8):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image[None]


def image_grid_summary(name, x):
    with tf.name_scope(name):
        tf.summary.image('grid', image_grid(x))


def scalars_summary(name, x):
    with tf.name_scope(name):
        x = tf.reshape(x, [-1])
        mean, var = tf.nn.moments(x, axes=0)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('std', tf.sqrt(var))
        tf.summary.histogram('histogram', x)