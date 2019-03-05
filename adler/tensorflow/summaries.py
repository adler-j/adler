import demandimport
with demandimport.enabled():
    import tensorflow as tf

    
__all__ = ('image_grid', 'image_grid_summary', 'scalars_summary')


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

        
def segmentation_overlay_summary(img, segmentation, alpha=0.5, gamma_factor=2.2):
    minv = tf.reduce_min(img, axis=[1, 2, 3], keepdims=True)
    maxv = tf.reduce_max(img, axis=[1, 2, 3], keepdims=True)
    img = (img - minv) / (maxv - minv)
    img = tf.concat(3 * [img], axis=-1)
    red = np.zeros([1, 1, 1, 3], 'float32')
    red[..., 0] = 1
    red = tf.convert_to_tensor(red)

    img_rgb_pow = img ** gamma_factor

    out_rgb_pow = red * alpha * segmentation + img_rgb_pow * (1. - alpha * segmentation)
    out_rgb = out_rgb_pow ** (1. / gamma_factor)
    return out_rgb