import os
from os.path import join, expanduser, exists


def get_base_dir():
    """Get the data directory."""
    base_odl_dir = os.environ.get('ADLER_HOME',
                                  expanduser(join('~', '.adler')))
    data_home = join(base_odl_dir, 'tensorflow')
    if not exists(data_home):
        os.makedirs(data_home)
    return data_home


def default_checkpoint_path(name):
    checkpoint_dir = join(get_base_dir(), 'checkpoints')
    if not exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = join(checkpoint_dir,
                           '{}.ckpt'.format(name))

    return checkpoint_path


def default_tensorboard_dir(name):
    tensorboard_dir = join(get_base_dir(), 'tensorboard', name)
    if not exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    return tensorboard_dir
