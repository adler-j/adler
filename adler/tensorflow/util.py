import os
import shutil
from os.path import join, expanduser, exists

import demandimport
with demandimport.enabled():
    import tensorflow as tf

__all__ = ('get_base_dir',
           'default_checkpoint_path', 'default_tensorboard_dir',
           'summary_writers')


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


def summary_writers(name, cleanup=False, session=None, write_graph=True):
    if session is None:
        session = tf.get_default_session()

    dname = default_tensorboard_dir(name)

    if cleanup and os.path.exists(dname):
        shutil.rmtree(dname, ignore_errors=True)

    if write_graph:
        graph = session.graph
    else:
        graph = None

    test_summary_writer = tf.summary.FileWriter(dname + '/test', graph)
    train_summary_writer = tf.summary.FileWriter(dname + '/train')

    return test_summary_writer, train_summary_writer


def run_with_profile(ops, feed_dict, name='profile.json', session=None):
    from tensorflow.python.client import timeline

    if session is None:
        session = tf.get_default_session()

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    result = session.run(ops, feed_dict=feed_dict,
                         options=options,
                         run_metadata=run_metadata)

    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(name, 'w') as f:
        f.write(chrome_trace)

    return result


if __name__ == '__main__':
    print('base dir: {}'.format(get_base_dir()))
