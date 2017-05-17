import numpy as np

import demandimport
with demandimport.enabled():
    import tensorflow as odl


def random_ellipse():
    return ((np.random.rand() - 0.5) * np.random.exponential(0.4),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            2 * np.random.rand() - 1.0, 2 * np.random.rand() - 1.0,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc, n_ellipse=50):
    n = np.random.poisson(n_ellipse)
    ellipses = [random_ellipse() for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, ellipses)
