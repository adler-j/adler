import numpy as np

import demandimport
with demandimport.enabled():
    import odl


def random_shapes(interior=False):
    if interior:
        x_0 = np.random.rand() - 0.5
        y_0 = np.random.rand() - 0.5
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0

    return ((np.random.rand() - 0.5) * np.random.exponential(0.4),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc, n_ellipse=50, interior=False, form='ellipse'):
    n = np.random.poisson(n_ellipse)
    shapes = [random_shapes(interior=interior) for _ in range(n)]
    if form == 'ellipse':
        return odl.phantom.ellipsoid_phantom(spc, shapes)
    if form == 'rectangle':
        return odl.phantom.cuboid_phantom(spc, shapes)
    else:
        raise Exception('unknown form')
