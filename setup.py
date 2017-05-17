"""Setup script for adler.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages

setup(
    name='adler',

    version='0.0.0',

    description='Adler',

    url='https://github.com/adler-j/adler',

    author='Jonas Adler',
    author_email='jonasadl@kth.se',

    license='GPLv3+',

    packages=find_packages(exclude=['*test*']),
    package_dir={'adler': 'adler'},

    install_requires=['numpy', 'demandimport']
)
