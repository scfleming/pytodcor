#!/usr/bin/env python
# Licensed under an MIT license - see LICENSE file in this project's repository.

"""
.. module:: setup

   :synopsis: This script is used to setup the pip packages.

.. moduleauthor:: Scott W. Fleming <fleming@stsci.edu>
"""

from setuptools import setup
from pytodcor import __version__

setup(name="pytodcor", version=__version__,
      description="Two-dimensional cross-correlation of one-dimensional spectra.",
      classifiers=["Programming Language :: Python :: 3"],
      url="https://github.com/scfleming/pytodcor",
      author="Scott W. Fleming",
      author_email="fleming@stsci.edu", license="MIT",
      packages=["pytodcor"],
      install_requires=["astropy>=5.3.1", "matplotlib>=3.7.2", "numpy>=1.25.2",
                        "specutils>=1.11.0"])
