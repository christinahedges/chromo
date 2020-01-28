#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/chromo*")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('chromo/version.py').read())

# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
# 2. What dependencies required to run the unit tests? (i.e. `pytest --remote-data`)
tests_require = ['pytest', 'pytest-cov', 'pytest-remotedata']
# 3. What dependencies are required for optional features?
# `BoxLeastSquaresPeriodogram` requires astropy>=3.1.
# `interact()` requires bokeh>=1.0, ipython.
# `PLDCorrector` requires scikit-learn, pybind11, celerite.
extras_require = {"all":  ["astropy>=3.1",
                           "bokeh>=1.0", "ipython",
                           "scikit-learn", "pybind11", "celerite"],
                  "test": tests_require}

setup(name='chromo',
      version=__version__,
      description="An unfriendly package for chromatic PSFs in TESS",
      long_description=open('README.md').read(),
      author='TESS Party',
      author_email='christina.l.hedges@nasa.gov',
      license='MIT',
      package_dir={
            'chromo': 'chromo'},
      packages=['chromo'],
      install_requires=install_requires,
      extras_require=extras_require,
      setup_requires=['pytest-runner'],
      tests_require=tests_require,
      include_package_data=True,
      classifiers=[
          "Development Status :: 0 - Rubbish",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
