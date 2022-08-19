import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np


# Parse the version from the module.
# Source: https://github.com/mapbox/rasterio/blob/master/setup.py
with open('rastercrf/version.py') as f:

    for line in f:

        if line.find("__version__") >= 0:

            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")

            continue

pkg_name = 'rastercrf'
maintainer = 'Jordan Graesser'
maintainer_email = ''
description = 'Conditional Random Fields for rasters'
git_url = 'http://github.com/jgrss/rastercrf.git'
download_url = 'https://github.com/jgrss/rastercrf/archive/{VERSION}.tar.gz'.format(VERSION=version)
keywords = ['raster', 'CRF']

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('requirements.txt') as f:
    required_packages = f.read()


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'rastercrf': ['transforms/*.so',
                          'data/*.bz2',
                          'models/*.model']}


def get_extensions():

    return [Extension('*',
                      sources=['rastercrf/transforms/_crf.pyx'],
                      language='c++')]


def setup_package():

    include_dirs = [np.get_include()]

    metadata = dict(name=pkg_name,
                    maintainer=maintainer,
                    maintainer_email=maintainer_email,
                    description=description,
                    license=license_file,
                    version=version,
                    long_description=long_description,
                    packages=get_packages(),
                    package_data=get_package_data(),
                    ext_modules=cythonize(get_extensions()),
                    zip_safe=False,
                    keywords=keywords,
                    url=git_url,
                    download_url=download_url,
                    install_requires=required_packages,
                    include_dirs=include_dirs,
                    classifiers=['Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 'Programming Language :: Python :: 3.7'])

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
