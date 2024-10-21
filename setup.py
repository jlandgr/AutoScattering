#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists
from pathlib import Path
import re

from setuptools import setup, find_packages

author = 'Jonas Landgraf'
email = 'Jonas.Landgraf@mpl.mpg.de Vittorio.Peano@mpl.mpg.de Florian.Marquardt@mpl.mpg.de'
description = 'Automatic design of coupled mode setups'
dist_name = 'autoscattering'
package_name = 'autoscattering'
year = '2024'
url = 'https://github.com/jlandgr/AutoScattering'


def get_version():
    content = open(Path(package_name) / '__init__.py').readlines()
    return "1.0.0"


setup(
    name=dist_name,
    author=author,
    author_email=email,
    url=url,
    version=get_version(),
    packages=find_packages(),
    package_dir={dist_name: package_name},
    include_package_data=True,
    license='MIT',
    description=description,
    long_description=open('README.md').read() if exists('README.md') else '',
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'jax', 'networkx'
    ],
    python_requires=">=3.10",
    classifiers=['Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 ],
    platforms=['ALL'],
    py_modules=[package_name],
    entry_points={
        'console_scripts': [
            'pytheus = pytheus.cli:cli',
        ],
    }
)
