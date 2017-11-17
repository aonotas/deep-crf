#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='DeepCRF',
    version='1.0',
    packages=find_packages(exclude=('tests*',)),
    include_package_data=False,
    install_requires=[
        'Click',
        'h5py'
    ],
    entry_points='''
        [console_scripts]
        deep-crf=deepcrf:cli
    '''
)
