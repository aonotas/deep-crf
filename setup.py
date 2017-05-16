# setup.py
from setuptools import setup, find_packages

setup(
    name='DeepCRF',
    version='1.0',
    packages=find_packages(exclude=('tests*',)),
    include_package_data=False,
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        deep-crf=deepcrf:cli
    '''
)
