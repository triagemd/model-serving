#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='model-serving',
    version='1.2.0',
    description='A tensorflow-serving model server via HTTP.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    scripts=[],
    install_requires=[
        'flask',
        'Flask-BasicAuth',
        'pyyaml',
        'simplejson',
        'tensorflow_serving_client>=0.0.10',
        'requests',
        'futures',
        'keras-model-specs>=1.2.0',
        'stored>=0.0.29',
        'sklearn',
        'numpy'
    ]
)
