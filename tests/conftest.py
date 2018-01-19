import pytest
import os
import csv

from model_serving.model import Model


@pytest.fixture(scope='session')
def imagenet_mobilenet_v1_model():
    return Model({'name': 'mobilenet_v1'})


@pytest.fixture(scope='session')
def imagenet_dictionary():
    with open(os.path.join('tests', 'files', 'dictionary.csv'), 'r') as file:
        return [name for _, name in csv.reader(file)]
