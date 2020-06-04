import pytest
import json
import base64
import numpy as np

from model_serving.model import Model


@pytest.fixture(scope='session')
def mobilenet_json():
    return {
        'spec': {
            'name': 'mobilenet_v1',
            'target_size': [224, 224, 3],
            'preprocess_func': 'between_plus_minus_1'
        }
    }


def test_initialize_with_name_spec(mobilenet_json):
    model = Model('mobilenet_v1')
    assert model.as_json() == mobilenet_json


def test_initialize_with_dict_spec(mobilenet_json):
    model = Model({'name': 'mobilenet_v1'})
    assert model.as_json() == mobilenet_json


def test_initialize_with_base64_dict_spec(mobilenet_json):
    encoded_model = base64.b64encode(json.dumps({'name': 'mobilenet_v1'}).encode()).decode()
    model = Model(encoded_model)
    assert model.as_json() == mobilenet_json


def test_as_json(imagenet_mobilenet_v1_model, mobilenet_json):
    actual = imagenet_mobilenet_v1_model.as_json()
    assert actual == mobilenet_json


def test_classify_image(imagenet_mobilenet_v1_model, imagenet_dictionary):
    with open('tests/files/cat.jpg', 'rb') as image_file:
        predictions = imagenet_mobilenet_v1_model.classify_image(image_file.name)
    assert len(predictions) == 1000
    predictions = zip(imagenet_dictionary, predictions)
    predictions = sorted(predictions, reverse=True, key=lambda kv: kv[1])[:5]
    predictions = [(label, float(score)) for label, score in predictions]
    print(predictions)
    expected_top_5 = [
        ['tiger cat', 0.334694504737854],
        ['Egyptian cat', 0.2851393222808838],
        ['tabby, tabby cat', 0.15471667051315308],
        ['kit fox, Vulpes macrotis', 0.03160465136170387],
        ['lynx, catamount', 0.030886519700288773]
    ]
    classes = [name for name, _ in predictions]
    scores = [score for _, score in predictions]
    expected_classes = [name for name, _ in expected_top_5]
    assert classes == expected_classes
    expected_scores = [score for _, score in expected_top_5]
    np.testing.assert_array_almost_equal(np.array(scores), np.array(expected_scores))
