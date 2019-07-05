import pytest
import numpy as np

from model_serving.app import create_app


def assert_cat_predictions(predictions, dictionary):
    assert len(predictions) == 1000
    predictions = zip(dictionary, predictions)
    predictions = sorted(predictions, reverse=True, key=lambda kv: kv[1])[:5]
    predictions = [(label, float(score)) for label, score in predictions]
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


@pytest.fixture(scope='function')
def client_app(imagenet_mobilenet_v1_model):
    app = create_app(imagenet_mobilenet_v1_model)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture(scope='function')
def client_app_features(imagenet_mobilenet_v1_features_model):
    app = create_app(imagenet_mobilenet_v1_features_model)
    app.config['TESTING'] = True
    return app.test_client()


def test_model(client_app):
    response = client_app.get('/')
    assert response.status_code == 200
    assert response.json == {
        'spec': {
            'name': 'mobilenet_v1',
            'target_size': [224, 224, 3],
            'preprocess_func': 'between_plus_minus_1'
        },
    }


def test_model_classify_no_image_url(client_app):
    response = client_app.get('/classify')
    assert response.status_code == 200
    assert response.json == {'error': 'missing image url'}


def test_model_classify_no_image_file(client_app):
    response = client_app.post('/classify')
    assert response.status_code == 200
    assert response.json == {'error': 'missing image file'}


def test_model_classify_invalid_image_url(client_app):
    response = client_app.get('/classify', query_string={'image': 'https://not-a-url/image.jpg'})
    assert response.status_code == 200
    assert response.json == {'error': 'unable to resolve image url'}


def test_model_classify_non_existent_image_url(client_app):
    response = client_app.get('/classify', query_string={'image': 'https://example.com/non-existent-image.jpg'})
    assert response.status_code == 200
    assert response.json == {'error': 'unable to resolve image url'}


def test_model_classify_invalid_image_file(client_app):
    with open('tests/files/not-an-image.jpg', 'rb') as file:
        response = client_app.post('/classify', data={'image': (file, 'not-an-image.jpg')})
    assert response.status_code == 200
    assert response.json == {'error': 'unable to read image file'}


def test_model_classify_with_image_url(client_app, imagenet_dictionary):
    response = client_app.get('/classify', query_string={'image': 'https://image.ibb.co/nu62ba/cat.jpg'})
    assert response.status_code == 200
    assert_cat_predictions(response.json, imagenet_dictionary)


def test_model_classify_with_image_file(client_app, imagenet_dictionary):
    with open('tests/files/cat.jpg', 'rb') as file:
        response = client_app.post('/classify', data={'image': (file, 'cat.jpg')})
    assert response.status_code == 200
    assert_cat_predictions(response.json, imagenet_dictionary)


def test_model_extract_features_with_image_file(client_app_features):
    with open('tests/files/cat.jpg', 'rb') as file:
        response = client_app_features.post('/classify', data={'image': (file, 'cat.jpg')},
                                            query_string={'extract_features': True})
    assert response.status_code == 200
    assert len(response.json) == 1024


def test_services_ping(client_app):
    response = client_app.get('/services/ping')
    assert response.status_code == 200
    assert response.json == {'ping': 'pong'}
