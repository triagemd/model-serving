from tensorflow_serving_client import TensorflowServingClient
from keras_model_specs import ModelSpec


class Model(object):

    def __init__(self, spec):
        if spec and not isinstance(spec, dict):
            spec = {'name': spec}
        self.spec = ModelSpec.get(spec['name'], **spec)
        self.serving_timeout = 30
        self.serving_client = TensorflowServingClient('localhost', 9000)

    def classify_image(self, image_path):
        image_data = self.spec.load_image(image_path)
        response = self.serving_client.make_prediction(image_data, 'image', timeout=self.serving_timeout)
        return response['class_probabilities'][0].tolist()

    def as_json(self):
        spec = {
            'name': self.spec.name,
            'target_size': self.spec.target_size,
            'preprocess_func': self.spec.preprocess_func,
        }
        if self.spec.preprocess_args:
            spec['preprocess_args'] = self.spec.preprocess_args
        return {
            'spec': spec
        }
