import os
import tempfile
import stored
import traceback

from flask import Flask, jsonify, request

from .model import Model


def create_app(model):
    app = Flask(__name__)
    app.config.from_object(__name__)
    app.model = model

    @app.errorhandler(Exception)
    def unhandled_exception(exception):
        traceback.print_exc()
        if str(exception) == '<Response [404]>' or 'Max retries exceeded with url' in str(exception):
            return jsonify({'error': 'unable to resolve image url'})
        elif 'cannot identify image file' in str(exception):
            return jsonify({'error': 'unable to read image file'})
        else:
            return jsonify({'error': str(exception)})

    @app.route('/')
    def index():
        return jsonify(app.model.as_json())

    @app.route('/classify', methods=['GET', 'POST'])
    def classify():
        with tempfile.NamedTemporaryFile() as image_file:
            image_path = image_file.name
            if request.method == 'POST':
                if 'image' not in request.files:
                    return jsonify({'error': 'missing image file'})
                image_file.write(request.files['image'].read())
                image_file.flush()
            else:
                if 'image' not in request.args:
                    return jsonify({'error': 'missing image url'})
                stored.sync(request.args['image'], image_path)
            if 'extract_features' in request.args:
                predictions = app.model.extract_features(image_path)
            else:
                predictions = app.model.classify_image(image_path)
            return jsonify(predictions)

    @app.route('/services/ping')
    def services_ping():
        return jsonify(ping='pong')

    return app


if __name__ == '__main__':
    encoded_spec = os.environ.get('SERVING_MODEL_SPEC')
    model = Model(encoded_spec)
    app = create_app(model)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))
