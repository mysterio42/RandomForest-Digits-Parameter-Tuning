import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from utils.model import load_model

app = Flask(__name__)
api = Api(app)
model = load_model()


class Handwritten(Resource):

    def post(self):
        posted_data = request.get_json()

        assert 'image' in posted_data

        pred = model.predict(
            np.array(posted_data['image']).reshape(1, -1)
        )[0]
        return jsonify({
            'prediction': {
                'digit': str(int(pred))
            }
        })


api.add_resource(Handwritten, '/handwritten')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
