import pickle 
from flask import Flask, render_template, request, session, Response, redirect, url_for
import json
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.sav",'rb'))


@app.route('/AI', methods=["GET"])
def test_AI():
    X_test = np.random.rand(1,1024)
    results = model.predict(X_test)
    message = {"result": results[0].tolist()}
    return Response(json.dumps(message), status=200, mimetype='application/json')


if __name__ == '__main__':
    app.secret_key = ".."
    app.run(port=8080, threaded=True, host=('127.0.0.1'))