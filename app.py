import os
import sys

#Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Utilites
import numpy as np
from util import base64_to_pil


#Declare a flask app
app = Flask(__name__)

MODEL_PATH = 'models/wildfire_detection.h5'

#Load your own trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img, model):
    img = img.resize((256, 256))

    #Preprocessing the image
    img = np.array(img)/255.0
    #img = tf.image.rgb_to_grayscale(img)
    x = image.img_to_array(img)    
    #x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    
    return preds


@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        #Get the image from post request
        img = base64_to_pil(request.json)
        
        #Make prediction
        preds = model_predict(img, model)

        pred_prob = "{:.3f}".format(np.amax(preds))

        classes = {0: "Fire Detected!", 1: "Fire Not Detected"}
        prediction = classes[preds[0]]
        
        result = str(prediction)              
        result = result.replace('_', ' ').capitalize()
        
        return jsonify(result=result, probability=pred_prob)

    return None



if __name__ == '__main__':
    #app.run(port=5002, threaded=False)

    #Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
