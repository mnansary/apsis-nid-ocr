#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
import sys
import os
import glob
import re
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
# models

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

# Define a flask app
app = Flask(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from nidocr.model import OCR
from nidocr.utils import *
from nidocr.data  import card
ocr=None

@app.route('/', methods=['GET'])
def index():
    global ocr
    ocr=OCR("models")
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,"tests",'uploads', secure_filename(f.filename))
        f.save(file_path)

        response=ocr.extract(file_path)
        return jsonify(response)
    return None


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
