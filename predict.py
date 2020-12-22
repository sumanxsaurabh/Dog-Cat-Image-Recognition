#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Suman Saurabh
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        model = load_model('model_cat_dog.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0] < 0.5:
            prediction = 'This is a Cat'
        elif 0.5 <= result[0] <= 1:
            prediction = 'This is a Dog'

        return [prediction]


