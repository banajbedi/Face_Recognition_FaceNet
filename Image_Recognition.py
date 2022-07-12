# -*- coding: utf-8 -*-
"""
@author: Banaj Bedi
"""

import tensorflow as tf
import cv2
import os
from PIL import Image
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.metrics import r2_score
import keras

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
tf.compat.v1.keras.backend.set_session(sess)

image_signatures = open("image_data.pkl", "rb")
database = pickle.load(image_signatures)
image_signatures.close()

def recognizeFaces(output, frame):
    x, y, width, height = output['box']
    cv2.rectangle(frame, pt1=(x,y), pt2=(x+width, y+height), color=(0,255,0), thickness=1)

def putTextImage(output, frame, name, acc):
    x, y, width, height = output['box']
    text = name + "\nScore : " + f"{acc:.2f}"
    y0, dy = y+height+50, 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(frame, line, (x-50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)


facenet = FaceNet()

# Image
img = cv2.imread("friends.jpg")
output = facenet.extract(img)
if len(output) > 0:
        for i in output:
            signature = i['embedding']

            min_distance = 1000
            name = ''

            for key, value in database.items() :
                distance = np.linalg.norm(value - signature)
                if distance < min_distance:
                    min_distance = distance
                    name = key
            
            recognizeFaces(i, img)
            putTextImage(i, img, name, min_distance)
        
        cv2.imshow('window', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()