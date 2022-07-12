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


def putText(output, frame, name, acc):
    x, y, width, height = output['box']
    text = name + "\nScore : " + f"{acc:.2f}"
    y0, dy = y-50, 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(frame, line, (x-50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

facenet = FaceNet()

# Video
cap = cv2.VideoCapture("VideoTest1.mp4")
while True:
    ret, frame = cap.read()
    
    output = facenet.extract(frame)
    
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
            
            recognizeFaces(i, frame)
            putText(i, frame, name, min_distance)
        
    cv2.imshow('window', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        cap.release()
        break
    

cv2.destroyAllWindows()