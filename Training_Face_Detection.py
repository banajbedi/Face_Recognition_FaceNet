# -*- coding: utf-8 -*-
"""
@author: Banaj Bedi
"""

import cv2
import os
from PIL import Image
import pickle
from keras_facenet import FaceNet

facenet = FaceNet()

folder='Data/'
database = {}
for filename in os.listdir(folder):

    path = folder + filename
    image = cv2.imread(folder + filename)
    
    detections = facenet.extract(image)
    signature = detections[0]['embedding']
    database[os.path.splitext(filename)[0]] = signature

    
image_signatures = open("image_data.pkl", "wb")
pickle.dump(database, image_signatures)
image_signatures.close()