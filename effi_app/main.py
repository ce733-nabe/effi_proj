import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import tensorflow as tf
import time

from django.conf import settings

def pred(img_path):
    #model = tf.keras.applications.EfficientNetB0(weights='imagenet')
    #model.save('./test_model')
    model = tf.keras.models.load_model(settings.STATIC_ROOT + '/test_model')

    width = 224
    height = 224

    #img_path = './panda.jpg'

    X = []
    img = img_to_array(load_img(img_path, target_size=(width,height)))
    img = img[:,:,::-1]
    X.append(img)
    X = np.asarray(X)
    print('X.shape: ', X.shape)

    y = model.predict(X)
    print(decode_predictions(y, top=1))

    return decode_predictions(y, top=1)

class Pred():
    def __init__(self):
        print('Pred コンストラクタが呼び出されました！')
      
        self.model = tf.keras.applications.EfficientNetB0(weights='imagenet')
        #self.model = tf.keras.models.load_model(settings.STATIC_ROOT + '/test_model')
        
    def __del__(self):
        print ('Pred デストラクタが呼び出されました!')

    def pred(self,img_path):
        #model = tf.keras.applications.EfficientNetB0(weights='imagenet')
        #model.save('./test_model')
        #model = tf.keras.models.load_model(settings.STATIC_ROOT + '/test_model')

        width = 224
        height = 224

        #img_path = './panda.jpg'

        X = []
        img = img_to_array(load_img(img_path, target_size=(width,height)))
        img = img[:,:,::-1]
        X.append(img)
        X = np.asarray(X)
        print('X.shape: ', X.shape)

        y = self.model.predict(X)
        print(decode_predictions(y, top=1))

        return decode_predictions(y, top=1)



