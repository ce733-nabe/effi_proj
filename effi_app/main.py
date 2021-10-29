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


class Preds():
    def __init__(self,filenames, batch_size):
        print('Preds コンストラクタが呼び出されました！')
        self.imgs = self.imgs_load(filenames)
 
        self.batch_size = batch_size
        self.model = tf.keras.applications.EfficientNetB0(weights='imagenet')
        
    def __del__(self):
        print ('Preds デストラクタが呼び出されました!')

    #def imgs_load(self,img_path):
    def imgs_load(self,filenames):
        #filenames = glob.glob(img_path + '*.jpg')
    
        X = []
        
        for filename in filenames:
            img = img_to_array(load_img(filename, target_size=(224 ,224)))
            img = img[:,:,::-1]
            X.append(img)
            
        X = np.asarray(X)
        
        print('X.shape: ', X.shape)
        
        return X
    
    def preds(self):
        batch_num = 0
        mbox= []
        for start_idx in range(0, self.imgs.shape[0], self.batch_size):
            batch_num += 1
            print("Validating batch {:}".format(batch_num))

            end_idx = min(start_idx + self.batch_size, self.imgs.shape[0])
            effective_batch_size = end_idx - start_idx

            X = self.imgs[start_idx:start_idx + effective_batch_size]

            y = self.model.predict(X)
            print(decode_predictions(y, top=1))
            mbox.extend(decode_predictions(y, top=1))
            
        return mbox
#Preds(img_path='./gazou_sample/',batch_size = 4).preds()



