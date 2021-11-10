import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split
import glob

from django.conf import settings

#import json

def pred(img_path):
    #model = tf.keras.applications.EfficientNetB0(weights='imagenet')
    #model.save('./test_model')
    model = tf.keras.models.load_model(settings.STATIC_ROOT + '/test_model')

    width = 224
    height = 224

    #img_path = './panda.jpg'

    X = []
    img = img_to_array(load_img(img_path, target_size=(width,height)))
    #img = img[:,:,::-1]
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
        #self.model = tf.keras.models.load_model(settings.STATIC_ROOT + '/my_model')
        
    def __del__(self):
        print ('Preds デストラクタが呼び出されました!')

    #def imgs_load(self,img_path):
    def imgs_load(self,filenames):
        #filenames = glob.glob(img_path + '*.jpg')
    
        X = []
        
        for filename in filenames:
            img = img_to_array(load_img(filename, target_size=(224 ,224)))
            #img = img[:,:,::-1]
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

            #class_names = ["bush", "obama", "tramp"]
            #pred_labels = np.argmax(y, axis=-1)
            #print(pred_labels)  # [2 0 0 2 2 2 4 2 3 3]
            #pred_label_names = [class_names[x] for x in pred_labels]
            #print(pred_label_names)
            #mbox.extend(pred_label_names)
            
        return mbox
#Preds(img_path='./gazou_sample/',batch_size = 4).preds()


class Effi_Pred():
    def __init__(self,filenames, batch_size):
        print('Effi_Pred コンストラクタが呼び出されました！')
        self.imgs = self.imgs_load(filenames)
 
        self.batch_size = batch_size
        #self.model = tf.keras.applications.EfficientNetB0(weights='imagenet')
        self.model = tf.keras.models.load_model(settings.STATIC_ROOT + '/my_model')
        
    def __del__(self):
        print ('Effi_Pred デストラクタが呼び出されました!')

    #def imgs_load(self,img_path):
    def imgs_load(self,filenames):
        #filenames = glob.glob(img_path + '*.jpg')
    
        X = []
        
        for filename in filenames:
            img = img_to_array(load_img(filename, target_size=(224 ,224)))
            #img = img[:,:,::-1]
            X.append(img)
            
        X = np.asarray(X)
        
        print('X.shape: ', X.shape)
        
        return X
    
    def effi_pred(self):
        batch_num = 0
        mbox= []
        for start_idx in range(0, self.imgs.shape[0], self.batch_size):
            batch_num += 1
            print("Validating batch {:}".format(batch_num))

            end_idx = min(start_idx + self.batch_size, self.imgs.shape[0])
            effective_batch_size = end_idx - start_idx

            X = self.imgs[start_idx:start_idx + effective_batch_size]

            y = self.model.predict(X)
            #print(decode_predictions(y, top=1))
            #mbox.extend(decode_predictions(y, top=1))

            with open(settings.STATIC_ROOT +'/label_list.txt', 'r', encoding='UTF-8') as f:
                label_list = f.readlines()
            dd = []
            for ii in label_list:
                dd.append(ii.rstrip('\n'))
            class_names = dd
            print(class_names)

            #class_names = ["bush", "obama", "tramp"]
            pred_labels = np.argmax(y, axis=-1)
            print(pred_labels)  # [2 0 0 2 2 2 4 2 3 3]
            pred_label_names = [class_names[x] for x in pred_labels]
            print(pred_label_names)
            mbox.extend(pred_label_names)
            
        return mbox
#Preds(img_path='./gazou_sample/',batch_size = 4).preds()

class Effi_Train():
    def __init__(self,filenames):
        print('Effi_train コンストラクタが呼び出されました！')
        self.filenames = filenames

    def __del__(self):
        print ('Effi_train デストラクタが呼び出されました!')
	
    def imgs_load(self,filenames):
        X=[]
        Y=[]
        L={}
        #filenames = glob.glob(base_path + '*.jpg')
    
        for filename in filenames:
            print(filename)
            img = img_to_array(load_img(filename, target_size=(224,224)))
            X.append(img)
            Y.append(os.path.basename(filename).split('_')[0])

            L[os.path.basename(filename).split('_')[1]] = os.path.basename(filename).split('_')[0]

        X = np.array(X)
        print(X.shape)
        Y = np.array(Y)
        print(Y.shape)
        
        self.label_list(L)

        return train_test_split(X, Y), np.unique(Y).size

    def label_list(self,dicts):

        print(dicts)

        dicts_sorted = sorted(dicts.items(), key=lambda x:x[1])
        print(dicts_sorted)

        dd = []
        for ii in dicts_sorted:
            dd.append(ii[0])
        dd = "\n".join(dd)
        print(dd)

        with open(settings.STATIC_ROOT +'/label_list.txt', 'w') as f:
            f.writelines(dd)

    def effi_train(self):
        (X_train, X_test, y_train, y_test), n_classes = self.imgs_load(self.filenames)

        X_train_processed = preprocess_input(X_train)
        X_test_processed = preprocess_input(X_test)
        y_train_categorical = tf.keras.utils.to_categorical(y_train)
        y_test_categorical = tf.keras.utils.to_categorical(y_test)
        print('y_test:{}'.format(y_test))
        print('y_test_categorical:{}'.format(y_test_categorical))

        print('n_classes:{}'.format(n_classes))

        base_model = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), weights='imagenet',include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

        # 学習
        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_processed, y_train_categorical,epochs=10,batch_size=4) #epoch数とか諸々のものは一般のkerasと同様ここでオプション追加する
        model.save(settings.STATIC_ROOT + '/my_model')

        #モデル評価
        model = tf.keras.models.load_model(settings.STATIC_ROOT + '/my_model')
        score = model.evaluate(X_test_processed, y_test_categorical, verbose=0)

        print("loss:", score[0])
        print("accuracy:", score[1])

        return score
        
#Effi_study().effi_train()



