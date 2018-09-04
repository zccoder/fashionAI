#encoding=utf-8
import os
import cv2
import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import h5py
import sys
import keras
from keras.utils import multi_gpu_model

# 指定第一块GPU可用
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
gpus=2
os.environ["CUDA_VISIBLE_DEVICES"] = "0,9"
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

index_=int(sys.argv[1])
#width = 331 # test 1
width=468
bs=3*gpus

test_b_anno_path = '/home2/data/fashionAI/z_rank/Tests/question.csv'
test_b_pict_path = '/home2/data/fashionAI/z_rank'

classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels', 'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels', 
           'pant_length_labels']

label_count = OrderedDict(
               [('coat_length_labels', 8), 
                ('collar_design_labels', 5), 
                ('lapel_design_labels', 5), 
                ('neck_design_labels', 5), 
                ('neckline_design_labels', 10),
                ('pant_length_labels', 6), 
                ('skirt_length_labels', 6), 
                ('sleeve_length_labels', 9)] 
)

cur_class = classes[index_]
n_class = label_count[classes[index_]]
prefix_cls = cur_class.split('_')[0]

fine_result_path = './fine_result_b/NASNET/Width/'
class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
        #val_acc = logs['val_acc']
        #if val_acc > self.best_acc:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
            self.best_loss = val_loss
            #print("\nValidation acc Increased from {} to {}, saving model".format(self.best_acc, val_acc))
            #self.best_acc = val_acc
            weight = self.model.get_weights()
            np.save(self.path, weight)

def Read_anno_data(index_):
   #df_train = pd.read_csv('train.csv', header=None)
   df_train = pd.read_csv(train_csv_path, header=None)
   df_train.columns = ['image_id', 'class', 'label']

   classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels', 
           'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels', 
           'pant_length_labels']
   cur_class = classes[index_]
   df_load = df_train[(df_train['class'] == cur_class)].copy()
   df_load.reset_index(inplace=True)
   del df_load['index']
   del df_train
   print('{0}: {1}'.format(cur_class, len(df_load)))   
   prefix_cls = cur_class.split('_')[0] 
   return df_load,cur_class,prefix_cls

def Read_picture_data(df_load,width):
  n = len(df_load)
  n_class = len(df_load['label'][0])
  print("nclass",n_class)

  X = np.zeros((n, width, width, 3), dtype=np.uint8)
  y = np.zeros((n, n_class), dtype=np.uint8)

  for i in range(n):
    tmp_label = df_load['label'][i]
    if len(tmp_label) > n_class:
        print(df_load['image_id'][i])
    X[i] = cv2.resize(cv2.imread('/home2/data/fashionAI/train_data/{0}'.format(df_load['image_id'][i])), (width, width))
    y[i][tmp_label.find('y')] = 1
 
  return X,y,n,n_class

def build_model(width,n_class):
  import tensorflow as tf
  with tf.device('/cpu:0'):
     cnn_model=NASNetLarge(include_top=False,input_shape=(width,width,3),weights=None)
     cnn_model.load_weights('NASNet-large-no-top.h5')
     #cnn_model.save('InceptionResNetV2.h5') 
     # cnn_model.load_weights('InceptionResNetV2.h5')  
     inputs = Input((width, width, 3))
     x = inputs
     x = Lambda(preprocess_input, name='preprocessing')(x)
     x = cnn_model(x)
     x = GlobalAveragePooling2D()(x)
     x = Dropout(0.5)(x)
     x = Dense(n_class, activation='softmax', name='softmax')(x)
     model = Model(inputs, x)
  #parallel_model = model
  parallel_model = keras.utils.training_utils.multi_gpu_model(model, gpus=gpus)
  adam = Adam(lr=0.0001) # 44
  #adam = Adam(lr=0.0008, decay=0.0001)
  #sdg = SGD(lr=0.0006,momentum=0.6,decay=0.0001) # give up
  parallel_model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
  return parallel_model,cnn_model

def fit_aug_model(model,epochs=7,train_batch_size=5,valid_batch_size=5):
  print("begin data aug:")
  checkpointer = ModelCheckpoint(filepath='./models/NASNET/Width/{0}.best.h5'.format(prefix_cls), verbose=1,save_best_only=True)
  datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
  datagen.fit(X_train)
  test_datagen=ImageDataGenerator()
  test_datagen.fit(X_valid)
  h=model.fit_generator(datagen.flow(X_train, y_train, batch_size=train_batch_size), epochs=epochs,
                      verbose=1, shuffle=True,
                      validation_data=(X_valid, y_valid),
                      callbacks=[EarlyStopping(patience=5), CustomModelCheckpoint(model, './models/NASNET/Width/{0}.best.npy'.format(prefix_cls))])

  # h=model.fit_generator(datagen.flow(X_train, y_train, batch_size=train_batch_size),
  #                   callbacks=[EarlyStopping(patience=4), checkpointer],
  #                   epochs=epochs,shuffle=True,
  #                   validation_data=test_datagen.flow(X_valid, y_valid,batch_size=valid_batch_size))
  return model

def setup_to_finetune(base_model,_stage=2,top_N_layer=-3):
    if(_stage==1):
        for layer in base_model.layers[:top_N_layer]:
           layer.trainable = False
        for layer in base_model.layers[top_N_layer]:
           layer.trainable = True
           print("layer name True:",layer.name)
    if(_stage==2):
       for layer in base_model.layers:
           layer.trainable = True
	
def fit_fine_tuning(model,cnn_model,top_n_layer=-3,fine_epoches=5,fine_batch_size=5):
  setup_to_finetune(cnn_model,_stage=2,top_N_layer=top_n_layer)
  adam = Adam(lr=0.00001) #此处以非常低的学习率微调
  #adam = Adam(lr=0.00005, decay=0.00001) #此处以非常低的学习率微调
  sgd = SGD(lr=0.00005,momentum=0.5,decay=0.00001)
  model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
  model=fit_aug_model(model,epochs=fine_epoches,train_batch_size=fine_batch_size,valid_batch_size=4)
  return model

def Read_test_data(test_anno_path,cur_class):
  df_test = pd.read_csv('%s'%test_anno_path, header=None)
  df_test.columns = ['image_id', 'class', 'x']
  del df_test['x']

  df_load = df_test[(df_test['class'] == cur_class)].copy()
  df_load.reset_index(inplace=True)
  del df_load['index']
  del df_test
  print('{0}: {1}'.format(cur_class, len(df_load)))
  return df_load

def Read_test_picture(df_load,width,test_pict_path):
  n = len(df_load)
  X_test = np.zeros((n, width, width, 3), dtype=np.uint8)
  for i in range(n):
    X_test[i] = cv2.resize(cv2.imread(test_b_pict_path + '/{0}'.format(df_load['image_id'][i])), (width, width))
  return n,X_test

def predict_result(stage_flag,df_load,X_test,batch_size=5):
 # if index_ == 0:
 #    test_np = model.predict(X_test, batch_size=16)
 # else:
 #    test_np = model.predict(X_test, batch_size=8)
  test_np = model.predict(X_test)
  result = []
  for i, row in df_load.iterrows():
    tmp_list = test_np[i]
    tmp_result = ''
    for tmp_ret in tmp_list:
        tmp_result += '{:.4f};'.format(tmp_ret)
        
    result.append(tmp_result[:-1])

  df_load['result'] = result
  if(stage_flag==1):
    df_load.to_csv(fine_result_path + '{}_stage1.csv'.format(prefix_cls), header=None, index=False)
  elif(stage_flag==2):
    df_load.to_csv(fine_result_path + '{}_stage2.csv'.format(prefix_cls), header=None, index=False)
 
#step1
import time
t1=time.time()
#df_load,cur_class,prefix_cls=Read_anno_data(index_)
#X,y,n,n_class=Read_picture_data(df_load,width)
#
## expand dataset
##print 'Starting expand dataset to 20000...'
##cnt = 20000/n + 1
##new_X = np.tile(X, (cnt, 1, 1, 1))
##new_y = np.tile(y, (cnt, 1))
##new_X = new_X[:20000,:,:,:]
##new_y = new_y[:20000,:]
##n = len(new_X)
##X = new_X.copy()
##y = new_y.copy()
##print 'The n is :{}'.format(n)
#
##step2
model,cnn_model=build_model(width,n_class)
#
##step3
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.12, random_state=66)
#len_X_train = len(X_train)
#if len_X_train % bs != 0:
#   X_train = np.append(X_train, X_train[0:bs-len_X_train%bs,:,:,:],axis=0)
#   y_train = np.append(y_train, y_train[0:bs-len_X_train%bs,:], axis=0)
#
#len_X_valid = len(X_valid)
#if len_X_valid % bs != 0:
#   X_valid = np.append(X_valid, X_valid[0:bs-len_X_valid%bs,:,:,:],axis=0)
#   y_valid = np.append(y_valid, y_valid[0:bs-len_X_valid%bs,:], axis=0)
#print('len of X_train:{},{}; X_valid:{},{}'.format(len(X_train), len(X_train)%bs, len(X_valid), len(X_valid)%bs))
## 进行数据增加，以便进行bs多gpu训练
##if index__ == 4:
##        bs=4
##	print 'Starting expand dataset to 20000...'
##	train_sum = int(*(1-0.12))
##	valid_sum = int(20000*0.12)
##	train_cnt = train_sum/n +1
##	valid_cnt = valid_sum/n + 1
##	new_train_X = np.tile(X_train, (train_cnt, 1, 1, 1))
##	new_train_y = np.tile(y_train, (train_cnt, 1))
##	new_train_X = new_train_X[:train_sum,:,:,:]
##	new_train_y = new_train_y[:train_sum,:]
##	X_train = new_train_X.copy()
##	y_train = new_train_y.copy()
##
##
##	new_valid_X = np.tile(X_valid, (valid_cnt, 1, 1, 1))
##	new_valid_y = np.tile(y_valid, (valid_cnt, 1))
##	new_valid_X = new_valid_X[:valid_sum,:,:,:]
##	new_valid_y = new_valid_y[:valid_sum,:]
##	X_valid = new_valid_X.copy()
##	y_valid = new_valid_y.copy()
#
#n = len(X_train) + len(X_valid)
#print 'n is {}'.format(n)
#print('X_train:',X_train.shape, 'y_train:', y_train.shape)
#print('X_valid:',X_valid.shape, 'y_valid:', y_valid.shape)
##step4
##if index_ == 0:
##   model=fit_aug_model(model,epochs=50,train_batch_size=20,valid_batch_size=4) #for idx 0
##else:
##   model=fit_aug_model(model,epochs=50,train_batch_size=23,valid_batch_size=10model=fit_aug_model(model,epochs=50,train_batch_size=20,valid_batch_size=20)
#model=fit_aug_model(model,epochs=15,train_batch_size=bs,valid_batch_size=bs)
weight = np.load('./models/NASNET/Width/{0}.best.npy'.format(prefix_cls))
model.set_weights(weight)
print ('Begin testing ...')
df_load=Read_test_data(test_b_anno_path,cur_class)
n,X_test=Read_test_picture(df_load,width,test_b_pict_path)
stage_flag=2
predict_result(stage_flag,df_load,X_test)

