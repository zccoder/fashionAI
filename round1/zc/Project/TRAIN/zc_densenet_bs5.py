# encoding=utf-8
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
index_ = int(sys.argv[1])
width = 334

train_csv_path = '/home2/data/fashionAI/train_data/Annotations/train_all.csv'
train_pict_path = '/home2/data/fashionAI/train_data'
test_anno_path = '/home2/data/fashionAI/test_data/Tests/question.csv'
test_pict_path = '/home2/data/fashionAI/test_data'


# train_pict_path='/home/flypiggy/Downloads/Shen/Fashion_AI_group/Data_set/base'
# test_anno_path='/home/flypiggy/Downloads/Shen/Fashion_AI_group/Data_set/rank/Tests/question.csv'
# test_pict_path='/home/flypiggy/Downloads/Shen/Fashion_AI_group/Data_set/rank'

def Read_anno_data(index_):
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
    return df_load, cur_class, prefix_cls


def Read_picture_data(df_load, width):
    n = len(df_load)
    n_class = len(df_load['label'][0])
    print("nclass", n_class)

    X = np.zeros((n, width, width, 3), dtype=np.uint8)
    y = np.zeros((n, n_class), dtype=np.uint8)

    for i in range(n):
        tmp_label = df_load['label'][i]
        if len(tmp_label) > n_class:
            print(df_load['image_id'][i])
        X[i] = cv2.resize(cv2.imread('/home2/data/fashionAI/train_data/{0}'.format(df_load['image_id'][i])),
                          (width, width))
        y[i][tmp_label.find('y')] = 1

    return X, y, n, n_class


def build_model(width, n_class):
    cnn_model = DenseNet201(include_top=False, input_shape=(width, width, 3), weights=None)
    cnn_model.load_weights('./imagenet_weights/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # cnn_model.save('InceptionResNetV2.h5')
    # cnn_model.load_weights('InceptionResNetV2.h5')
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='softmax', name='softmax')(x)
    model = Model(inputs, x)
    # model = keras.utils.training_utils.multi_gpu_model(model, gpus=2)
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model, cnn_model


def fit_aug_model(model, epochs=7, train_batch_size=26, valid_batch_size=10):
    print("begin data aug:")
    checkpointer = ModelCheckpoint(filepath='./models/DenseNet201/BS5/{0}.best.h5'.format(prefix_cls), verbose=1, monitor='val_loss', save_best_only=True)
    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)
    test_datagen = ImageDataGenerator()
    test_datagen.fit(X_valid)
    h = model.fit_generator(datagen.flow(X_train, y_train, batch_size=train_batch_size),
                            callbacks=[EarlyStopping(patience=4), checkpointer],
                            epochs=epochs, shuffle=True,
                            validation_data=test_datagen.flow(X_valid, y_valid, batch_size=valid_batch_size))
    return model


def setup_to_finetune(base_model, _stage=2, top_N_layer=-3):
    if (_stage == 1):
        for layer in base_model.layers[:top_N_layer]:
            layer.trainable = False
        for layer in base_model.layers[top_N_layer:]:
            layer.trainable = True
            print("layer name True:", layer.name)
    if (_stage == 2):
        for layer in base_model.layers:
            layer.trainable = True


def fit_fine_tuning(model, cnn_model, top_n_layer=-3, fine_epoches=5, fine_batch_size=10):
    setup_to_finetune(cnn_model, _stage=2, top_N_layer=top_n_layer)
    adam = Adam(lr=0.00001)  # 此处以非常低的学习率微调
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model = fit_aug_model(model, epochs=fine_epoches, train_batch_size=fine_batch_size, valid_batch_size=5)
    return model


def Read_test_data(test_anno_path, cur_class):
    df_test = pd.read_csv('%s' % test_anno_path, header=None)
    df_test.columns = ['image_id', 'class', 'x']
    del df_test['x']

    df_load = df_test[(df_test['class'] == cur_class)].copy()
    df_load.reset_index(inplace=True)
    del df_load['index']
    del df_test
    print('{0}: {1}'.format(cur_class, len(df_load)))
    return df_load


def Read_test_picture(df_load, width, test_pict_path):
    n = len(df_load)
    X_test = np.zeros((n, width, width, 3), dtype=np.uint8)
    for i in range(n):
        X_test[i] = cv2.resize(cv2.imread('/home2/data/fashionAI/test_data/{0}'.format(df_load['image_id'][i])),
                               (width, width))
    return n, X_test


def predict_result(stage_flag, df_load, X_test):
    test_np = model.predict(X_test)
    result = []
    for i, row in df_load.iterrows():
        tmp_list = test_np[i]
        tmp_result = ''
        for tmp_ret in tmp_list:
            tmp_result += '{:.4f};'.format(tmp_ret)

        result.append(tmp_result[:-1])

    df_load['result'] = result
    if (stage_flag == 1):
        df_load.to_csv('./fine_result/DenseNet201/BS5/{}_stage1.csv'.format(prefix_cls), header=None, index=False)
    elif (stage_flag == 2):
        df_load.to_csv('./fine_result/DenseNet201/BS5/{}_stage2.csv'.format(prefix_cls), header=None, index=False)


# step1
import time

t1 = time.time()
df_load, cur_class, prefix_cls = Read_anno_data(index_)
X, y, n, n_class = Read_picture_data(df_load, width)

# step2
model, cnn_model = build_model(width, n_class)

# step3
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.12, random_state=66)

# step4
model = fit_aug_model(model, epochs=16, train_batch_size=5, valid_batch_size=5)
print(model.evaluate(X_valid, y_valid, batch_size=5))
model.load_weights('./models/DenseNet201/BS5/{0}.best.h5'.format(prefix_cls))

df_load = Read_test_data(test_anno_path, cur_class)
n, X_test = Read_test_picture(df_load, width, test_pict_path)
stage_flag = 1
predict_result(stage_flag, df_load, X_test)

#

# step5
model = fit_fine_tuning(model, cnn_model, top_n_layer=-3, fine_epoches=16, fine_batch_size=5)
model.load_weights('./models/DenseNet201/BS5/{0}.best.h5'.format(prefix_cls))
print(model.evaluate(X_valid, y_valid, batch_size=5))

# step6
# step7
stage_flag = 2
predict_result(stage_flag, df_load, X_test)

# step8
#model.save('./fine_model_backup/DenseNet201/%s_dense_sgd_0328.h5' % prefix_cls)
#print("all time:", time.time() - t1)
