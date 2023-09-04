'''
This file create datasets, neural network model and train it 
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  #to use CPU not GPU

import tensorflow as tf
import json
from keras.applications.vgg16 import VGG16
from nn_model import *

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def load_to_dataset(part):
    dataset = tf.data.Dataset.list_files(f'data\\aug_data\\{part}\\images\\*.jpg', shuffle=False)
    dataset = dataset.map(load_image)
    dataset = dataset.map(lambda x: tf.image.resize(x, (128,128)))
    dataset = dataset.map(lambda x: x/255)
    return dataset

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)
    return [label['class']], label['bbox']

def create_final_dataset(images, labels, shuffle, batch, prefetch):
    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)
    return dataset

train_images = load_to_dataset('train')
test_images = load_to_dataset('test')
validate_images = load_to_dataset('validate')

train_labels = tf.data.Dataset.list_files('data\\aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('data\\aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

validate_labels = tf.data.Dataset.list_files('data\\aug_data\\validate\\labels\\*.json', shuffle=False)
validate_labels = validate_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

train_ds = create_final_dataset(train_images, train_labels, 5000, 8, 4)
test_ds = create_final_dataset(test_images, test_labels, 1300, 8, 4)
validate_ds = create_final_dataset(validate_images, validate_labels, 1000, 8, 4)

def build_model():
    input_layer = tf.keras.layers.Input(shape=(128,128,3))
    vgg = tf.keras.applications.VGG16(include_top=False)(input_layer)
    f1 = tf.keras.layers.GlobalMaxPooling2D()(vgg)
    class1 = tf.keras.layers.Dense(2048, activation='relu')(f1)
    class2 = tf.keras.layers.Dense(1, activation='sigmoid')(class1)

    f2 = tf.keras.layers.GlobalMaxPooling2D()(vgg)
    regress1 = tf.keras.layers.Dense(2048, activation='relu')(f2)
    regress2 = tf.keras.layers.Dense(4, activation='sigmoid')(regress1)

    facetracker = tf.keras.models.Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

facetracker = build_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

def loc_loss(y_true, y_pred):
    delta_cords = tf.reduce_sum(tf.square(y_true[:,:2] - y_pred[:,:2]))
    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]

    h_pred = y_pred[:,3] - y_pred[:,1]
    w_pred = y_pred[:,2] - y_pred[:,0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_cords + delta_size

class_loss = tf.keras.losses.BinaryCrossentropy()

reg_loss = loc_loss

model = FaceTracker(facetracker)

model.compile(optimizer, class_loss, reg_loss)

hist = model.fit(train_ds, epochs=10, validation_data=validate_ds)

facetracker.save('facetracker.h5')