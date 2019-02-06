"""
########################################################################################################################
imports
########################################################################################################################
"""
from gan_settings import _IMG_ROWS, _IMG_COLS, _CHANNEL, \
    _TRAIN_IMG_PATH, _TRUE_PHOTOS_DIR, _IMAGES_SAVE_PATH, \
    _GRAPHVIZ_PATH,_ROTATION, _LIGHTNING, \
    _TRAIN_STEPS,  _BATCH_SIZE, _SAVE_INTERVAL, _SUMMARY_PATH, \
    _OUTPUT_IMAGES_X, _OUTPUT_IMAGES_Y, \
    _MOBILENET_INPUT_SHAPE, \
    _INPUT_TENSOR_SHAPE, _PERSONS


import numpy as np
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, InputLayer, Input, Add
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout, LSTM
from keras.layers.merge import concatenate as mconc
from keras.layers import Concatenate as conc

from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.backend import clear_session
from keras.utils import to_categorical

from keras.utils import plot_model

import keras.layers as layers
# from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet

from contextlib import redirect_stdout
import matplotlib.pyplot as plt

# from cv2 import imread, imwrite
from imageio import imread, imwrite

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["PATH"] += os.pathsep + _GRAPHVIZ_PATH




clear_session()







"""
########################################################################################################################
Local variables
########################################################################################################################
"""










"""
########################################################################################################################
FUNCTIONS
########################################################################################################################
"""

def _get_labeled_true_photos(directory):
    for path, _, file_names in os.walk(directory):
        for file_name in file_names:
            for rotation_label in _get_rotation_labels():
                for light_label in _get_lightning_labels():
                    yield (imread(os.path.join(path, file_name)).reshape(1, _IMG_COLS * _IMG_ROWS * _CHANNEL),
                           rotation_label.reshape(1, -1),
                           light_label.reshape(1, -1))


def _get_true_photos_paths(directory):
    for path, _, file_names in os.walk(directory):
        for file_name in file_names:
            yield os.path.join(path, file_name)


def _get_rotation_labels(rotation=_ROTATION):
    return to_categorical(list(range(rotation)))


def _get_lightning_labels(light=_LIGHTNING):
    return to_categorical(list(range(light)))


def _open_and_reshape_image(dir=_TRAIN_IMG_PATH,
                            rows=_IMG_ROWS,
                            cols=_IMG_COLS,
                            channel=_CHANNEL):
    img_list = ((_TRAIN_IMG_PATH + one_image) for one_image in os.listdir(dir) if one_image.endswith('.jpg'))

    x_train = np.array([imread(img_path) for img_path in img_list])
    return x_train


def load_mobilenet_cnn(size_output=None, default_input_shape=_MOBILENET_INPUT_SHAPE, input_tensor_shape=None,
                       batch_size=_BATCH_SIZE):
    base_model = MobileNet(include_top=False, input_shape=default_input_shape,
                           alpha=1, depth_multiplier=1,
                           dropout=0.001, weights="imagenet",
                           input_tensor=input_tensor_shape, pooling=None)

    # add fully connected layers
    fc0 = base_model.output
    fc0_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='fc0_pool')(fc0)
    fc1 = layers.Dense(256, activation='relu', name='fc1_dense')(fc0_pool)
    fc2 = layers.Dense(_IMG_ROWS * _IMG_COLS * _CHANNEL, activation='tanh', name='fc2_dense')(fc1)

    model = Model(inputs=base_model.input, outputs=fc2)

    # freeze the early layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='sgd', loss='mean_squared_error')

    return model







"""
########################################################################################################################
TRAIN FUNCTION
########################################################################################################################
"""

def train(gen,
          train_steps=_TRAIN_STEPS,
          save_interval=_SAVE_INTERVAL,
          train_images=_TRAIN_IMG_PATH,
          true_photos=_TRUE_PHOTOS_DIR):

    true_photos_with_labels = _get_labeled_true_photos(true_photos)

    blender_photos = (imread(blender_photo) for blender_photo in _get_true_photos_paths(train_images))

    #     bot.send("Generator started training")
    for i, true_photo_with_labels, blender_photo in zip(range(_ROTATION * _LIGHTNING * _PERSONS),
                                                        true_photos_with_labels, blender_photos):
        blender_photo = np.expand_dims(blender_photo, axis=0).reshape(1, _IMG_COLS * _IMG_ROWS * _CHANNEL)
        img, rot_lbl, light_lbl = true_photo_with_labels
        rot_lbl = rot_lbl.reshape(1, -1)
        light_lbl = light_lbl.reshape(1, -1)

        images_fake = gen.predict([img, rot_lbl, light_lbl])
        images_fake = images_fake.reshape(600, 600, 3)
        #         os.mkdir

        if int(i%save_interval)==0:
            imwrite(_IMAGES_SAVE_PATH + 'face_{}.jpg'.format(i), images_fake)


        print(i, "/", train_steps, "epoches")

        gen.train_on_batch([img, rot_lbl, light_lbl], blender_photo)
        if i == train_steps: break
#     bot.send("Generator finished training")







"""
########################################################################################################################
TRAIN FUNCTION
########################################################################################################################
"""


def generator(_compile=False,
              do_plot=True,
              ):
    """
    """
    dropout = 0.8

    reshape_param_1 = 150
    reshape_param_2 = 3
    depth = reshape_param_1 * reshape_param_1 * reshape_param_2


    input_image = Input(shape=(_IMG_COLS * _IMG_ROWS * _CHANNEL,), dtype='float32', name='input_image')
    input_image_dense = Dense(64, activation='relu', name='input_image_dense')(input_image)

    input_label_rotation = Input(shape=(_ROTATION,), dtype='float32', name='input_rotation')
    input_label_rotation_dense = Dense(64, activation='relu', name='input_label_rotation_dense')(input_label_rotation)

    input_label_lightning = Input(shape=(_LIGHTNING,), dtype='float32', name='input_lightning')
    input_label_lightning_dense = Dense(64, activation='relu', name='input_label_lightning_dense')(
        input_label_lightning)

    # generator_inputs = [input_image, input_label_rotation, input_label_lightning]
    generator_input_denses = [input_image_dense, input_label_rotation_dense, input_label_lightning_dense]

    merged_inputs = Add(name='add_generator_input_denses')(generator_input_denses)

    x1 = Dense(depth, activation='relu', name='')(merged_inputs)
    x1_reshaped = Reshape((reshape_param_1, reshape_param_1, reshape_param_2))(x1)
    x1_out = Dropout(dropout)(x1_reshaped)

    """
    """

    mobile_net_cnn = MobileNet(include_top=False,
                               alpha=1, depth_multiplier=1,
                               dropout=0.001, weights="imagenet",
                               pooling=None)

    mobile_layers = [one_layer for one_layer in mobile_net_cnn.layers]

    mx = x1_out

    for i in range(1, len(mobile_layers)):
        mobile_layers[i].trainable = False
        mx = mobile_layers[i](mx)

    fc0_pool = layers.GlobalAveragePooling2D(data_format='channels_last', name='fc0_pool')(mx)
    fc1 = layers.Dense(256, activation='relu', name='fc1_dense')(fc0_pool)
    # print(fc1.shape)
    fc2 = layers.Dense(_IMG_ROWS * _IMG_COLS * _CHANNEL, activation='tanh', name='fc2_dense')(fc1)

    model = Model(inputs=[input_image, input_label_rotation, input_label_lightning], outputs=fc2)
    opt = RMSprop(lr=0.001, decay=0.1, epsilon=0.1)



    if _compile:
        model.compile(loss='mean_squared_error', optimizer=opt,
                      metrics=['accuracy'])

    model.summary()

    with open(_SUMMARY_PATH + '.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    if do_plot:
        plot_model(model, to_file='mobilenet_custom_input.png', show_shapes=True)

    return model








"""
########################################################################################################################
Let's do it
########################################################################################################################
"""
if __name__=='__main__':

    gen = generator(True,False)

    train(gen)

# # clear_session()
# # import tensorflow as tf
# # print(tf.__version__)
# train(gen)