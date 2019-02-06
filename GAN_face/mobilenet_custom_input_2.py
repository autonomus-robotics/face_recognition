"""
########################################################################################################################
imports
########################################################################################################################
"""
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'

# from tensorflow import RunOptions
# run_options = RunOptions(report_tensor_allocations_upon_oom = True)
# sess.run(op, feed_dict=fdict, options=run_options)


import numpy as np
import time
# from tensorflow.examples.tutorials.mnist import input_data

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


import matplotlib.pyplot as plt

# from cv2 import imread, imwrite
from imageio import imread, imwrite


from gan_settings import _IMG_ROWS, _IMG_COLS, _CHANNEL, \
    _TRAIN_IMG_PATH, _TRUE_PHOTOS_DIR, _BLENDER_PHOTOS_DIR, \
    _ROTATION, _LIGHTNING, \
    _TRAIN_STEPS,  _BATCH_SIZE, _SAVE_INTERVAL, \
    _OUTPUT_IMAGES_X, _OUTPUT_IMAGES_Y, \
    _MOBILENET_INPUT_SHAPE, \
    _GENERATED_FACES_PATH, _INPUT_TENSOR_SHAPE, _PERSONS



clear_session()







"""
########################################################################################################################
Local variables
########################################################################################################################
"""

__COLS = 600
__ROWS = 600
# _TRAIN_STEPS = 1000
_NAME = 'mobilenet'







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

def train(gen, train_steps=_TRAIN_STEPS, batch_size=_BATCH_SIZE, save_interval=_SAVE_INTERVAL,
          true_photos=_TRUE_PHOTOS_DIR):
    true_photos_with_labels = _get_labeled_true_photos(_TRUE_PHOTOS_DIR)

    blender_photos = (imread(blender_photo) for blender_photo in _get_true_photos_paths(_TRAIN_IMG_PATH))

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

        print(images_fake.shape)

        imwrite('faces_mobilenet_generator\\face_{}.jpg'.format(i), images_fake)
        print(i, "/", _TRAIN_STEPS, "epoches")
        gen.train_on_batch([img, rot_lbl, light_lbl], blender_photo)
        if i == _TRAIN_STEPS: break
#     bot.send("Generator finished training")







"""
########################################################################################################################
TRAIN FUNCTION
########################################################################################################################
"""


def generator(_compile=False,
              input_output_return=False,
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

    generator_inputs = [input_image, input_label_rotation, input_label_lightning]
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
    print(fc1.shape)
    fc2 = layers.Dense(_IMG_ROWS * _IMG_COLS * _CHANNEL, activation='tanh', name='fc2_dense')(fc1)

    """
    x1_ = Dense(int(depth / 8), activation='relu', name='newnew')(x1_out)

    x2 = UpSampling2D()(x1_)
    x2_conv2d = Conv2DTranspose(int(depth / 200), 5, padding='same')(x2)
    x2_batch_norm = BatchNormalization(momentum=0.99)(x2_conv2d)
    x2_drop = Dropout(dropout)(x2_batch_norm)
    x2_out = Activation(activation='relu')(x2_drop)

    x3 = UpSampling2D()(x2_out)
    x3_conv2d = Conv2DTranspose(int(depth / 400), 5, padding='same')(x3)
    x3_batch_norm = BatchNormalization(momentum=0.99)(x3_conv2d)
    x3_drop = Dropout(dropout / 2)(x3_batch_norm)
    x3_out = Activation(activation='relu')(x3_drop)

    x4_conv2d = Conv2DTranspose(int(depth / 800), 5, padding='same')(x3_out)
    x4_batch_norm = BatchNormalization(momentum=0.99)(x4_conv2d)
    x4_drop = Dropout(dropout / 2)(x3_out)
    x4_out = Activation(activation='relu')(x4_batch_norm)

    x5_conv2d = Conv2DTranspose(1, 5, padding='same')(x4_out)
    x5_out = Activation(activation='relu')(x5_conv2d)

    x6 = Dense(int(depth / 128), activation='relu')(x5_out)
    xOut = Dense((3), activation='relu')(x6)
    """

    # model = Model(inputs=[input_image, input_label_rotation, input_label_lightning], outputs=xOut)
    model = Model(inputs=[input_image, input_label_rotation, input_label_lightning], outputs=fc2)
    opt = RMSprop(lr=0.001, decay=0.1, epsilon=0.1)



    if _compile:
        model.compile(loss='mean_squared_error', optimizer=opt,
                      metrics=['accuracy'])

    model.summary()
    from contextlib import redirect_stdout

    with open(_NAME + '.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    plot_model(model, to_file='mobilenet_custom_input.png', show_shapes=True)

    if input_output_return:
        return generator_inputs, x6
    else:
        return model








"""
########################################################################################################################
Let's do it
########################################################################################################################
"""
if __name__=='__main__':

    gen = generator(True)

    train(gen)

# # clear_session()
# # import tensorflow as tf
# # print(tf.__version__)
# train(gen)









