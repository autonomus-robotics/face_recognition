'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import time
import os
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

import keras.layers as layers
# from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet


import matplotlib.pyplot as plt

from cv2 import imread, imwrite
# from imageio import imread, imwrite


from LAST_BUILD.gan_settings import _IMG_ROWS, _IMG_COLS, _CHANNEL, \
    _TRAIN_IMG_PATH, _TRUE_PHOTOS_DIR, _BLENDER_PHOTOS_DIR, \
    _ROTATION, _LIGHTNING, \
    _TRAIN_STEPS,  _BATCH_SIZE, _SAVE_INTERVAL, \
    _OUTPUT_IMAGES_X, _OUTPUT_IMAGES_Y, \
    _MOBILENET_INPUT_SHAPE, \
    _GENERATED_FACES_PATH, _INPUT_TENSOR_SHAPE, _PERSONS








clear_session()





"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""

__COLS = 4016
__ROWS = 6016

def _get_labeled_true_photos(directory):
    for path, _, file_names in os.walk(directory):
        for file_name in file_names:
            for rotation_label in _get_rotation_labels():
                for light_label in _get_lightning_labels():

                    yield (imread(os.path.join(path, file_name)).reshape(1, _IMG_COLS * _IMG_ROWS * _CHANNEL),
                           rotation_label,
                           light_label)
                    # yield (imread(os.path.join(path, file_name)),
                    #        rotation_label,
                    #        light_label)

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

    # print(x_train.shape)
    # x_train = x_train.reshape(156, channel, rows, cols)
    # print(x_train.shape)
    return x_train


def load_mobilenet_cnn(size_output=None, default_input_shape=_MOBILENET_INPUT_SHAPE, input_tensor_shape=None,
                       batch_size=_BATCH_SIZE):
    base_model = MobileNet(include_top=False, input_shape=default_input_shape,
                           alpha=1, depth_multiplier=1,
                           dropout=0.001, weights="imagenet",
                           input_tensor=input_tensor_shape, pooling=None)
    # if any(input_tensor_shape):
    #     base_model.input_tensor = InputLayer(input_shape=input_tensor_shape, batch_size=batch_size)

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




class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()


    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"


    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )




class DCGAN(object):
    def __init__(self, img_rows=_IMG_ROWS, img_cols=_IMG_COLS, channel=_CHANNEL):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model



    # (W−F+2P)/S+1
    def discriminator(self,
                      input_output_return=False):
        if self.D:
            return self.D
        # self.D = Sequential()
        depth = 64
        dropout = 0.4

        input_shape = (self.img_rows, self.img_cols, self.channel)

        inputs = Input(shape=input_shape)

        self.x1 = Flatten()(inputs)
        self.x2 = Dense(32, activation='relu')(self.x1)
        self.x3 = Dense(16, activation='relu')(self.x2)
        self.x4 = Dense(8, activation='sigmoid')(self.x3)
        self.pred = Dense(1, activation='softmax')(self.x4)

        self.model = Model(inputs=inputs, outputs=self.pred)

        if input_output_return:
            return inputs, self.pred
        else:
            return self.model



    def generator(self,
                   compile=False,
                   input_output_return=False):

        if self.G:
            return self.G

        dropout = 0.8

        reshape_param_1 = 150
        reshape_param_2 = 3
        self.depth = reshape_param_1 * reshape_param_1 * reshape_param_2

        """
        """
        self.input_image = Input(shape=(_IMG_COLS * _IMG_ROWS * _CHANNEL,), dtype='float32')
        self.input_image_dense = Dense(64, activation='relu')(self.input_image)

        self.input_label_rotation = Input(shape=(_ROTATION,), dtype='float32')
        self.input_label_rotation_dense = Dense(64, activation='relu')(self.input_label_rotation)

        self.input_label_lightning = Input(shape=(_LIGHTNING,), dtype='float32')
        self.input_label_lightning_dense = Dense(64, activation='relu')(self.input_label_lightning)

        generator_inputs = [self.input_image, self.input_label_rotation, self.input_label_lightning]
        self.generator_input_denses = [self.input_image_dense, self.input_label_rotation_dense, self.input_label_lightning_dense]

        self.merged_inputs = Add()(self.generator_input_denses)

        self.x1 = Dense(self.depth, activation='relu')(self.merged_inputs)
        # x1 = Dense(depth), activation='relu')(x1)
        self.x1_reshaped = Reshape((reshape_param_1, reshape_param_1, reshape_param_2))(self.x1)
        self.x1_out = Dropout(dropout)(self.x1_reshaped)

        self.x2 = UpSampling2D()(self.x1_out)
        self.x2_conv2d = Conv2DTranspose(int(self.depth / 200), 5, padding='same')(self.x2)
        self.x2_batch_norm = BatchNormalization(momentum=0.9)(self.x2_conv2d)
        self.x2_out = Activation(activation='relu')(self.x2_batch_norm)

        self.x3 = UpSampling2D()(self.x2_out)
        self.x3_conv2d = Conv2DTranspose(int(self.depth / 400), 5, padding='same')(self.x3)
        self.x3_batch_norm = BatchNormalization(momentum=0.9)(self.x3_conv2d)
        self.x3_out = Activation(activation='relu')(self.x3_batch_norm)

        self.x4_conv2d = Conv2DTranspose(int(self.depth / 800), 5, padding='same')(self.x3_out)
        self.x4_batch_norm = BatchNormalization(momentum=0.9)(self.x4_conv2d)
        self.x4_out = Activation(activation='relu')(self.x4_batch_norm)

        self.x5_conv2d = Conv2DTranspose(1, 5, padding='same')(self.x4_out)
        self.x5_out = Activation(activation='relu')(self.x5_conv2d)

        # self.X6 = Dense(int(depth/16), activation='relu')(self.x5_out)
        self.x6 = Dense((3), activation='relu')(self.x5_out)

        # x6_a = Activation(activation='relu')

        model = Model(inputs=[self.input_image, self.input_label_rotation, self.input_label_lightning], outputs=self.x6)

        if compile:
            model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        model.summary()



        if input_output_return:
            return generator_inputs, self.x6
        else:
            return model



    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,
                        metrics=['accuracy'])
        return self.DM



    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        # self.AM = Sequential()
        # self.AM.add(self.generator())
        # self.AM.add(self.discriminator())
        #
        #
        # # self.AM.add(self.generator())
        # # self.AM.add(self.discriminator_model())
        #
        # self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,
        #                 metrics=['accuracy'])
        # return self.AM

        # am = Add()(inputs=(self.generator(),self.discriminator()))


        #
        # am_model.compile(loss='binary_crossentropy', optimizer=optimizer,
        #                 metrics=['accuracy'])
        """        print(type(self.discriminator()(inputs)))
        merge = conc()([self.generator().outputs,self.discriminator().inputs])

        hidden1 = Dense(10, activation='relu')(merge)
        output = Dense(1, activation='sigmoid')(hidden1)

        self.AM = Model(inputs=self.generator().inputs, outputs=output)

        self.AM.summary()
        """



        # gen_inputs, gen_outputs = self.generator(input_output_return=True)
        # dis_inputs, dis_outputs = self.discriminator(input_output_return=True)
        #
        # adv_input_image_dense = Dense(512, activation='relu')(gen_inputs[0])
        # adv_input_label_rotation_dense = Dense(512, activation='relu')(gen_inputs[1])
        # adv_input_label_lightning_dense = Dense(512, activation='relu')(gen_inputs[2])
        #
        # adv_merged_inputs = Add()([adv_input_image_dense, adv_input_label_rotation_dense, adv_input_label_lightning_dense])

        gen = self.generator()
        dis = self.discriminator()(gen.output)
        # print(dis.input)
        # print(type(dis.input))
        #
        # dis.input = gen.output
        #
        # print(dis.input)
        # print(type(dis.input))



        # adv_top_input = gen.generator_input_denses
        #
        # adv_top_merged_input = Add()(adv_top_input)
        #
        # x1 = Dense(gen.depth, activation='relu')(adv_top_merged_input)
        #
        # adv_medium_merged_input = Add()([x1,dis.inp])
        #
        # # x2 = Dense(gen.depth, activation='relu')(adv_medium_merged_input)
        #
        # self.AM = Model(inputs=adv_top_input, outputs=dis.outputs)


        self.AM = Model(inputs=gen.inputs, outputs=dis)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,
                        metrics=['accuracy'])

        return self.AM

class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = _IMG_ROWS
        self.img_cols = _IMG_COLS
        self.channel = _CHANNEL


        """
        self.x_train = input_data.read_data_sets("mnist",
                                                 one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.img_rows,
                                            self.img_cols, 1).astype(np.float32)
        """

        self.x_train = _open_and_reshape_image()

        self.DCGAN = DCGAN()
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()


    def train(self, train_steps=_TRAIN_STEPS, batch_size=_BATCH_SIZE, save_interval=_SAVE_INTERVAL,
              true_photos=_TRUE_PHOTOS_DIR):

        true_photos_with_labels = _get_labeled_true_photos(true_photos)

        blender_photos = (imread(blender_photo) for blender_photo in _get_true_photos_paths(_TRAIN_IMG_PATH))


        for i, true_photo_with_labels, blender_photo in zip(range(_ROTATION * _LIGHTNING * _PERSONS),
                                                            true_photos_with_labels, blender_photos):
            blender_photo = np.expand_dims(blender_photo, axis=0)
            img, rot_lbl, light_lbl = true_photo_with_labels

            images_fake = self.generator.predict([img, rot_lbl.reshape(1, -1), light_lbl.reshape(1, -1)])
            imwrite('face_{}.jpg'.format(i), images_fake[0])

            # print(blender_photo.shape, images_fake.shape)
            x = np.concatenate((blender_photo, images_fake))
            y = np.ones([2 * batch_size, 1])
            # print(x.shape,y.shape)
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])

            a_loss = self.adversarial.train_on_batch([img, rot_lbl.reshape(1, -1), light_lbl.reshape(1, -1)], y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval > 0:
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, samples=9,
                                     noise=true_photo_with_labels, step=(i + 1))






    def plot_images(self, save2file=False, fake=True, samples=9, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                # noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])

                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = _GENERATED_FACES_PATH
                filename += "face_{}.png".format(str(100001 + step))
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            # print(i)
            # print(images.shape)
            plt.subplot(_OUTPUT_IMAGES_X, _OUTPUT_IMAGES_Y, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols, self.channel])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            # print(filename)
            plt.close('all')
        else:
            plt.show()





if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()

    # mnist_dcgan.adversarial.layers()

    # mnist_dcgan.train(train_steps=50000, batch_size=_BATCH_SIZE, save_interval=10)
    mnist_dcgan.train(train_steps=50000, batch_size=_BATCH_SIZE, save_interval=0)
    timer.elapsed_time()
    # mnist_dcgan.plot_images(fake=True)
    # mnist_dcgan.plot_images(fake=False, save2file=True)
