'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time
import os
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, InputLayer, Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.backend import clear_session

import keras.layers as layers
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet


import matplotlib.pyplot as plt

# from cv2 import imread
from imageio import imread

from gan_settings import _IMG_ROWS, _IMG_COLS, _CHANNEL, \
    _TRAIN_IMG_PATH,\
    _TRAIN_STEPS,  _BATCH_SIZE, _SAVE_INTERVAL, \
    _OUTPUT_IMAGES_X, _OUTPUT_IMAGES_Y, \
    _MOBILENET_INPUT_SHAPE, \
    _GENERATED_FACES_PATH, _INPUT_TENSOR_SHAPE









clear_session()





"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
"""



def _open_and_reshape_image(dir=_TRAIN_IMG_PATH,
                            rows=_IMG_ROWS,
                            cols=_IMG_COLS,
                            channel=_CHANNEL):
    img_list = [(_TRAIN_IMG_PATH + one_image) for one_image in os.listdir(dir) if one_image.endswith('.jpg')]

    x_train = np.array([imread(img_path) for img_path in img_list])
    print(x_train.shape)
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
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,
                          padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D



    def generator(self):
        if self.G:
            return self.G
        # self.G = Sequential()
        # dropout = 0.4
        # depth = 64+64+64+64
        # dim = 75
        # # In: 100
        # # Out: dim x dim x depth
        # self.G.add(Dense(dim*dim*depth, input_dim=100))
        # self.G.add(BatchNormalization(momentum=0.9))
        # self.G.add(Activation('relu'))
        # self.G.add(Reshape((dim, dim, depth)))
        # self.G.add(Dropout(dropout))
        #
        # # In: dim x dim x depth
        # # Out: 2*dim x 2*dim x depth/2
        # self.G.add(UpSampling2D())
        # self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.9))
        # self.G.add(Activation('relu'))
        #
        # self.G.add(UpSampling2D())
        # self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.9))
        # self.G.add(Activation('relu'))
        #
        # self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.9))
        # self.G.add(Activation('relu'))
        #
        # self.G.add(UpSampling2D())
        # self.G.add(Conv2DTranspose(int(depth/16), 5, padding='same'))
        # self.G.add(BatchNormalization(momentum=0.9))
        # self.G.add(Activation('relu'))
        #
        # # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        # self.G.add(Conv2DTranspose(3, 5, padding='same'))
        # self.G.add(Activation('sigmoid'))

        self.G = load_mobilenet_cnn()
        self.G.compile(optimizer='adam', loss='mean_squared_error')

        self.G.summary()
        return self.G



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
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
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


    def train(self, train_steps=_TRAIN_STEPS, batch_size=_BATCH_SIZE, save_interval=_SAVE_INTERVAL):
        print(batch_size)
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[9, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            print(images_train.shape)


            noise = np.random.uniform(-1.0, 1.0, size=_INPUT_TENSOR_SHAPE)



            images_fake = self.generator.predict(noise).reshape(_IMG_ROWS, _IMG_COLS, _CHANNEL)
            print(images_fake.shape)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])

            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])

            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],
                                     noise=noise_input, step=(i+1))



    def plot_images(self, save2file=False, fake=True, samples=9, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
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
            print(i)
            print(images.shape)
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
    mnist_dcgan.train(train_steps=50000, batch_size=_BATCH_SIZE, save_interval=10)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)
