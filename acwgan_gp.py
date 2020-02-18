
# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from functools import partial
from PIL import Image

import keras.backend as K

import matplotlib.pyplot as plt

import sys
import os
import pickle
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cProfile
import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class ACWGANGP():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 8
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator([noise,label])

        # Discriminator determines validity of the real and fake images
        fake,target_label = self.critic(fake_img)
        valid,valid_label = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated, validity_label = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img,label, noise],
                            outputs=[valid, fake, validity_label, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        'sparse_categorical_crossentropy',
                                        partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 1, 10], metrics=['accuracy'])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        # Generate images based of noise
        img = self.generator([z_gen,label])
        # Discriminator determines validity
        valid,target_label = self.critic(img)
        # Defines generator model
        self.generator_model = Model([z_gen,label], [valid, target_label])
        self.generator_model.compile(loss=[self.wasserstein_loss,'sparse_categorical_crossentropy'],
                                     optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise,label], img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())


        model.summary()
        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        X_train = pickle.load(open("X.pickle", "rb"))
        y_train = pickle.load(open("y.pickle", "rb"))
        X_train = np.array(X_train, dtype=np.float32)
        #(X_train, y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = np.array(y_train).reshape(-1, 1)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        Dloss = []
        Gloss = []
        DAcc = []
        DopAcc = []
        EpochNum = []
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                img_labels = y_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # The labels of the digits that the generator tries to create an
                # image representation of
                sampled_labels = np.random.randint(0, 8, (batch_size, 1))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict([noise, sampled_labels])

                # Train the critic
                d_loss_real = self.critic_model.train_on_batch([imgs, img_labels, noise],
                                                                [valid, fake, img_labels, dummy])
                d_loss_fake = self.critic_model.train_on_batch([gen_imgs, sampled_labels, noise],
                                                                [valid, fake, sampled_labels, dummy])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            g_loss = self.generator_model.train_on_batch([noise,sampled_labels], [valid,sampled_labels])

            # Plot the progress

            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[6], 100*d_loss[7], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch % 100 == 0:
                Dloss.append(d_loss[0])
                Gloss.append(g_loss[0])
                EpochNum.append(epoch)
                DAcc.append(d_loss[6])
                DopAcc.append(d_loss[7])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(EpochNum, Dloss, 'g', linestyle='solid', label='D-Loss')
        plt.plot(EpochNum, Gloss, 'b', linestyle='solid', label='G-loss')
        plt.plot(EpochNum, DAcc, 'y', linestyle='solid', label='acc')
        plt.plot(EpochNum, DopAcc, 'r', linestyle='solid', label='op-acc')
        plt.legend(loc='upper left')
        plt.show()

    def sample_images(self, epoch):
        r, c = 8, 8
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/mnist%d.png" % epoch)
        plt.close()
        #noise = np.random.normal(0, 1, (1, self.latent_dim))
        #gen_imgs = self.generator.predict(noise)
        #im = np.array(gen_imgs[0, :, :, 0])
        #newim = Image.fromarray(im)

        # newim = newim.convert('L')
        #newim.save("images/%d.tiff" % epoch, quality=90)


if __name__ == '__main__':
    acwgan = ACWGANGP()
    acwgan.train(epochs=14000, batch_size=32, sample_interval=100)
