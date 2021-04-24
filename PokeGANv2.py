from tensorflow.keras.layers import Dense, Conv2D, Reshape, Activation, LeakyReLU, Conv2DTranspose, Input, \
    BatchNormalization, Dropout, GaussianNoise, Flatten, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
from functools import partial

from utils import RandomWeightedAverage, wasserstein, set_trainable, gradient_penalty_loss
from DataLoader import DataLoader

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class PokeGAN:
    def __init__(self, is_test=False):
        self.image_dim = (128, 128, 3)
        self.noise_dim = 100
        self.batch_size = 32

        self.generator = self.create_generator()
        self.generator.summary()
        plot_model(self.generator, 'generator.png', show_shapes=True, show_layer_names=True)
        if not is_test:
            self.discriminator = self.create_discriminator()
            self.discriminator.summary()
            plot_model(self.discriminator, 'discriminator.png', show_shapes=True, show_layer_names=True)

            self.G_train, self.D_train = self.build_WGANgp(self.generator, self.discriminator)

            self.data_loader = DataLoader(self.batch_size, self.image_dim)
            self.sample_X = np.random.normal(0, 1, (3, self.noise_dim))

    def build_WGANgp(self, generator, discriminator):
        z = Input(shape=(self.noise_dim,))
        f_img = generator(z)
        f_out = discriminator(f_img)

        r_img = Input(shape=self.image_dim)
        r_out = discriminator(r_img)

        epsilon = K.placeholder(shape=(None, 1, 1, 1))
        a_img = Input(shape=self.image_dim,
                      tensor=epsilon * r_img + (1 - epsilon) * f_img)
        a_out = discriminator(a_img)

        r_loss = K.mean(r_out)
        f_loss = K.mean(f_out)

        grad_mixed = K.gradients(a_out, [a_img])[0]
        norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
        grad_penalty = K.mean(K.square(norm_grad_mixed - 1))
        penalty = 10. * grad_penalty

        d_loss = f_loss - r_loss + penalty

        d_updates = Adam(lr=0.0002, beta_1=0.5).get_updates(d_loss, discriminator.trainable_weights)
        d_train = K.function([r_img, z, epsilon],
                             [r_loss, f_loss, penalty, d_loss],
                             d_updates)

        g_loss = -1. * f_loss
        g_updates = Adam(lr=0.0002, beta_1=0.5).get_updates(g_loss, generator.trainable_weights)
        g_train = K.function([z], [g_loss], g_updates)

        return g_train, d_train

    def create_discriminator(self):
        input_tensor = Input(shape=self.image_dim)

        x = Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False)(input_tensor)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(1, kernel_size=1, strides=1, padding="same", use_bias=False)(x)
        x = Flatten()(x)
        output_tensor = Dense(units=1, activation=None)(x)

        return Model(input_tensor, output_tensor)

    def create_generator(self):
        input_tensor = Input(shape=(self.noise_dim,))

        x = Dense(16 * 16 * 512, activation="relu")(input_tensor)
        x = Reshape((16, 16, 512))(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64 * 4, kernel_size=3, strides=1, padding="same",
                   use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Conv2D(64 * 4, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, )(x, training=1)
        x = Activation("relu")(x)
        x = Conv2D(64 * 4, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Activation("relu")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64 * 2, kernel_size=3, strides=1, padding="same",
                   use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Conv2D(64 * 2, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, )(x, training=1)
        x = Activation("relu")(x)
        x = Conv2D(64 * 2, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Activation("relu")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64 * 1, kernel_size=3, strides=1, padding="same",
                   use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Conv2D(64 * 1, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, )(x, training=1)
        x = Activation("relu")(x)
        x = Conv2D(64 * 1, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Activation("relu")(x)
        output_tensor = Conv2D(3, kernel_size=3, strides=1, padding="same", activation="tanh",
                               use_bias=False)(x)

        return Model(input_tensor, output_tensor)

    def plot_results(self, epoch):
        fake = self.generator.predict(self.sample_X)
        res = (np.hstack([fake[0, :, :, :], fake[1, :, :, :], fake[2, :, :, :]]) + 1) / 2
        plt.clf()
        # plt.figure(figsize=(10, 15))
        plt.imshow(res)
        plt.axis('off')
        plt.savefig(f'./RESULTS/images/epoch_{epoch}.jpg')
        plt.close()

    def train(self, epochs):
        dis_losses = []
        gen_losses = []
        for epoch in range(epochs):
            set_trainable(self.discriminator, True)
            set_trainable(self.generator, False)
            for _ in range(5):
                # Generator in
                z = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
                # Generator out Images
                f_imgs = self.generator.predict(z)
                # Real Images
                r_imgs = self.data_loader.load_batch()
                # train the discriminator
                epsilon = np.random.uniform(size=(self.batch_size, 1, 1, 1))
                r_loss, f_loss, penalty, d_loss = self.D_train([r_imgs, z, epsilon])

                #### Generator
                # Generator in
            z = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
            # train the generator
            g_loss = self.G_train([z])

            print(
                f"Epoch {epoch}/{epochs}:\t[Adv_loss: {g_loss}]\t[D_loss: {d_loss}]")
            gen_losses.append(g_loss)
            dis_losses.append(d_loss)

            if epoch % 5 == 0:
                self.plot_results(epoch)
            if epoch % 100 == 0:
                self.generator.save_weights(f'./RESULTS/weights/gen_epoch_{epoch}.h5')
            if epoch % 2 == 0:
                plt.clf()
                plt.plot(gen_losses, label="Gen Loss", alpha=0.8)
                plt.plot(dis_losses, label="Total Dis Loss", alpha=0.2)
                plt.legend()
                # plt.xlim(0, num_epochs)
                plt.savefig(f'./RESULTS/metrics.png')
                plt.close()


# gan = PokeGAN()
# gan.train(90_000)
