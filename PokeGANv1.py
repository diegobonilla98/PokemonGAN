from tensorflow.keras.layers import Dense, Conv2D, Reshape, Activation, LeakyReLU, Conv2DTranspose, Input, \
    BatchNormalization, Dropout, GaussianNoise, Flatten
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
    def __init__(self):
        self.image_dim = (256, 256, 3)
        self.noise_dim = 256
        self.batch_size = 8

        self.data_loader = DataLoader(self.batch_size, self.image_dim)
        self.sample_X = np.random.normal(0, 1, (3, self.noise_dim))

        self.layer_init = RandomNormal(mean=0.0, stddev=0.02)
        self.dis_opt = Adam(lr=0.0002, beta_1=0.5)
        self.gen_opt = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.create_discriminator()
        self.discriminator.summary()
        plot_model(self.discriminator, 'discriminator.png', show_shapes=True, show_layer_names=True)

        self.generator = self.create_generator()
        self.generator.summary()
        plot_model(self.generator, 'generator.png', show_shapes=True, show_layer_names=True)

        set_trainable(self.generator, False)
        real_img = Input(shape=self.image_dim)

        z_disc = Input(shape=(self.noise_dim,))
        fake_img = self.generator(z_disc)

        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)

        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])
        validity_interpolated = self.discriminator(interpolated_img)

        partial_gp_loss = partial(gradient_penalty_loss,
                                  interpolated_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(loss=[wasserstein, wasserstein, partial_gp_loss], optimizer=self.dis_opt, loss_weights=[1., 1., 10.])

        set_trainable(self.discriminator, False)
        set_trainable(self.generator, True)

        model_input = Input(shape=(self.noise_dim,))
        img = self.generator(model_input)
        model_output = self.discriminator(img)
        self.adversarial = Model(model_input, model_output)

        self.adversarial.compile(optimizer=self.gen_opt, loss=wasserstein)
        self.adversarial.summary()
        plot_model(self.adversarial, 'adversarial.png', show_shapes=True, show_layer_names=True)

        set_trainable(self.discriminator, True)

    def create_discriminator(self):
        input_tensor = Input(shape=self.image_dim)
        num_filters = 64

        def conv2d_block(inp, filters, strides=2):
            u = Conv2D(filters, kernel_size=4, strides=strides, padding='same', kernel_initializer=self.layer_init)(inp)
            u = BatchNormalization(momentum=0.9)(u)
            u = LeakyReLU(0.2)(u)
            u = Dropout(0.15)(u)
            return u

        x = GaussianNoise(0.1)(input_tensor)
        x = conv2d_block(x, num_filters)
        x = conv2d_block(x, num_filters)
        x = conv2d_block(x, num_filters * 2)
        x = conv2d_block(x, num_filters * 2)
        x = conv2d_block(x, num_filters * 4)
        x = conv2d_block(x, num_filters * 4)
        x = conv2d_block(x, num_filters * 8)
        x = conv2d_block(x, num_filters * 8, 1)

        x = Flatten()(x)

        output_tensor = Dense(1, kernel_initializer=self.layer_init)(x)
        # output_tensor = Activation('sigmoid')(x)

        return Model(input_tensor, output_tensor)

    def create_generator(self):
        input_tensor = Input(shape=(self.noise_dim, ))

        init_shape = (16, 16, 1024)
        x = Dense(np.prod(init_shape), kernel_initializer=self.layer_init)(input_tensor)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)

        x = Reshape(init_shape)(x)
        x = Dropout(0.3)(x)

        def up_conv2d_block(inp, filters, strides):
            u = Conv2DTranspose(filters, kernel_size=5, strides=strides, padding='same', kernel_initializer=self.layer_init)(inp)
            u = BatchNormalization(momentum=0.9)(u)
            u = LeakyReLU(0.2)(u)
            u = Dropout(0.15)(u)
            return u

        x = up_conv2d_block(x, 512, 2)
        x = up_conv2d_block(x, 256, 2)
        x = up_conv2d_block(x, 128, 2)
        x = up_conv2d_block(x, 64, 2)
        x = Conv2D(3, kernel_size=7, padding='same', kernel_initializer=self.layer_init)(x)
        output_tensor = Activation('tanh')(x)

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
                valid = np.ones((self.batch_size, 1), dtype=np.float32)
                fake = -np.ones((self.batch_size, 1), dtype=np.float32)
                dummy = np.zeros((self.batch_size, 1), dtype=np.float32)

                if np.random.rand() < 0.1:
                    valid, fake = fake, valid

                noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
                true_images = self.data_loader.load_batch()
                d_loss = self.critic_model.train_on_batch([true_images, noise], [valid, fake, dummy])

            set_trainable(self.discriminator, False)
            set_trainable(self.generator, True)
            valid = np.ones((self.batch_size, 1), dtype=np.float32)
            noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
            gen_loss = self.adversarial.train_on_batch(noise, valid)

            print(
                f"Epoch {epoch}/{epochs}:\t[Adv_loss: {gen_loss}]\t[D_loss: {d_loss}]")
            gen_losses.append(gen_loss)
            dis_losses.append(d_loss)

            if epoch % 5 == 0:
                self.plot_results(epoch)
            if epoch % 100 == 0:
                self.generator.save_weights(f'./RESULTS/weights/gen_epoch_{epoch}.h5')
            if epoch % 2 == 0:
                plt.clf()
                plt.plot(gen_losses, label="Gen Loss", alpha=0.8)
                plt.plot(dis_losses, label="Total Dis Loss", alpha=0.2)
                plt.ylim([-2, 2])
                plt.legend()
                # plt.xlim(0, num_epochs)
                plt.savefig(f'./RESULTS/metrics.png')
                plt.close()


gan = PokeGAN()
gan.train(30_000)
