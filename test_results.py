from PokeGANv2 import PokeGAN
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

gan = PokeGAN(is_test=True)

weights_path = './RESULTS/weights/gen_epoch_25000.h5'
generator = gan.generator
generator.load_weights(weights_path)

noise_sample = np.random.normal(0, 1, (122, gan.noise_dim))
generated = generator.predict(noise_sample)

fig, axs = plt.subplots(11, 11)
for x in range(11):
    for y in range(11):
        img = (generated[y+x*11, :, :, ::-1] + 1) / 2
        axs[y, x].imshow(img)
        axs[y, x].axis('off')
plt.show()
