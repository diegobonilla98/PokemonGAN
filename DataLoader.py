import cv2
import numpy as np
import glob
import os


class DataLoader:
    def __init__(self, batch_size, image_size):
        self.image_size = image_size[:2]
        self.batch_size = batch_size
        self.root_dir = '/media/bonilla/HDD_2TB_basura/databases/pokemon/pokemon_jpg/pokemon_jpg'
        self.images = glob.glob(os.path.join(self.root_dir, '*.jpg'))

    def _load_image(self, path):
        image = cv2.imread(path)
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
        return (image.astype(np.float32) - 127.5) / 127.5

    def load_batch(self):
        image_paths = np.random.choice(self.images, self.batch_size)
        X = [self._load_image(path) for path in image_paths]
        return np.array(X)


