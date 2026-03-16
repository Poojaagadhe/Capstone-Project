import numpy as np
import torch
import cv2


def preprocess_image(image):

    image = cv2.resize(image, (256, 256))

    image = image.astype("float32")

    image = (image - image.min()) / (image.max() - image.min())

    image = image * 2 - 1

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    image = torch.tensor(image)

    return image
