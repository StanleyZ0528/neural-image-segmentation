import cv2
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from .unet import UNet

chk_path = "unet/saved_model_0.98.ckpt"  # saved best model
unet_model = UNet.load_from_checkpoint(chk_path)


def gamma_correction(input_image):
    # read image
    input_image = cv2.imread(input_image)

    # convert img to gray
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid * 255) / math.log(mean)

    # do gamma correction
    img_gamma = np.power(input_image, gamma).clip(0, 255).astype(np.uint8)
    return img_gamma


def image_resample(input_image_dir, size=(5, 4), sample_size=512):
    # crop the given image to targe samples
    img = cv2.imread(input_image_dir)
    image_samples = []
    temp_np = np.array(img)
    # row
    for row in range(size[0]):
        # col
        for col in range(size[1]):
            r_start = row * sample_size
            r_stop = r_start + sample_size
            c_start = col * sample_size
            c_stop = c_start + sample_size
            if c_stop > len(img):
                c_stop = len(img)
                c_start = c_stop - sample_size

            cur_sample_np = temp_np[c_start:c_stop, r_start:r_stop]

            # add results
            image_samples.append(cur_sample_np)
    return image_samples


def unet_predict(input_image):
    input_image = input_image[:, :, 0]
    if len(input_image.shape) == 2:
        input_image = np.expand_dims(input_image, axis=-1)
    input_image = input_image.transpose((2, 0, 1))
    if input_image.max() > 1:
        input_image = input_image / 255.

    input_image = torch.from_numpy(input_image).float()
    output_image = unet_model(input_image[None, :]).argmax(dim=1)[0].detach().numpy()
    result = np.zeros((output_image.shape[0], output_image.shape[1], 3), dtype=int)

    result[output_image == 0] = np.array([250, 170,  30])
    result[output_image == 1] = np.array([244,  35, 232])
    result[output_image == 2] = np.array([119,  11,  32])

    return result
