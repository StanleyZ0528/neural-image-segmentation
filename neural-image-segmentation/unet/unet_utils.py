import cv2
import math
import numpy as np
from .unet import UNet

chk_path = "/saved_model_0.98.ckpt" # saved best model
unet_model = UNet.load_from_checkpoint(chk_path)

def gamma_correction(input_image):
    # convert img to gray
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid * 255) / math.log(mean)
    print(gamma)

    # do gamma correction
    img_gamma = np.power(input_image, gamma).clip(0, 255).astype(np.uint8)
    return img_gamma


def unet_predict(input_image):
    return unet_model(input_image[None,:]).argmax(dim=1)[0].detach().numpy()


