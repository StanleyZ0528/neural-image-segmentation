import os
import torch
import cv2
import numpy as np
from unet import UNet
from dataset import DirDataModule
from pytorch_lightning import Trainer

image_path= "image/Snap-563.tif"
mask_path= "mask/Snap-563_all_objects.png"

def import_data():
    mask_map = {
        (250, 170,  30): 0, # Background
        (244,  35, 232): 1, # Cell
        (119,  11,  32): 2, # Neurite
        (255, 255, 255): 3  # Border
    }
    mask = cv2.imread(mask_path)
    mask = np.array(mask)
    mask = torch.from_numpy(mask).long()

    input_image = cv2.imread(image_path)
    input_image = input_image[:, :, 0]
    if len(input_image.shape) == 2:
        input_image = np.expand_dims(input_image, axis=-1)
    input_image = input_image.transpose((2, 0, 1))
    if input_image.max() > 1:
        input_image = input_image / 255.

    input_image = torch.from_numpy(input_image).float()

    result = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

    result[mask == [0, 0, 0]] = 0
    result[mask == [255,  0, 255]] = 1
    result[mask == [255,  129,  31]] = 2
    result[mask == [255,  255,  255]] = 3

    mask = torch.from_numpy(result).long()

    return input_image, mask

def main():
    chk_path = "saved_model_0.98.ckpt"
    img, y = import_data()
    
    # model = UNet(lr = 0.0001,
    #         num_classes = 3,
    #         num_layers = 5,
    #         input_channels = 1,
    #         features_start = 64,
    #         bilinear = False,
    #         loss_weight = [.5, 3, 5],
    #         ignore_index = 3)
    
    # trainer = Trainer()
    unet_model = UNet.load_from_checkpoint(chk_path)
    # trainer.test(best_model, dataloaders=dm)
    y_bar = unet_model(img[None, :]).argmax(dim=1)[0]
    print(y_bar.shape)

    fg_mask = (y==1).logical_or(y==2)
    ne_mask = (y==2)
    val_acc    = torch.sum(y_bar==y).item()/(torch.numel(y))

    print("acc: ", val_acc)


if __name__ == '__main__':
    main()