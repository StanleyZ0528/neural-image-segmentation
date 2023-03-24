import cv2
import glob
import math
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex

from .unet import UNet

chk_path = "unet/saved_model_0.98.ckpt"  # saved best model
small_chk_path = "unet/saved_small_model.ckpt"  # saved small model
unet_model = UNet.load_from_checkpoint(chk_path)
small_unet_model = UNet.load_from_checkpoint(small_chk_path)


def gamma_correction(input_image):
    # # read image
    # input_image = cv2.imread(input_image)

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

def image_resample_row(input_image_dir, size = (4, 5), sample_size = 512):
    # crop the given image to targe samples
    img = cv2.imread(input_image_dir)
    image_samples = []
    temp_np = np.array(img)
    # row
    for row in range(size[0]):
        # col
        for col in range(size[1]):
            r_start = row*sample_size
            r_stop = r_start+sample_size
            c_start = col*sample_size
            c_stop = c_start+sample_size
            if r_stop > len(img):
                r_stop = len(img)
                r_start = r_stop-sample_size
            
            cur_sample_np = temp_np[r_start:r_stop, c_start:c_stop]

            # add results
            image_samples.append(cur_sample_np)
    return image_samples

def unet_predict(input_image, model="best"):
    input_image = input_image[:, :, 0]
    if len(input_image.shape) == 2:
        input_image = np.expand_dims(input_image, axis=-1)
    input_image = input_image.transpose((2, 0, 1))
    if input_image.max() > 1:
        input_image = input_image / 255.

    input_image = torch.from_numpy(input_image).float()
    if model == "best":
        output_image = unet_model(input_image[None, :]).argmax(dim=1)[0].detach().numpy()
    elif model == "small":
        output_image = small_unet_model(input_image[None, :]).argmax(dim=1)[0].detach().numpy()
    result = np.zeros((output_image.shape[0], output_image.shape[1], 3), dtype=int)

    result[output_image == 0] = np.array([0, 0, 0])
    result[output_image == 1] = np.array([255, 0, 255])
    result[output_image == 2] = np.array([255, 129, 31])
    result[output_image == 3] = np.array([255,  255,  255])

    return result

def mask_stitching_loop(masks, overlap_size=128, shape=(4,5)):
    row_masks = []

    for idx, mask in enumerate(masks):
        row = idx // shape[1]
        col = idx - (row * shape[1])

        if row == shape[0] - 1:
            mask = mask[overlap_size:,]
        if len(row_masks) < row + 1:
            row_masks.append(mask)
        else:
            row_masks[row] = np.hstack((row_masks[row], mask))

    complete_mask = np.vstack(row_masks)

    return complete_mask

def mask_stitching(masks, overlap_size=128, shape=(4,5)):
    row_block = []
    for i in range(0, len(masks), shape[1]):
        if i == 15:
            break
        j = i + shape[1]
        row_block.append(np.hstack(tuple(masks[i:j])))
    
    last_row = np.hstack(tuple(masks[15:20]))
    row_block.append(last_row[overlap_size:,])

    complete_mask = np.vstack(row_block)

    return complete_mask

def get_weight_matrix(size):
    N = int(size / 2)
    area_of_interest = int(size - (size*0.1)) if int(size - (size*0.1)) % 2 == 0 else int(size - (size*0.1)) - 1
    base_array = np.array([[N for _ in range(area_of_interest)] for _ in range(area_of_interest)])
    pad_width = size - area_of_interest
    y = np.pad(base_array, (pad_width, pad_width), 'linear_ramp', end_values=(pad_width,pad_width)).astype(float)

    return y

if __name__ == '__main__':
    img_path = '902-complete.tif'
    mask_path = '902-complete-mask.png'

    mask = cv2.imread(mask_path)
    mask = np.array(mask)

    mask_one_hot = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

    red, green, blue = mask[:,:,0], mask[:,:,1], mask[:,:,2]
    background = (red == 0) & (green == 0) & (blue == 0)
    cell = (red == 255) & (green == 0) & (blue == 255)
    neurite = (red == 0) & (green == 145) & (blue == 247)
    mask_one_hot[:,:][background] = [0]
    mask_one_hot[:,:][cell] = [1]
    mask_one_hot[:,:][neurite] = [2]
    mask_one_hot = torch.from_numpy(mask_one_hot)

    mask = torch.from_numpy(mask).long()

    print("Resampling image....")
    resampled_image = image_resample_row(img_path, size=(4, 5), sample_size=512)

    print("Performinng segmentation task....")
    masks = []
    for image in resampled_image:
        masks.append(unet_predict(image, model="small"))
   
    print("Running mask_stitching....")
    complete_mask = mask_stitching(masks, overlap_size=128, shape=(4, 5))
    one_hot_complete_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

    red, green, blue = complete_mask[:,:,0], complete_mask[:,:,1], complete_mask[:,:,2]
    background = (red == 0) & (green == 0) & (blue == 0)
    cell = (red == 255) & (green == 0) & (blue == 255)
    neurite = (red == 255) & (green == 129) & (blue == 31)
    one_hot_complete_mask[:,:][background] = [0]
    one_hot_complete_mask[:,:][cell] = [1]
    one_hot_complete_mask[:,:][neurite] = [2]

    result = torch.from_numpy(complete_mask).long()
    one_hot_result = torch.from_numpy(one_hot_complete_mask).long()

    print("Calculate IoU.......")
    jaccard = JaccardIndex(task="multiclass", num_classes=3)
    iou = jaccard(one_hot_result, mask_one_hot)
    print("iou: ", iou)

    print("Calculating accuracy..... ")
    fg_mask = (mask_one_hot==1)
    ne_mask = (mask_one_hot==2)
    ne_result_mask = (one_hot_result==2)
    val_acc = torch.sum(one_hot_result==mask_one_hot).item()/(torch.numel(mask_one_hot))
    val_fg_acc = torch.sum(one_hot_result[fg_mask]==mask_one_hot[fg_mask]).item()/max(torch.sum(fg_mask).item(), 1e-7)
    val_ne_acc = torch.sum(one_hot_result[ne_mask]==mask_one_hot[ne_mask]).item()/max(torch.sum(ne_mask).item(), 1e-7)

    print("val_acc: ", val_acc, " val_fg_acc: ", val_fg_acc, " val_ne_acc: ", val_ne_acc)

    plt.imshow(complete_mask)
    plt.show()
