import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from scipy import ndimage
from pytorch_segmentation import models
from pytorch_segmentation.utils.palette import CityScpates_palette
from pytorch_segmentation.utils.helpers import colorize_mask
from math import ceil

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

def sliding_predict(model, image, num_classes, flip=True):
    image_size = image.shape
    tile_size = (int(image_size[2]//2.5), int(image_size[3]//2.5))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = padded_img.flip(-1)
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions
    return total_predictions

import time

def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img))

        # if flip:
        #     fliped_img = scaled_img.flip(-1).to(device)
        #     fliped_predictions = upsample(model(fliped_img).cpu())
        #     scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions

def inference_init(model_path):
    """
    Initializes the following global objects:
        - model: object to be later used for inference in images
        - num_classes: integer number of classes
        - to_tensor and normalize: functions
        - device: torch device. CPU or CUDA GPU if available
        - palette: color palette for CityScapes dataset
    """

    filepaths_prefix = os.getcwd() + '/catkin_ws/src/tfg/'
    model_path = filepaths_prefix + model_path

    global to_tensor, normalize, model, device, num_classes, palette
    # Check if the pretrained model is available
    if not model_path.endswith('.pth'):
        raise RuntimeError('Unknown file passed. Must end with .pth')

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.28689529, 0.32513294, 0.28389176], [0.17613647, 0.18099176, 0.17772235]) # MEAN, STD
    num_classes = 19 #loader.dataset.num_classes
    palette = CityScpates_palette#loader.dataset.palette

    # Model
    config_arch_args = {'backbone': 'resnet50', 'freeze_bn': False, 'freeze_backbone': False}
    model = getattr(models, 'PSPNet')(num_classes, backbone='resnet50', freeze_bn=False, freeze_backbone=False)
    availble_gpus = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    print('model path:', model_path)
    checkpoint = torch.load(model_path,  map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    
    num_classes = 19

    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

def inference_segment_image(image, mode):
    with torch.no_grad():
        input_image = normalize(to_tensor(image)).unsqueeze(0)
        
        if mode == 'multiscale':
            #scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] 
            scales = [0.75, 1.0] # Maybe this way it doesn't eat up all my memory
            prediction = multi_scale_predict(model, input_image, scales, num_classes, device)
        elif mode == 'sliding':
            prediction = sliding_predict(model, input_image, num_classes)
        else:
            prediction = model(input_image.to(device))
            prediction = prediction.squeeze(0).cpu().numpy()
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
        output_image = colorize_mask(prediction, palette)
    
    return output_image
