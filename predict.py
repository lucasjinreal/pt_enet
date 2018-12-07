import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

import transforms as ext_transforms
from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils
from alfred.dl.torch.common import device
from data.cityscapes import Cityscapes
import cv2
from PIL import Image
import numpy as np
from alfred.vis.image.seg import draw_seg_by_dataset
from alfred.vis.image.get_dataset_colormap import label_to_color_image

target_size = (512, 1024)
data_dir = '/media/jintain/sg/permanent/datasets/Cityscapes'


def predict():
    image_transform = transforms.Compose(
        [transforms.Resize(target_size),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize(target_size),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = Cityscapes(
        data_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)

    class_encoding = train_set.color_encoding

    num_classes = len(class_encoding)
    model = ENet(num_classes).to(device)

    # Initialize a optimizer just so we can retrieve the model from the
    # checkpoint
    optimizer = optim.Adam(model.parameters())

    # Load the previoulsy saved model state to the ENet model
    model = utils.load_checkpoint(model, optimizer, 'save', 'ENet_cityscapes_mine.pth')[0]
    # print(model)

    image = Image.open('images/mainz_000000_008001_leftImg8bit.png')
    images = Variable(image_transform(image).to(device).unsqueeze(0))
    image = np.array(image)

    # Make predictions!
    predictions = model(images)
    _, predictions = torch.max(predictions.data, 1)
    # 0~18
    prediction = predictions.cpu().numpy()[0] - 1

    mask_color = np.asarray(label_to_color_image(prediction, 'cityscapes'), dtype=np.uint8)
    mask_color = cv2.resize(mask_color, (image.shape[1], image.shape[0]))
    print(image.shape)
    print(mask_color.shape)
    res = cv2.addWeighted(image, 0.3, mask_color, 0.7, 0.6)
    # cv2.imshow('rr', mask_color)
    cv2.imshow('combined', res)
    cv2.waitKey(0)




if __name__ == '__main__':
    predict()