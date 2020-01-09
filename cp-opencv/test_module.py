#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
import copy
import os
from ctypes import *


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

import time

def random_colour_masks(image):

    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img_path, threshold):
    img = img_path
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    model.cuda()
    img = img.cuda()
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_score_c = copy.deepcopy(pred_score)
    pred_t = [pred_score_c.index(x) for x in pred_score_c if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class

def run(img):
    threshold = 0.92
    rect_th = 3
    text_size = 1
    text_th = 3

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks, boxes, pred_cls = get_prediction(img, threshold)
    point2 = np.zeros(shape=len(masks)*4)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(122, 33, 144), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (6, 66, 6),
                    thickness=text_th)
        point2[4*i-4] = int(boxes[i][0][0])
        point2[4*i-3] = int(boxes[i][0][1])
        point2[4*i-2] = int(boxes[i][1][0])
        point2[4*i-1] = int(boxes[i][1][1])
    point2 = ' '.join(str(i) for i in point2)
    point2 =point2 + ' '
    #print(point2)
    cv2.imshow("img", img)
    cv2.waitKey(1)
    return point2
