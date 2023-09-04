'''
This file create pipeline that generate new images and new json files with coords of boxes

MULTIPLIER - defines how many times create new image from one original image
'''

import json
import numpy as np
import albumentations as A
import cv2
import os

# Pipeline to create new images

transform = A.Compose([
    A.RandomCrop(width=600, height=600),    # new images have size (600 x 600)
    A.HorizontalFlip(p=0.5),                # with propability 0.5 image is flipped horizontally
    A.RandomBrightnessContrast(p=0.2),      # with propability 0.2 brightness of image is changed
    A.RandomGamma(p=0.3),                   # with propability 0.3 gamma of image (color) is changed
    A.RGBShift(p=0.2),                      # with propability 0.2 RGB image is switched
    A.VerticalFlip(p=0.4)],                 # with propability 0.4 image is flipped vertically
    bbox_params=A.BboxParams(format='albumentations',
                             label_fields=['class_labels']) # bounding box paramiters, albumentations means that coords are normalized
)

MULTIPLIER = 30

# loop cheacking every image from every folder
for part in ['train', 'test', 'validate']:
    for item in os.listdir(os.path.join('data', part, 'images')):  
        img = cv2.imread(os.path.join('data', part, 'images', item))
        label_path = os.path.join('data', part, 'labels', f'{item.split(".")[0]}.json')
        # checking if json file with coords exist
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
            cords = []
            for i in label['shapes'][0]['points']:
                for j in i:
                    cords.append(j)
        # if there is not json file for image take coords 0, 0, 0, 0
        else:
            cords = [0, 0, 0.00001, 0.00001]
        # normalize coords
        cords = list(np.divide(cords, [1280, 720, 1280, 720]))
        try:
            # loop creating new images
            for i in range(MULTIPLIER):
                aug_img = transform(image=img, bboxes=[cords], class_labels=['face'])
                cv2.imwrite(os.path.join('data', 'aug_data', part, 'images', f'{item.split(".")[0]}_{i}.jpg'), aug_img['image'])

                # creating json files with coords
                annotation = {}
                annotation['image'] = item
    
                if os.path.exists(label_path):
                    if len(aug_img['bboxes']) == 0:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = aug_img['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0
                
                with open(os.path.join('data', 'aug_data', part, 'labels', f'{item.split(".")[0]}_{i}.json'), 'w') as f:
                    json.dump(annotation, f)
        except Exception as e:
            print(e)