#!/usr/bin/env python3

import numpy as np

from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
import cv2
import os

DATASET_DIR="../../dataset"
os.makedirs(f"{DATASET_DIR}/annotation", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/frames", exist_ok=True)

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1

def clean_segmented_image(seg_img, real_img=None):
    # TODO
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    # print(seg_img.shape)
    # print(seg_img)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
    """
    classes:
    bg: [255, 0 , 255]
    duckie: [100, 117, 226]: 1
    cone: [226, 111, 101]: 2
    truck: [116, 114, 117]: 3
    bus: [216, 171, 15]: 4
    """
    mask = seg_img.copy()
    new_mask = np.zeros(shape=mask.shape[:-1],dtype=np.uint8)
    # mask[mask == [255, 0, 255]] = 0
    # remove pixels in RGB [from, to]
    remove = [[[175, 6, 14], [248, 139, 147]]] # stop lines (red)
    remove.append([[126, 126, 125],[255, 255, 255]]) # white lines
    remove.append([[202, 186, 45],[255, 254, 185 ]]) # yellow lines
    remove.append([[255, 0, 255],[255, 0, 255]]) # bg
    for i, (low, high) in enumerate(remove):
        lowerBound = np.array(low)[::-1] # to bgr format
        upperBound = np.array(high)[::-1] # to bgr format
        _mask = cv2.inRange(mask, lowerBound, upperBound)
        _mask = cv2.bitwise_not(_mask)
        
        # print(_mask.shape,mask.shape)
        # print(_mask.dtype,mask.dtype)
        if i==0: # keeping cone pixels since it falls in range of red lines
            pixel = np.array([226, 111, 101])[::-1] # to bgr format
            __mask = cv2.inRange(mask, pixel, pixel)
            _mask = cv2.bitwise_or(_mask, __mask)
        mask = cv2.bitwise_and(mask, mask, mask=_mask)
    
    
    # remove.append([[],[]])
    kernel = np.ones((4, 3),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    # cv2.imshow("image", mask)
    # cv2.waitKey(0)
    
    boxes = []
    classes = []
    for i, color in enumerate([[100, 117, 226], [226, 111, 101], [116, 114, 117], [216, 171, 15]]):
        color = np.array(color)[::-1] # to bgr format
        binary_mask = (mask == color).all(axis=2)
        new_mask[binary_mask] = i+1
        ## bbox finder:
        thresh = binary_mask.astype(np.uint8)*255
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(real_img, contours, -1, (0,255,0), 2)
        for object in contours:
            # print(object.shape)
            object = np.squeeze(object)
            xmin, ymin = np.min(object, axis=0)
            xmax, ymax = np.max(object, axis=0)

            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(i)
        
    if real_img is not None:
        pass
        # masked = 
        # print(np.sum(mask == 1))
        # display_img_seg_mask(real_img, seg_img, new_mask)
    return np.array(boxes), np.array(classes)

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        if nb_of_steps % 7 != 0:
            nb_of_steps += 1
            continue

        boxes, classes = clean_segmented_image(segmented_obs, obs)
        if boxes.size == 0:
            nb_of_steps += 1
            continue

        # yolov5s training data
        img_h, img_w, _ = obs.shape
        
        x_center, y_center = (boxes[:, 0] + boxes[:, 2])/(2 * img_w), (boxes[:, 1] + boxes[:, 3])/(2 * img_h)
        width, height = (boxes[:, 2] - boxes[:, 0])/img_w, (boxes[:, 3] - boxes[:, 1])/img_h
        f = open(f"{DATASET_DIR}/annotation/{npz_index}.txt", 'w')
        for cat_id,x_cen,y_cen,w,h in zip(classes,x_center, y_center,width,height):
            if w < 0.006 or h < 0.006:
                continue
            print(f'{cat_id} {x_cen:.6f} {y_cen:.6f} {w:.6f} {h:.6f}', file=f)
        f.close()
        cv2.imwrite(f"{DATASET_DIR}/frames/{npz_index}.png", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

        # assignment data
        save_npz(obs, boxes, classes)
        if npz_index % 100 == 0:
            print(npz_index, "saved.")
        nb_of_steps += 1
        if done or nb_of_steps > MAX_STEPS:
            break

    if npz_index == 2500:
        break
