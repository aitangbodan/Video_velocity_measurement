#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 王根一 
# Gitlab: ECCOM  
# Creat Time:  2022/11月/23 14:43  
# File: manager_test.py  
# Project: TRT_inference    
# Software: PyCharm   
"""
Function:
"""
import random
import time

import numpy as np
import cv2
import sys
import torch

import torch.cuda

sys.path.append('../')
from application import get_manager, get_manager_v1, get_manager_v2
from model.inference.inference_engine_2 import EngineInstance
import pycuda.autoinit as cudainit
from utils.log import logger
import os

#service_config = r'../application/config/water_flow_velocity_config.json'
service_config = r'../application/config/cal_config.json'
batch_size = 1
frame_ids = [0, 0, 1, 1, 1, 1, 1, 1, 1]
logger.info(f'frame_ids:{frame_ids}')
# frame_ids = [0, 1, 2, 5, 1, 2]  # , 1, 2, 5, 4, 3, 0 # #0, 1, 1, 0
cal_camera = {0: [0.15911753302948042, 0.1203215154],
              1: [0.20608207136392595],
             }
model = get_manager(service_config)
#model = get_manager_v1(service_config)
save_result = False
if save_result:
    fourcc = cv2.VideoWriter_fourcc('F', 'L', 'V', '1')  # 'X', 'V', 'I', 'D'
    writer = cv2.VideoWriter('./test_images/bytetracking.flv',
                             fourcc=fourcc,
                             fps=25, frameSize=(img_w, img_h))


def compose_imgs():
    imgs = []
    for i in range(batch_size):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_w, img_h))
        imgs.append(img)

    imgs = np.stack(imgs)
    return imgs


def compose_imgs_2(imgs):
    #for i in range(len(imgs)):
        #imgs[i] = cv2.resize(imgs[i], (img_w, img_h))
    imgs = np.stack(imgs)
    return imgs


def draw_bboxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = (0, 255, 0)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def test_on_image(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predicted_speed = []
    input_data = []
    batch_size = 1
    image_list = os.listdir(path)
    image_list = sorted(image_list)
    #print(image_list)
    for img in image_list:
        img = cv2.imread(path + img)
        size = img.shape
        #img = cv2.resize(img, (356, 252))
        input_data.append(img)
    imgs = compose_imgs_2(input_data)
    imgs = imgs.flatten()
    speed_list = model.detect(imgs, frame_ids)
    # speed_list = model.detect(imgs, frame_ids)
    return speed_list
    
def calibration_30frame(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predicted_speed = []
    input_data = []
    batch_size = 1
    image_list = os.listdir(path)
    image_list = sorted(image_list)
    #print(image_list)
    for img in image_list:
        img = cv2.imread(path + img)
        size = img.shape
        #img = cv2.resize(img, (356, 252))
        input_data.append(img)
    imgs = compose_imgs_2(input_data)
    imgs = imgs.flatten()
    first = time.time()
    speed_list = model.calibration(imgs, frame_ids)
    print(time.time() - first)
    # speed_list = model.detect(imgs, frame_ids, cal_camera)
    return speed_list

def test_on_video():
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    imgs = []
    frame_id = 0
    total_cost = 0
    while ret:
        frame_id += 1
        if frame_id > 1000:
            break
        for i in range(batch_size):
            imgs.append(frame)

        datas = compose_imgs_2(imgs)
        t0 = time.time()
        preds = get_manager(service_config).detect([datas.tobytes()], frame_ids)
        # logger.info(len(preds))
        total_cost += time.time() - t0
        logger.info(f'cost :{time.time() - t0}')
        logger.info(f'frame id:{frame_id}')
        # logger.info(f'preds:{preds}')
        logger.info('--------------------------')

        if save_result:
            bboxes = []
            ids = []
            if preds:
                for box_info in preds[0][0]['IllegalStay']:
                    # logger.info(f'box_info:{box_info}')
                    bboxes.append(box_info[:4])
                    ids.append(box_info[4])
            draw_bboxes(imgs[0], bboxes, ids)
            # cv2.imwrite('IllegalStay_show.jpg', imgs[0])
            writer.write(imgs[0])
        imgs.clear()
        ret, frame = cap.read()
    cap.release()
    if save_result:
        writer.release()
    logger.info(f'avg fps:{frame_id / total_cost}')


if __name__ == '__main__':
    # test_on_video()
    path = '/dataset/wanggenyi/test/'
    #test_on_image(path)
    calibration_30frame(path)