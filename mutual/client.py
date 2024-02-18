#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/6月/24 14:06  
# File: client.py
# Project: TRT_inference    
# Software: PyCharm   
"""
Function:
    
"""
import sys

sys.path.append('./')
import cv2

import grpc
import dto_pb2
import dto_pb2_grpc
import numpy as np
from utils.log import logger

img_path = r'./test_images/test_3.jpg'
video_path = r'/dataset/lijin/car_video.mp4'

batch_size = 1
import time
import json

img_w = 640
img_h = 640
frame_ids = [0]  # #, 0, 1, 0, 1, 0, 0, 1


def run_video():
    logger.info(f'start to send data....')
    # grpc建立连接
    channel = grpc.insecure_channel('localhost:12347')
    stub = dto_pb2_grpc.CVServiceStub(channel)
    cap = cv2.VideoCapture(video_path)

    # imgs = np.ascontiguousarray(imgs)

    # logger.info(f'img shape {imgs.shape}')

    # 构建初始化参数
    cfg_file = './application/config/highway_config_1.json'
    with open(cfg_file, mode='r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg_json = json.dumps(cfg)
    # 初始化请求
    req = dto_pb2.imgRequest(stream_id='test',
                             json_input=cfg_json)
    res = stub.configuration(req)
    logger.info(f'config response res:{res.json_out}')

    # 创建请求对象，请求中的参数为我们在proto文件中定义的类型

    # req.set_imgdata(imgs.data, imgs.size)
    # 发送请求并接收返回 res的数据类型为imganalyse所定义的返回类型

    total_time = 0
    run_time = 0
    ret, frame = cap.read()
    imgs = []

    def compose_imgs(images):
        for i in range(batch_size):
            images[i] = cv2.resize(images[i], (img_w, img_h))
        images = np.stack(images)
        return images

    frame_id = 0
    while ret:
        frame_id += 1
        if frame_id > 450:
            break
        t0 = time.time()
        imgs.append(frame)
        if len(imgs) == batch_size:
            req = dto_pb2.imgRequest(stream_id='test',
                                     scale_imgW=img_w,
                                     scale_imgH=img_h,
                                     imgdata=compose_imgs(imgs).tobytes(),
                                     json_input=json.dumps({'frame_inds': frame_ids}))
            res = stub.imganalyse(req)
            t1 = time.time()
            logger.info(f'calling time:{t1 - t0}')
            logger.info(f'algorithm run time:{res.send_time}')
            logger.info(f'frame id :{frame_id}')
            # out_list = json.loads(res.json_out)['rec']
            total_time += res.send_time
            logger.info('-----------------------------------')
            # logger.info(f'json_out:{len(out_list)},{len(out_list[0])}')
            # logger.info(f'json_out:{json.dumps(out_list)}')

            imgs.clear()
        ret, frame = cap.read()
    logger.info(f'total fps:{frame_id / total_time}')


def run():
    logger.info(f'start to send data....')
    # grpc建立连接
    channel = grpc.insecure_channel('localhost:12345')
    stub = dto_pb2_grpc.CVServiceStub(channel)

    imgs = []
    for i in range(batch_size):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_w, img_h))
        imgs.append(img)

    imgs = np.stack(imgs)
    # imgs = np.ascontiguousarray(imgs)

    logger.info(f'img shape {imgs.shape}')

    # 构建初始化参数
    cfg_file = './application/config/highway_config_1.json'
    with open(cfg_file, mode='r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg_json = json.dumps(cfg)
    # 初始化请求
    req = dto_pb2.imgRequest(stream_id='test',
                             json_input=cfg_json)
    res = stub.configuration(req)
    logger.info(f'config response res:{res.json_out}')
    # 创建请求对象，请求中的参数为我们在proto文件中定义的类型
    req = dto_pb2.imgRequest(stream_id='test',
                             scale_imgW=img_w,
                             scale_imgH=img_h,
                             imgdata=imgs.tobytes(),
                             json_input=json.dumps({'frame_inds': frame_ids}))
    # req.set_imgdata(imgs.data, imgs.size)
    # 发送请求并接收返回 res的数据类型为imganalyse所定义的返回类型
    times = 10010
    warm_up = 10
    total_time = 0
    run_time = 0
    for i in range(times):
        t0 = time.time()
        res = stub.imganalyse(req)
        t1 = time.time()
        logger.info(f'calling time:{t1 - t0}')
        logger.info(f'algorithm run time:{res.send_time}')
        out_list = json.loads(res.json_out)['rec']
        # logger.info(f'json_out:{len(out_list)},{len(out_list[0])}')
        logger.info(f'json_out:{json.dumps(out_list)}')
        if i < warm_up:
            continue
        total_time += (t1 - t0)
        run_time += res.send_time
    logger.info(f'avg 1/fps:{total_time * 1000 / (times - warm_up)}')
    logger.info(f'avg algorithm run time:{run_time * 1000 / (times - warm_up)}')
    logger.info(f'avg grpc transfer:{(total_time - run_time) * 1000 / (times - warm_up)}')


if __name__ == '__main__':
    run_video()
    # run()
