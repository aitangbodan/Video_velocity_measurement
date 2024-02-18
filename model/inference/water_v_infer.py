#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/9月/28 14:01  
# File: seg_infer.py  
# Project: TRT_inference    
# Software: PyCharm   
"""
Function:场景分割模型推理
    
"""
import sys
import cv2
import time
import pandas as pd
import numpy as np
import os
import pycuda.autoinit as cudainit

sys.path.append('./')

from .inference_engine_2 import EngineInstance

import torch

def change_brightness(image, bright_factor):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    
    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb

def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    hsv = np.zeros(image_current.shape)
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]
 
    # Flow Parameters
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        0)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  
        
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype= np.float32)    

    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    
    return rgb_flow

def preprocess_image_from_path(img, scale_factor=0.5, bright_factor=1):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = change_brightness(img, bright_factor)
    # img = crop_image(img, scale_factor)
    return img

def pre_func(imgs_ts, half, device, ):
    """
    图像预处理回调函数
    :param imgs_ts: 必需参数, torch.cuda.Tensor
    :param half:必需参数, 是否使用fp16精度
    :return:
    """
    batch_image_ts = imgs_ts[..., (2, 1, 0)].permute(0, 3, 1, 2)
    batch_image_ts = batch_image_ts.contiguous()
    mean = torch.tensor((0.485, 0.456, 0.406)).resize(1, 3, 1, 1).to(device)
    std = torch.tensor((0.229, 0.224, 0.225)).resize(1, 3, 1, 1).to(device)
    batch_image_ts = batch_image_ts / 255.0
    batch_image_ts = (batch_image_ts - mean) / std
    batch_image_ts = batch_image_ts.half() if half else batch_image_ts.float()
    return batch_image_ts


class water_flow_velocity_detection(EngineInstance):

    def __init__(self, engine_file,
                 binding_names,
                 names=[],
                 device=0,
                 half=True):
        super(water_flow_velocity_detection, self).__init__(engine_file, binding_names,
                                                 device=device, half=half)
        self.names = names

    def inference_v2(self, input_imgs, imgw, imgh):
        # 图像预处理，包括数据转换，图像预处理
        input_datas = [self.pre_process(input_imgs[0], imgw, imgh,
                                        batch_image_ts=self.input_tensors[0],
                                        func=None,
                                        args=[self.device])]
        input_datas[0] = input_datas[0].permute(0, 3, 1, 2)
        #print(self.input_tensors[0].shape, input_datas[0].shape)
        outputs = self.run(input_datas)

        return outputs

    def post_process(self, logit_ts):
        prob = torch.softmax(logit_ts, dim=1)
        pred = torch.argmax(prob, dim=1)
        # print(torch.sum(pred == 1))
        return pred.cpu().numpy()


if __name__ == '__main__':
    import cv2
    import numpy as np
    import os
    import pycuda.autoinit as cudainit
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predicted_speed = []
    input_data = []
    batch_size = 1
    path = '/dataset/wanggenyi/test/'
    image_list = os.listdir(path)
    image_list = sorted(image_list)
    print(image_list)
    for img in image_list:
        img = cv2.imread(path + img)
        size = img.shape
        #img = cv2.resize(img, (356, 252))
        input_data.append(img)
    # image_list = sorted(image_list)
    print(size)
    engine = water_flow_velocity_detection(
        engine_file=r'../engine_file/model.432FP32.engine',
        binding_names=['inputs', 'outputs'],
        names=['bg', 'water'],
        half=False,
        device=0
    )
    for i in range(0, len(input_data)-1):
        ii = i // 2  
        x1 = preprocess_image_from_path(input_data[i])
        x2 = preprocess_image_from_path(input_data[i+1])

        rgb_diff = opticalFlowDense(x1, x2)
        rgb_diff = rgb_diff.reshape(1, rgb_diff.shape[0], rgb_diff.shape[1], rgb_diff.shape[2])
        rgb_diff = np.moveaxis(rgb_diff, 3, 1)
        print(rgb_diff.shape)
        #rgb_diff = torch.from_numpy(rgb_diff).to(device)

        rgb_diff = rgb_diff.tobytes()
        prediction = engine.inference_v2([rgb_diff], imgw=size[1], imgh=size[0])
        #prediction = model(rgb_diff)
        #prediction = prediction.cpu().detach().numpy()
        predicted_speed.append(prediction[0][0].cpu().detach().numpy() * 3.6)
    print(predicted_speed, np.mean(sorted(np.abs(predicted_speed))))
    # save results
