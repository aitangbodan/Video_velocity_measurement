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
import time

sys.path.append('./')

from model.inference.inference_engine_2 import EngineInstance

import torch


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


class SegmentationEngine(EngineInstance):

    def __init__(self, engine_file,
                 binding_names,
                 names=[],
                 device=0,
                 half=True):
        super(SegmentationEngine, self).__init__(engine_file, binding_names,
                                                 device=device, half=half)
        self.names = names

    def inference_v2(self, input_imgs, imgw, imgh):
        # 图像预处理，包括数据转换，图像预处理
        input_datas = [self.pre_process(input_imgs[0], imgw, imgh,
                                        batch_image_ts=self.input_tensors[0],
                                        func=pre_func,
                                        args=[self.device])]
        outputs = self.run(input_datas)

        return self.post_process(outputs[0])

    def post_process(self, logit_ts):
        prob = torch.softmax(logit_ts, dim=1)
        pred = torch.argmax(prob, dim=1)
        # print(torch.sum(pred == 1))
        return pred.cpu().numpy()


if __name__ == '__main__':
    import pycuda.autoinit as cudainit
    import cv2
    import numpy as np
    import os

    img_path = './test_images/shuiku1.jpg'
    batch_size = 10
    size = (480, 640)
    input_datas = []
    for i in range(batch_size):
        # input_img = pre_process(img_path, (size[-1], size[0]), dtype=dtype)
        input_img = cv2.imread(img_path)
        input_img = cv2.resize(input_img, (size[-1], size[0]))
        input_datas.append(input_img)
    input_datas = np.stack(input_datas)
    # input_datas = np.expand_dims(input_datas[0], axis=0)
    input_datas = input_datas.tobytes()

    engine = SegmentationEngine(
        engine_file=r'./model/engine_file/STDCNet1446_model_maxmIOU75.480.FP16.engine',
        binding_names=['inputs', 'outputs'],
        names=['bg', 'water'],
        half=True,
        device=0
    )

    for i in range(20):
        t0 = time.time()
        masks = engine.inference_v2([input_datas], imgw=size[-1], imgh=size[0])
        print(f'cost time:{time.time() - t0}')
        print(f'masks.shape:{masks.shape}')
    # save results
    masks[masks > 0] = 255
    mask = np.repeat(np.expand_dims(masks[0], axis=2), 3, axis=2)
    mask_red = np.zeros((size[0], size[1], 3))
    mask_red[:, :, 2] = masks[0]
    # concat source image and result mask
    # results = np.concatenate((input_img, np.repeat(np.expand_dims(masks[0], axis=2), 3, axis=2)), axis=1)
    r = 0.5  # mixup
    input_img[mask > 0] = r * input_img[mask > 0] + (1 - r) * mask_red[mask > 0]
    result_path = os.path.join(os.path.splitext(img_path)[0] + '_mask.jpg')
    cv2.imwrite(result_path, input_img)
