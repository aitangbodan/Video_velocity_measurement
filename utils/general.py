#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/3月/26 18:12  
# File: general.py  
# Project: hzjt_algorithm_backend    
# Software: PyCharm   
"""
Function:
    
"""

import json
import yaml


def bbox_iou(box1, box2):
    """
    计算两个矩形框的交并比:
    Args:
        box1:x1,y1,x2,y2
        box2:x1,y1,x2,y2

    Returns:

    """
    # Intersection area 交集
    i_w = (min(box1[2], box2[2]) - max(box1[0], box2[0]))
    i_h = (min(box1[3], box2[3]) - max(box1[1], box2[1]))

    inter = i_w * i_h if i_w > 0 and i_h > 0 else 0

    # Union Area 并集
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1] + 1e-6
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1] + 1e-6
    union = w1 * h1 + w2 * h2 - inter + 1e-6
    iou = inter / union
    return iou


def load_json_file(json_file_path):
    try:
        with open(json_file_path, mode='r', encoding='utf-8') as f:
            return json.load(f)

    except Exception as e:
        raise Exception(f'{e}:error occured when loading json file \'{json_file_path}\'')


def load_yaml_file(json_file_path):
    try:
        with open(json_file_path, mode='r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)
    except Exception as e:
        raise Exception(f'{e}:error occured when loading file \'{json_file_path}\'')
