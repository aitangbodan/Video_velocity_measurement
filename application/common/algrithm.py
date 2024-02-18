#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/7月/11 17:00  
# File: algrithm.py  
# Project: TRT_inference    
# Software: PyCharm   
"""
Function:
    
"""
import cv2
import numpy as np

ALG_LIST = {
    1: 'IllegalStay',
    2: 'TruckForbidden',
    3: 'tracking',
    4: 'WaterLevel'
}
from utils.log import logger


def IllegalStay(objs, mask, roi_ids):
    event_res = []
    xyxy, cls = objs
    cent_x, cent_y = (xyxy[:, 0] + xyxy[:, 2]) // 2, (xyxy[:, 1] + xyxy[:, 3]) // 2
    evt_roi_ids = mask[cent_y, cent_x]
    for i, evt_roi_id in enumerate(evt_roi_ids):
        if evt_roi_id in roi_ids:
            event_res.append(xyxy[i].tolist() + [cls[i]])  # + [conf[i]]

    return event_res


def TruckForbidden(objs, mask, roi_ids):
    event_res = []
    # xyxys, clses, confs = xyxy_cls_conf
    xyxys, clses = objs
    for (xyxy, cls) in zip(xyxys, clses):  # confs
        if cls.find('truck') != -1:
            cent_x, cent_y = (xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2
            roi_id = mask[cent_y, cent_x]
            if roi_id in roi_ids:
                event_res.append(xyxy.tolist() + [cls])  # + [conf]
    return event_res


def tracking(objs, mask, roi_ids):
    # 目标位置与id,目标描述
    tracking_xyxys_ids, clses = objs
    event_res = []
    for tracking_xyxy_id, tracking_cat in zip(tracking_xyxys_ids, clses):
        event_res.append(tracking_xyxy_id[:5].tolist() + [tracking_cat])
    return event_res


def WaterLevel(seg_mask, mask, roi_ids):
    """
    
    :param seg_mask:分割结果掩码
    :param mask:水位线配置掩码
    :param roi_ids:水位线
    :return:
    """
    event_res = []
    seg_mask = seg_mask.astype(np.uint8)
    edge = cv2.Canny(seg_mask, 0, 1, apertureSize=5)
    logger.info(np.sum(edge > 0))
    # cv2.imwrite('test_images/seg_mask.jpg', edge)
    '''insert code here'''
    # 返回当前水位状态
    return [roi_ids[0]]
