#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/1月/25 10:46  
# File: models.py  
# Project: hzjt_algorithm_backend    
# Software: PyCharm   
"""
Function:
    
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .inference.inference_engine_2 import DetectEngine, FeatureExtractEngine
from .inference.seg_infer import SegmentationEngine
from .deep_sort.deep_sort import DeepSort
from .bytetrack.byte_tracker import BYTETracker
from utils.cuda_gpu_man import *
from utils.general import load_yaml_file
from pycuda.tools import clear_context_caches

from utils import logger
import os

mc = None
pool = None
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

DEEPSORT_TRACKING = 'deepsort_tracking'
BYTETRACKING = 'bytetracking'

MODE_DETECT = 'detect'
MODE_TRACKING = 'tracking'
MODE_SEGMENTATION = 'segment'


class ModelController(object):

    def __init__(self, config_file):

        self.mc_config = load_yaml_file(config_file)
        # logger.info(self.mc_config)
        self.device_id = self.mc_config['device']
        self.mode = self.mc_config['mode']
        if self.mode == 'tracking':
            self.tracker = self.mc_config['tracker']
        if self.device_id < 0:
            self.device, self.ctx = self.__set_device()
        else:
            self.device, self.ctx = self.__set_device(self.device_id)
        if self.mode == MODE_DETECT or self.mode == MODE_TRACKING:
            self.det_engine = DetectEngine(self.mc_config[MODE_DETECT]['model_path'],
                                           [self.mc_config[MODE_DETECT]['input_name'],
                                            self.mc_config[MODE_DETECT]['output_name']],
                                           names=self.mc_config[MODE_DETECT]['cates'],
                                           half=self.mc_config[MODE_DETECT]['half'],
                                           device=self.device_id)
            logger.info(f'initialized object detection engine')
        if self.mode == MODE_TRACKING and self.tracker == DEEPSORT_TRACKING:
            self.feature_engine = FeatureExtractEngine(self.mc_config['reid']['model_path'],
                                                       [self.mc_config['reid']['input_name'],
                                                        self.mc_config['reid']['output_name']],
                                                       half=self.mc_config['reid']['half'],
                                                       device=self.device_id)
            logger.info(f'initialized feature extraction engine')
        if self.mode == MODE_SEGMENTATION:
            self.segment_engine = SegmentationEngine(self.mc_config[MODE_SEGMENTATION]['model_path'],
                                                     [self.mc_config[MODE_SEGMENTATION]['input_name'],
                                                      self.mc_config[MODE_SEGMENTATION]['output_name']],
                                                     names=self.mc_config[MODE_SEGMENTATION]['cates'],
                                                     half=self.mc_config[MODE_SEGMENTATION]['half'],
                                                     device=self.device_id)
            logger.info(f'initialized object segementation engine')

    def get_tracker(self):

        if self.tracker == DEEPSORT_TRACKING:

            # 初始化deepsort 跟踪器
            tracker = DeepSort(max_dist=self.mc_config[DEEPSORT_TRACKING]['MAX_DIST'],
                               min_confidence=self.mc_config[DEEPSORT_TRACKING]['MIN_CONFIDENCE'],
                               max_iou_distance=self.mc_config[DEEPSORT_TRACKING]['MAX_IOU_DISTANCE'],
                               max_age=self.mc_config[DEEPSORT_TRACKING]['MAX_AGE'],
                               n_init=self.mc_config[DEEPSORT_TRACKING]['N_INIT'],
                               nn_budget=self.mc_config[DEEPSORT_TRACKING]['NN_BUDGET'])
        elif self.tracker == BYTETRACKING:
            # 初始化bytetrack跟踪器
            tracker = BYTETracker(self.mc_config[BYTETRACKING])

        return tracker

    def __set_device(self, device=None):
        """
        set device for detection model, keep default if  device is None
        Args:
            device: int or str
        Returns:

        """
        if device is None:
            device_id = get_max_memory_free_device()
        else:
            device_id = device
        if device_id is None:
            raise ValueError('no available device to use')
        ctx, dev = device_init(device_id)
        logger.info(f'device id :{device_id}')
        return dev, ctx

    def __release(self):
        if self.ctx is not None:
            self.ctx.pop()
            clear_context_caches()
        if self.device is not None:
            torch.cuda.empty_cache()

    def detect(self, imgdata, box_ts=False):
        """
        批量检测
        :param box_ts:是否输出检测框的在输入张量中的截取子图
        :param imgdata:批量数据,byte or bytearray
        :param img_h:
        :param img_w:
        :return:
        """
        self.ctx.push()
        if self.mode == MODE_DETECT or self.mode == MODE_TRACKING:
            res = self.det_engine.inference_v2(imgdata,
                                               self.mc_config[MODE_DETECT]['width'],
                                               self.mc_config[MODE_DETECT]['height'],
                                               ret_box_ts=box_ts)
        elif self.mode == MODE_SEGMENTATION:
            res = self.segment_engine.inference_v2(imgdata,
                                                   self.mc_config[MODE_SEGMENTATION]['width'],
                                                   self.mc_config[MODE_SEGMENTATION]['height'])
        self.ctx.pop()
        # torch.cuda.empty_cache()
        return res

    def feat_extract(self, imgdata, pre_proc=True):
        """
        批量特征提取
        :param imgdata:
        :param img_h:
        :param img_w:
        :return:
        """
        self.ctx.push()
        feat_ts = self.feature_engine.inference_v2(imgdata,
                                                   self.mc_config['reid']['width'],
                                                   self.mc_config['reid']['height'],
                                                   pre_proc=pre_proc)
        # torch.cuda.empty_cache()
        self.ctx.pop()
        return feat_ts

    def detect_with_feat_extract(self, imgdata, img_h, img_w):
        """
        批量检测并返回目标的特征向量
        :param imgdata:批量数据,byte or bytearray
        :param img_h:
        :param img_w:
        :return:
        """
        # 检测并返回输出在输入图像中的cropped pitches
        # batched检测结果,merged cropped boxes 张量列表,cropped boxes对应的batch索引
        xyxy_cls_conf, boxes_ts_l = self.detect(imgdata, img_h, img_w, box_ts=True)
        box_bs_ind = []
        boxes_merged_ts_l = []
        for i, boxes_tss in enumerate(boxes_ts_l):
            box_bs_ind.extend([i] * len(boxes_tss))
            boxes_merged_ts_l.extend(boxes_tss)
        box_bs_ind = torch.IntTensor(box_bs_ind)

        feat_ts = self.feat_extract(boxes_merged_ts_l, 64, 64, pre_proc=False)
        ret_feats = []
        for batch_i in range(len(xyxy_cls_conf)):
            batch_i_feats = feat_ts[box_bs_ind == batch_i]
            ret_feats.append(batch_i_feats)

        return xyxy_cls_conf, ret_feats

    def __del__(self):
        self.__release()


def get_mc(config_file=r'./model/config/config.yaml'):
    global mc
    if mc is None:
        mc = ModelController(config_file)
        logger.info(f'model controller init ...')
    return mc


def get_pool():
    global pool
    if pool is None:
        pool = ThreadPoolExecutor(3)
        logger.info(f'initialized thread pool...')
    return pool


def tracking_from_list(tracker, input_list):
    if not len(input_list):
        return None
    result = []
    if isinstance(tracker, DeepSort):
        for i, (bboxes, confs, feats, cats) in enumerate(input_list):
            tracking_obj, tracking_cat = tracker.update(bboxes, confs, feats, cats)
            result.append([tracking_obj, tracking_cat])
    elif isinstance(tracker, BYTETracker):
        for i, (bboxes, confs, cats) in enumerate(input_list):
            box_confs = np.concatenate((bboxes, np.array(confs).reshape(-1, 1)), axis=1)
            tracking_obj, tracking_cat = tracker.update(box_confs, cats)
            result.append([tracking_obj, tracking_cat])
    return result
