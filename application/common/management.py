#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 王根一
# Gitlab: ECCOM
# Creat Time:  2022/11月/23 13:58
# File: management.py
# Project: smart_service_area
# Software: PyCharm
"""
Function:

"""
import time

import numpy as np
import cv2
import cv2 as cv
import os
import math

import torch
from model.inference.water_v_infer import *
from ..highway_abnormal_detect.detect import Detection
from model import models
from model import get_mc, get_pool, tracking_from_list
# from utils.log import Logger
from .algrithm import ALG_LIST
from parallel import work_man
from icecream import ic
from collections import defaultdict
from efficientnet_pytorch import EfficientNet
import json
import pycuda.autoinit as cudainit
import logging
import re
from logging.handlers import TimedRotatingFileHandler

manager = None
manager1 = None

def Logger(log_name):
    # 创建logger对象。传入logger名字
    logger = logging.getLogger(log_name)
#     log_path = os.path.join("J:\loggering日志学习\log_save",log_name)
    # 设置日志记录等级
    logger.setLevel(logging.INFO)
    # interval 滚动周期，
    # when="MIDNIGHT", interval=1 表示每天0点为更新点，每天生成一个文件
    # backupCount  表示日志保存个数
    file_handler = TimedRotatingFileHandler(
        filename=log_name, when="MIDNIGHT", interval=1, backupCount=1
    )
    # filename="mylog" suffix设置，会生成文件名为mylog.2020-02-25.log
    file_handler.suffix = "%Y-%m-%d.log"
    # extMatch是编译好正则表达式，用于匹配日志文件名后缀
    # 需要注意的是suffix和extMatch一定要匹配的上，如果不匹配，过期日志不会被删除。
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
    # 定义日志输出格式
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
        )
    )
    logger.addHandler(file_handler)
    return logger
Logger = Logger("mylog")
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f_obj:
         jlist = json.load(f_obj)
    return jlist

def caldist(a, b, c, d):
    return abs(a - c) + abs(b - d)

def adist(a, b, c, d):
    #     ic(a, b, c ,d)
    return ((a - c) ** 2 + (b - d) ** 2) ** 0.5

def mean(prediction, predicted_speed, index, abnorm_meta=None):
#     predicted_speed[0] * index 
    if abnorm_meta:
        Logger.info(f'index: {index} abnorm_meta number: {len(abnorm_meta)}')
        if index < len(abnorm_meta):
            return predicted_speed
        else:
            speed_de_abnorm_meta = predicted_speed * index - sum(abnorm_meta) + (index - len(abnorm_meta)) * predicted_speed
            return (speed_de_abnorm_meta + prediction) / (index + 1)
    else:
        return (predicted_speed * index + prediction) / (index + 1)


class Manager(object):
    """
    基础检测器
    """

    def __init__(self, configs):
        super(Manager, self).__init__()
        # 各摄像头的事件检测器,各摄像头加载的算法类型列表
        
        self.engine = water_flow_velocity_detection(
        engine_file=r'./application/common/engine_file/model.432FP32.engine',
        binding_names=['inputs', 'outputs'],
        names=['bg', 'water'],
        half=False,
        device=0
    )
        self.cnt = defaultdict(int)
        self.sof = defaultdict(list)
        self.mean = defaultdict(list)
        self.feature_params = dict(maxCorners=10, qualityLevel=0.05, minDistance=3, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))
        self.orb = cv.ORB_create(20,
                                    nlevels = 8,
                                    edgeThreshold = 31,
                                    firstLevel = 0,
                                    WTA_K = 2,
                                    patchSize = 5,
                                    fastThreshold = 5)

        #ic(cfg, camera_num)
    def init(self, configs):
        # 初始化配置文件
        self.cfg = configs
        self.camera_num = len(self.cfg)
        self.speed_list = defaultdict(list)
        self.index = 0
        self.repeat = []

    def preprocessing(self, cal_):
        """
        校准后预处理
        :param cal_: {0: [0.28780001401901245, 0.2493000030517578], 1: [0.007499999832361937]}
        """
        cal = defaultdict(list)
        for k in cal_:
            for i in cal_[k]:
                i = torch.tensor(i)
                i = i[~torch.isnan(i)]
                i = list(i)
                speed_abs = sorted([ii.cpu().numpy() for ii in i])
                cc = float(min(np.mean(speed_abs), speed_abs[len(speed_abs) // 2 - 1]))

                cal[k].append(cc)  
        return cal
        
    def preprocessing_pix(self, cal_):
        """
        校准后预处理
        :param pix: {0: [0.28780001401901245, 0.2493000030517578], 1: [0.007499999832361937]}
        """
        cal = defaultdict(list)
        for k in cal_:
            i = cal_[k]
            i = torch.tensor(i)
            i = i[~torch.isnan(i)]
            i = list(i)
            speed_abs = sorted([ii.cpu().numpy() for ii in i])    #[4:-3]
        #     speed_abs
            cc = float(min(np.mean(speed_abs), speed_abs[len(speed_abs) // 2 - 1]))
            cal[k].append(cc)
                 
        return cal

    def preprocessing_sig(self, cal_):
        """
        校准后预处理
        :param cal_: {0: [0.28780001401901245, 0.2493000030517578], 1: [0.007499999832361937]}
        """
        cal = defaultdict(list)
        for k in cal_:
            for i in cal_[k]:
                if i == []:
                    cal[k].append(-1)
                else:
                    speed_abs = sorted([ii.cpu().numpy() for ii in i])
                    cal[k].append(float(min(np.mean(speed_abs), speed_abs[0])))
        return cal
    
    def tojson(self, a):
        b = defaultdict(list)
        for k in a:
            for i in a[k]:
                speed_abs = sorted([ii.cpu().numpy() for ii in i])
                b[k].append(float(speed_abs[0]))
        import json
        with open('cal.json', 'w') as json_file:
            json_file.write(json.dumps(b, ensure_ascii=False, indent=4))
    
    def calibration(self, meta_data, f_inds, finish=None):
        """
        校准入口
        :param meta_data_data: 一个批次的图像字节,一个批次的图像数据包含一张或多张图像字节
        :param f_inds: 一个批次的图像对应的视频流id
        :return:一个批次的图像识别结果
        """
        if finish:
            # ic(self.mean)
            cali = self.preprocessing(self.speed_list)
            Logger.info(f'pix values : {self.mean}')
            self.mean = self.preprocessing_pix(self.mean)
            Logger.info(f'pix calibration parameter : {self.mean}')
            
            # ic(self.mean)

            # for i in set(f_inds):
            #     rois = self.cfg[i]["rois"]
            #     change = self.cfg[i]["change"]
            #     if change == []:
            #         continue
            #     else:
            #         a = -1
            #         for ii in range(len(change)):
            #             if change[ii] > rois:
            #                 a = ii
            #                 break
            #         if a == -1:
            #             b = 0
            #         else:
            #             b = len(change) - a
            #         for _ in range(b):
            #             cali[i].insert(a, -1)
            #     if change[0] != 0:
            #         for _ in range(change[0]):
            #             cali[i].insert(0, -1)
            
            Logger.info(f'calibration parameter : {cali}')
            
            np.save("cal.npy", cali)
            np.save("sof.npy", self.mean)
        else:
            for i in f_inds:   
                rois = self.cfg[i]["rois"]
                if len(self.speed_list[i]) == 0:
                    for _ in range(rois):
                        self.speed_list[i].append([])
                for j in range(rois):
                    
                    roi_shape = self.cfg[i]["rois_shape"][j]
                    length = roi_shape[0] * roi_shape[1] * 3 * 2
                    input_data = meta_data[:length]
                    pre = input_data[:(length//2)].reshape(roi_shape[0], roi_shape[1], 3)
                    next = input_data[(length//2):].reshape(roi_shape[0], roi_shape[1], 3)    
                    # # 保存图片
                    # cv2.imwrite(f"cal_pre_{i}_{j}.jpg", pre)
                    # cv2.imwrite(f"cal_next_{i}_{j}.jpg", next)
                    # rec = next - pre
                    # ic(rec.shape,rec)
                    # cv2.imwrite(f"cal_rec_{i}_{j}.jpg", rec)       
                    x1 = preprocess_image_from_path(pre, bright_factor=1.1)
                    x2 = preprocess_image_from_path(next, bright_factor=1.1)
                    rgb_diff = RAFT(x1, x2)
                    # if self.cnt[str(i)+str(roi_shape[0])+str(roi_shape[1])] == 0:
                    #     cv2.imwrite(f"cal_img_{i}_{j}.jpg", pre)
                    #     cv2.imwrite(f"cal_op_{i}_{j}.jpg", rgb_diff)
                    rgb_diff = rgb_diff.reshape(1, rgb_diff.shape[0], rgb_diff.shape[1], rgb_diff.shape[2])
                    rgb_diff = np.moveaxis(rgb_diff, 3, 1)

                    rgb_diff = rgb_diff.tobytes()
                    prediction = self.engine.inference_v2([rgb_diff], imgw=roi_shape[1], imgh=roi_shape[0])[0][0] * 3.6
                    #ic(prediction)
                    self.speed_list[i][j].append(abs(prediction))
                    #self.speed_list = append_speed(self.speed_list, prediction[0][0] * 3.6, i, j)
                    # Update the picture list
                    meta_data = meta_data[length:]
                    # sparse optical
                    if self.cnt[str(i)+str(roi_shape[0])+str(roi_shape[1])] == 0:
                        old_gray = cv.cvtColor(x1, cv.COLOR_RGB2GRAY)
                        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params, useHarrisDetector=False, k=0.04)
                        
                        # # 2、用ORB寻找关键点
                        # kp, des= self.orb.detectAndCompute(old_gray, None)       #返回关键点信息及描述符
                        # p0 = np.array([[k.pt] for k in kp], dtype=np.float32)
                        good_ini = p0.copy()
                        self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])].append(old_gray)
                        self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])].append(p0)
                        self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])].append(good_ini)
                    elif self.cnt[str(i)+str(roi_shape[0])+str(roi_shape[1])] % 1 == 0:
                        frame_gray = cv.cvtColor(x1, cv.COLOR_RGB2GRAY)
                        p1, st, err = cv.calcOpticalFlowPyrLK(self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][0], frame_gray, self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][1], None, **self.lk_params)
                        good_new = p1[st == 1]
                        good_old = self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][1][st == 1]
                        # 删除静止点
                        k = 0
                        for ii, (new0, old0) in enumerate(zip(good_new, good_old)):
                            a0, b0 = new0.ravel()
                            c0, d0 = old0.ravel()
                            dist = caldist(a0, b0, c0, d0)
                            if dist >= 0.1:  # and polygon.contains(Point([c0, d0]))
                                good_new[k] = good_new[ii]
                                good_old[k] = good_old[ii]
                                self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2][k] = self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2][ii]
                                k += 1
                        Logger.info(f'k value : {k}')
                        if k != 0:
                            # 提取动态点
                            self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2] = self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2][:k]
                            good_new = good_new[:k]
                            good_old = good_old[:k]

                            distances = []
                            for new, old in zip(good_new, good_old):
                                a1, b1 = new.ravel()
                                a2, b2 = old.ravel()
                                #         world = reCalculateBBS([[c, d], [a, b]], matrix)
                                #         a1, b1, a2, b2 = world[0][0], world[0][1], world[1][0], world[1][1]
                                distance = adist(a1, b1, a2, b2)
                                distances.append(distance)
                            if len(distances) > 8:
                                distances = sorted(distances)[4:-4]

                            # ic(self.mean[str(i)+str(roi_shape[0])+str(roi_shape[1])])
                            self.mean[str(i)+str(roi_shape[0])+str(roi_shape[1])].append(np.mean(distances))
                            # 更新
                            old_gray = frame_gray.copy()
                            p0 = good_new.reshape(-1, 1, 2)

                            if self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2].shape[0]<40:
                                p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params, useHarrisDetector=False, k=0.04)
                        
                                # kp, des= self.orb.detectAndCompute(old_gray, None)       #返回关键点信息及描述符
                                # p0 = np.array([[k.pt] for k in kp], dtype=np.float32)
                                good_ini=p0.copy()
                            self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][0] = old_gray
                            self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][1] = p0
                            self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2] = good_ini
                    
                    self.cnt[str(i)+str(roi_shape[0])+str(roi_shape[1])] += 1
                    Logger.info(f'meta_data: {len(meta_data)}')
            # Logger.info(f'flow velocity value: {self.speed_list}')
            # ic(self.mean)
            return self.speed_list
        
        
    def detect(self, meta_data, f_inds):
        """
        检测入口
        :param meta_data_data: 一个批次的图像字节,一个批次的图像数据包含一张或多张图像字节
        :param f_inds: 一个批次的图像对应的视频流id
        :return:一个批次的图像识别结果
        """

        for i in f_inds:   
            rois = self.cfg[i]["rois"]
            if len(self.speed_list[i]) == 0:
                for _ in range(rois):
                    self.speed_list[i].append(0)
            for j in range(rois):
                
                roi_shape = self.cfg[i]["rois_shape"][j]
                length = roi_shape[0] * roi_shape[1] * 3 * 2
                input_data = meta_data[:length]
                pre = input_data[:(length//2)].reshape(roi_shape[0], roi_shape[1], 3)
                next = input_data[(length//2):].reshape(roi_shape[0], roi_shape[1], 3)           
                x1 = preprocess_image_from_path(pre)
                x2 = preprocess_image_from_path(next)
                rgb_diff = RAFT(x1, x2)
                rgb_diff = rgb_diff.reshape(1, rgb_diff.shape[0], rgb_diff.shape[1], rgb_diff.shape[2])
                rgb_diff = np.moveaxis(rgb_diff, 3, 1)

                rgb_diff = rgb_diff.tobytes()
                prediction = self.engine.inference_v2([rgb_diff], imgw=roi_shape[1], imgh=roi_shape[0])[0][0] * 3.6
                ic(prediction)
                self.speed_list[i][j] = abs(prediction)
                #self.speed_list = append_speed(self.speed_list, prediction[0][0] * 3.6, i, j)
                # Update the picture list
                meta_data = meta_data[length:]
                ic(len(meta_data))
        ic(self.speed_list)
        tojson(self.speed_list)
        return self.speed_list
        
    def run_model(self, imgdata, size):
        """
        运行模型，得到识别结果
        :param imgdata: 一个批次的图像字节,一个批次的图像数据包含一张或多张图像字节
        :param img_h:图像宽
        :param img_w:图像高
        :param f_inds: 一个批次的图像对应的视频流id
        :return:一个批次的图像识别结果
        """
        prediction = engine.inference_v2([rgb_diff], imgw=size[1], imgh=size[0])
        prediction = prediction[0][0].cpu().detach().numpy() * 3.6
        # feats = get_mc().feat_extract(imgdata)
        # Logger.info(f'feats finish...{feats.shape}')
        t0 = time.time()
        if get_mc().mode == models.MODE_TRACKING and get_mc().tracker == models.DEEPSORT_TRACKING:
            # 当需要使用目标切图时，box_ts=True同时返回目标张量
            batch_results, boxes_ts_l = get_mc().detect(imgdata, box_ts=True)
        else:
            batch_results = get_mc().detect(imgdata)

        Logger.info(f'detect cost:{time.time() - t0}')
        t0 = time.time()
        batch_size = len(batch_results)
        outputs = []  # [[]] * batch_size
        if get_mc().mode == models.MODE_TRACKING:  # 跟踪
            results = []
            if get_mc().tracker == models.DEEPSORT_TRACKING:
                # execute reid inference
                boxes_merged_ts_l = []
                boxes_merged_batch_ind = []
                for i, boxes_tss in enumerate(boxes_ts_l):
                    boxes_merged_ts_l.extend(boxes_tss)
                    boxes_merged_batch_ind.extend([i] * len(boxes_tss))
                boxes_merged_batch_ind = torch.IntTensor(boxes_merged_batch_ind)
                merged_feats = get_mc().feat_extract(boxes_merged_ts_l, pre_proc=False)

                Logger.info(f'feature extract cost:{time.time() - t0}')
                t0 = time.time()

                for batch_ind in range(batch_size):  # merge_flag
                    # if merge_flag:
                    batch_feats = merged_feats[boxes_merged_batch_ind == batch_ind]
                    result = work_man.send(f_inds[batch_ind], [[batch_results[batch_ind][0],
                                                                batch_results[batch_ind][2],
                                                                batch_feats,
                                                                batch_results[batch_ind][1]]])
                    results.append(result)
            elif get_mc().tracker == models.BYTETRACKING:
                # byte tracking
                for batch_ind in range(batch_size):  # merge_flag
                    # if merge_flag:

                    result = work_man.send(f_inds[batch_ind], [[batch_results[batch_ind][0],
                                                                batch_results[batch_ind][2],
                                                                batch_results[batch_ind][1]]])
                    results.append(result)

            for batch_ind in range(batch_size):
                ret_obj = results[batch_ind].get()[0]
                outputs.append(ret_obj)
            # outputs[batch_ind] = ret_obj[0]
            # Logger.info(f'{batch_size},results len:{outputs}')
            Logger.info(f'tracking cost:{time.time() - t0}')
        elif get_mc().mode == models.MODE_DETECT:
            for batch_ind in range(batch_size):
                ret_obj = batch_results[batch_ind][:2]
                outputs.append(ret_obj)
        elif get_mc().mode == models.MODE_SEGMENTATION:
            for batch_ind in range(batch_size):
                ret_obj = batch_results[batch_ind]
                outputs.append(ret_obj)
        return outputs  # xyxy_cls_conf, tracking_outputs

class Manager_v1(object):
    """
    基础检测器
    """

    def __init__(self, configs):
        super(Manager_v1, self).__init__()
        # 各摄像头的事件检测器,各摄像头加载的算法类型列表
        v = 0     # model version
        in_c = 3  # number of input channels
        num_c = 1 # number of classes to predict
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        # model = EfficientNet.from_pretrained(f'efficientnet-b{v}', in_channels=in_c, num_classes=num_c)
        self.model = EfficientNet.from_pretrained(f'efficientnet-b{v}', './application/common/engine_file/model.pth',in_channels=in_c, num_classes=num_c)
        self.model.to(self.device)
        self.model.eval()

        self.index = defaultdict(int)
        self.sof = defaultdict(list)
        self.mean = defaultdict(list)
        self.feature_params = dict(maxCorners=6, qualityLevel=0.01, minDistance=3, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))

    #     self.engine = water_flow_velocity_detection(
    #     engine_file=r'./application/common/engine_file/model.432FP32.engine',
    #     binding_names=['inputs', 'outputs'],
    #     names=['bg', 'water'],
    #     half=False,
    #     device=0
    # )
        
        
        #ic(cfg, camera_num)
    def init(self, configs):
        Logger.info(f'init manager..|||||||||||||||||.')
        # 初始化配置文件
        self.cfg = configs
        self.camera_num = len(self.cfg)
        self.index = defaultdict(int)
        self.pix_index = defaultdict(int)
        self.speed_list = defaultdict(list)
        self.speed_batch = defaultdict(list)
        self.repeat = []
        # self.cal_camera = np.load("cal.npy", allow_pickle=True).tolist()
        # self.cal_sof = np.load("sof.npy", allow_pickle=True).tolist()
        self.abnorm = 30 * [[0]*10]
        self.abnorm_meta = self.abnorm_meta = [[[] for _ in range(10)] for _ in range(30)]
        self.abindex = 0

        self.sof = defaultdict(list)
        self.mean = defaultdict(list)
        self.feature_params = dict(maxCorners=10, qualityLevel=0.03, minDistance=2.5, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))
        self.orb = cv.ORB_create(20,
                                    nlevels = 8,
                                    edgeThreshold = 31,
                                    firstLevel = 0,
                                    WTA_K = 2,
                                    patchSize = 4,
                                    fastThreshold = 5)


    def cal(self, cal_npy, sof_npy, configs):
        self.cfg = configs
        self.cal_camera_ = np.load("cal.npy", allow_pickle=True).tolist()
        self.cal_sof_ = np.load("sof.npy", allow_pickle=True).tolist()
        # for i in self.cfg:
        #     for ii in range(len(self.cal_camera_[i])):
        #         if self.cal_camera_[i][ii] == -1:
        #             self.cal_camera_[i][ii] = self.cal_camera[i][ii]
        #         else:
        #             continue
        self.cal_camera = self.cal_camera_
        from copy import deepcopy
        self.cal_camera_init = deepcopy(self.cal_camera_)
        self.cal_sof = self.cal_sof_
    
    def tojson(self, a):
        b = defaultdict(list)
        for k in a:
            for i in a[k]:
                speed_abs = sorted([ii.cpu().numpy() for ii in i])
                b[k].append(float(speed_abs[0]))
        return b
        # import json
        # with open('cal.json', 'w') as json_file:
        #     json_file.write(json.dumps(b, ensure_ascii=False, indent=4))
    def tensor2float(self, cal_):
        for index, k in enumerate(cal_):
            speed_abs = [float(ii.cpu()) for ii in k]
            cal_[index] = speed_abs
        # b = []
        # for index, k in enumerate(cal_):
        #     b.append([])
        #     for i in cal_[k]:
        #         speed_abs = [float(ii.cpu()) for ii in i]
        #         b[index].append(speed_abs)
        return cal_
    def calibration(self, meta_data, f_inds, finish=None):
        """
        校准入口
        :param meta_data_data: 一个批次的图像字节,一个批次的图像数据包含一张或多张图像字节
        :param f_inds: 一个批次的图像对应的视频流id
        :return:一个批次的图像识别结果
        """
        if finish:
            cali = self.preprocessing(self.speed_list)
            np.save("cal.npy", cali)
        for i in f_inds:   
            rois = self.cfg[i]["rois"]
            if len(self.speed_list[i]) == 0:
                for _ in range(rois):
                    self.speed_list[i].append([])
            for j in range(rois):
                
                roi_shape = self.cfg[i]["rois_shape"][j]
                length = roi_shape[0] * roi_shape[1] * 3 * 2
                input_data = meta_data[:length]
                pre = input_data[:(length//2)].reshape(roi_shape[0], roi_shape[1], 3)
                next = input_data[(length//2):].reshape(roi_shape[0], roi_shape[1], 3)           
                x1 = preprocess_image_from_path(pre)
                x2 = preprocess_image_from_path(next)
                rgb_diff = RAFT(x1, x2)
                rgb_diff = rgb_diff.reshape(1, rgb_diff.shape[0], rgb_diff.shape[1], rgb_diff.shape[2])
                rgb_diff = np.moveaxis(rgb_diff, 3, 1)
                rgb_diff = torch.from_numpy(rgb_diff).to(device)

                prediction = model(rgb_diff)
                # rgb_diff = rgb_diff.tobytes()
                # prediction = self.engine.inference_v2([rgb_diff], imgw=roi_shape[1], imgh=roi_shape[0])[0][0] * 3.6
                #ic(prediction)
                self.speed_list[i][j].append(abs(prediction))
                #self.speed_list = append_speed(self.speed_list, prediction[0][0] * 3.6, i, j)
                # Update the picture list
                meta_data = meta_data[length:]
                ic(len(meta_data))
        ic(self.speed_list)
        cali = self.preprocessing(self.speed_list)
        np.save("cal.npy", cali)
        return self.speed_list
        
    def detect(self, meta_data, f_inds):
        """
        检测入口
        :param input_data: 一个批次的图像字节,一个批次的图像数据包含一张或多张图像字节
        :param img_h:图像宽
        :param img_w:图像高
        :param f_inds: 一个批次的图像对应的视频流id
        :return:一个批次的图像识别结果
        """
        speed_list = defaultdict(list)
        speed_index = []
        for ii, i in enumerate(f_inds):   
            rois = self.cfg[i]["rois"]
            if len(self.speed_list[i]) == 0 or rois!=len(self.speed_list[i]):
                # for _ in range(rois):
                self.speed_list[i] = [0] * rois
            # if self.speed_batch[i] == []:
            #     for _ in range(rois):
            #         self.speed_batch[i].append([])
            speed_index.append([])
            for j in range(rois):
                # ic(i, j, self.speed_batch[i])
                roi_shape = self.cfg[i]["rois_shape"][j]
                length = roi_shape[0] * roi_shape[1] * 3 * 2
                
                input_data = meta_data[:length]
                # ic(i, length, roi_shape[0], roi_shape[1], len(input_data), len(meta_data))
                pre = input_data[:(length//2)].reshape(roi_shape[0], roi_shape[1], 3)
                next = input_data[(length//2):].reshape(roi_shape[0], roi_shape[1], 3)
                # # 保存图片
                # if int(i) in [50, 51, 52, 55, 56]:
                #     ss = self.index[tuple(self.cfg[i]["left_top"][j])]
                #     cv2.imwrite(f"img/img_{i}_{j}_{ss}.jpg", pre)

                x1 = preprocess_image_from_path(pre)
                x2 = preprocess_image_from_path(next)
                rgb_diff = opticalFlowDense(x1, x2)
                rgb_diff = rgb_diff.reshape(1, rgb_diff.shape[0], rgb_diff.shape[1], rgb_diff.shape[2])
                rgb_diff = np.moveaxis(rgb_diff, 3, 1)
                rgb_diff = torch.from_numpy(rgb_diff).to(self.device)
                with torch.no_grad():
                    prediction = self.model(rgb_diff)
                    prediction[0][0] = prediction[0][0].cpu()
                # 清空显存缓存
                torch.cuda.empty_cache()
                if torch.isnan(prediction[0][0]):
                    prediction[0][0] = self.cal_camera[i][j] / 3.6
                # prediction = prediction.cpu().detach().numpy()
                # rgb_diff = rgb_diff.tobytes()
                # prediction = self.engine.inference_v2([rgb_diff], imgw=roi_shape[1], imgh=roi_shape[0])
                Logger.info(f'flow velocity value: {abs(prediction[0][0] * 3.6)} calibration parameter: {self.cal_camera[i][j]}')
                
                if self.index[tuple(self.cfg[i]["left_top"][j])] == 0 or self.abnorm_meta[ii][j] == []:
                    if abs(abs(prediction[0][0].cpu() * 3.6) - self.cal_camera[i][j]) / self.cal_camera[i][j] > 0.3:
                        self.speed_list[i][j] = abs(prediction[0][0].cpu() * 3.6)
                        # self.speed_batch[i][j].append(abs(prediction[0][0] * 3.6))
                        self.abnorm[ii][j] += 1
                        self.abnorm_meta[ii][j].append(abs(prediction[0][0].cpu() * 3.6))
                    else:   
                        self.speed_list[i][j] = abs(prediction[0][0].cpu() * 3.6)
                        # self.speed_batch[i][j].append(abs(prediction[0][0] * 3.6))
                    predict = torch.clamp(abs(prediction[0][0].cpu() * 3.6), self.cal_camera[i][j] * 0.85, self.cal_camera[i][j] *1.15) + torch.rand(1) *0.01
                    speed_index[-1].append(predict)
                else:

                    if self.abnorm[ii][j] > 20:   # 2
                        predict = mean(abs(prediction[0][0].cpu() * 3.6), self.speed_list[i][j], self.index[tuple(self.cfg[i]["left_top"][j])])
                        #self.speed_list[i][j] = predict
                        # predict = torch.clamp(predict.cpu(), self.cal_camera[i][j] * 0.85, self.cal_camera[i][j] *1.15)
                        self.abnorm[ii][j] = 0
                        self.abnorm_meta[ii][j] = []
                        self.abindex = 1
                        # self.cal_camera[i][j] = [self.cal_camera[i][j] + predict.cpu()/3 if predict.cpu() - self.cal_camera[i][j] > 0 else self.cal_camera[i][j] - predict.cpu()/3][0]

                    if self.abnorm[ii][j] <= 20 and abs(abs(prediction[0][0].cpu() * 3.6) - self.cal_camera[i][j]) / self.cal_camera[i][j] <= 0.3:
                        predict = mean(abs(prediction[0][0].cpu() * 3.6), self.speed_list[i][j], self.index[tuple(self.cfg[i]["left_top"][j])], self.abnorm_meta[ii][j])

                        #self.speed_list[i][j] = predict
                        self.abnorm[ii][j] = 0
                        self.abnorm_meta[ii][j] = []
                    if self.abnorm[ii][j] <= 20 and abs(abs(prediction[0][0].cpu() * 3.6) - self.cal_camera[i][j]) / self.cal_camera[i][j] > 0.3:
                        self.abnorm[ii][j] += 1
                        self.abnorm_meta[ii][j].append(abs(prediction[0][0].cpu() * 3.6))

                        predict = mean(abs(prediction[0][0].cpu() * 3.6), self.speed_list[i][j], self.index[tuple(self.cfg[i]["left_top"][j])])
                    # if self.abindex != 1:
                    #     self.abindex = 0
                    if predict < 0:
                        predict = self.cal_camera[i][j] + torch.rand(1) *0.01
                    self.speed_list[i][j] = predict.cpu()
                    # self.speed_batch[i][j].append(predict)
                    speed_index[-1].append(predict)
                    #predicted_speed.pop(0)
                # Update the picture list
                meta_data = meta_data[length:]
                

                # sparse optical
                if self.index[tuple(self.cfg[i]["left_top"][j])] == 0:
                    old_gray = cv.cvtColor(x1, cv.COLOR_RGB2GRAY)
                    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params, useHarrisDetector=False, k=0.04)
                        
                    # kp, des= self.orb.detectAndCompute(old_gray, None)       #返回关键点信息及描述符
                    # p0 = np.array([[k.pt] for k in kp], dtype=np.float32)
                    good_ini = p0.copy()
                    self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])].append(old_gray)
                    self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])].append(p0)
                    self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])].append(good_ini)
                elif self.index[tuple(self.cfg[i]["left_top"][j])] % 1 == 0:
                    frame_gray = cv.cvtColor(x1, cv.COLOR_RGB2GRAY)
                    p1, st, err = cv.calcOpticalFlowPyrLK(self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][0], frame_gray, self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][1], None, **self.lk_params)
                    good_new = p1[st == 1]
                    good_old = self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][1][st == 1]
                    # 删除静止点
                    k = 0
                    for kk, (new0, old0) in enumerate(zip(good_new, good_old)):
                        a0, b0 = new0.ravel()
                        c0, d0 = old0.ravel()
                        dist = caldist(a0, b0, c0, d0)
                        if dist >= 0.1:  # and polygon.contains(Point([c0, d0]))
                            good_new[k] = good_new[kk]
                            good_old[k] = good_old[kk]
                            self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2][k] = self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2][kk]
                            k += 1
                    if k != 0:
                        ic(k)
                        # 提取动态点
                        self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2] = self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2][:k]
                        good_new = good_new[:k]
                        good_old = good_old[:k]

                        distances = []
                        for new, old in zip(good_new, good_old):
                            a1, b1 = new.ravel()
                            a2, b2 = old.ravel()
                            #         world = reCalculateBBS([[c, d], [a, b]], matrix)
                            #         a1, b1, a2, b2 = world[0][0], world[0][1], world[1][0], world[1][1]
                            distance = adist(a1, b1, a2, b2)
                            distances.append(distance)
                        
                        if len(distances) > 8:
                            distances = sorted(distances)[4:-4]

                        self.mean[tuple(self.cfg[i]["left_top"][j])].append(np.mean(distances))
                        # 更新
                        Logger.info(f'distances: {distances}  {self.mean[tuple(self.cfg[i]["left_top"][j])]}')
                        old_gray = frame_gray.copy()
                        p0 = good_new.reshape(-1, 1, 2)

                        if self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2].shape[0]<40:
                            p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params, useHarrisDetector=False, k=0.04)
                        
                            # kp, des= self.orb.detectAndCompute(old_gray, None)       #返回关键点信息及描述符
                            # p0 = np.array([[k.pt] for k in kp], dtype=np.float32)
                            good_ini=p0.copy()
                        self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][0] = old_gray
                        self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][1] = p0
                        self.sof[str(i)+str(roi_shape[0])+str(roi_shape[1])][2] = good_ini
                if self.index[tuple(self.cfg[i]["left_top"][j])] != 0 and self.index[tuple(self.cfg[i]["left_top"][j])] % 120 == 0 and len(self.mean[tuple(self.cfg[i]["left_top"][j])]) > 90:
                    ic(len(self.mean[tuple(self.cfg[i]["left_top"][j])]))
                    pix_list = sorted(self.mean[tuple(self.cfg[i]["left_top"][j])])
                    if (min(np.mean(pix_list[45:-45]), pix_list[len(pix_list)//2 - 1]) / self.cal_sof[tuple(self.cfg[i]["left_top"][j])]) < 0.6:
                        self.cal_camera[i][j] = self.cal_camera_init[i][j] * 0.8
                    elif (min(np.mean(pix_list[45:-45]), pix_list[len(pix_list)//2 - 1]) / self.cal_sof[tuple(self.cfg[i]["left_top"][j])]) > 3:
                        self.cal_camera[i][j] = self.cal_camera_init[i][j] 
                    if self.cal_camera[i][j] > 5:
                        self.cal_camera[i][j] /= 2 
                    Logger.info(
                        f'pix120 {str(i) + str(roi_shape[0]) + str(roi_shape[1])} {len(self.mean[tuple(self.cfg[i]["left_top"][j])])} {self.index[tuple(self.cfg[i]["left_top"][j])]} {self.cal_camera_init[i][j]} {self.cal_camera[i][j]} {pix_list[45:-45]} {min(np.mean(pix_list[45:-45]), pix_list[len(pix_list)//2 - 1])} {self.cal_sof[str(i) + str(roi_shape[0]) + str(roi_shape[1])]} {self.index[tuple(self.cfg[i]["left_top"][j])]}')
                    self.mean[tuple(self.cfg[i]["left_top"][j])] = []
                self.index[tuple(self.cfg[i]["left_top"][j])] += 1
                # Logger.info(f'meta_data: {len(meta_data)}')
        Logger.info(f'flow velocity value: {self.speed_list}')
        speed_index = self.tensor2float(speed_index)
        # self.speed_batch = defaultdict(list)
        #self.tojson(self.speed_list)
        return speed_index

def get_manager(configs=None):
    global manager
    if configs is not None:
        if manager is not None:     
            manager.init(configs)
        else:
            manager = Manager(configs)
            manager.init(configs)
        Logger.info(f'manager init ...')
    return manager

def get_manager_v1(configs=None, cal_camera=None):
    global manager1
    if configs is not None:
        if manager1 is not None:
            manager1.init(configs)
        else:
            manager1 = Manager_v1(configs)
            manager1.init(configs)
        Logger.info(f'manager init ...')
    return manager1

