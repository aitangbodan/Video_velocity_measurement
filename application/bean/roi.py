#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2021/11月/03 11:36  
# File: roi.py  
# Project: smart_service_area    
# Software: PyCharm   
"""
Function:
    
"""
import numpy as np
import math
import cv2


class BaseROI(object):
    def __init__(self, id, vertexes, type, desc=None, algs=[]):
        """
        虚拟线圈基类
        Args:
            id:线圈id
            vertexes:线圈轮廓点
            type:线圈类型
            desc:描述
        """
        self.id = id
        self.vertexes = np.array(vertexes)
        self.type = type
        self.desc = desc
        self.algs = algs

    def _cross_point(self, p1, p2, p3, p4):
        """
        计算两条直线的交点，其中p1，p2是直线A上的不同的两点，p3，p4是直线B上的不同的两点，
        ka = y2-y1/x2-x1
        kb = y4-y3/x4-x3
        Args:
            p1:直线A上的点1
            p2:直线A上的点2
            p3:直线B上的点1
            p4:直线B上的点2
        Returns:交点坐标

        """
        ka = (p2[1] - p1[1]) / (p2[0] - p1[0]) if (p2[0] - p1[0]) != 0 else (p2[1] - p1[1])
        kb = (p4[1] - p3[1]) / (p4[0] - p3[0]) if (p4[0] - p3[0]) != 0 else (p4[1] - p3[1])
        # print(f'ka={ka},kb={kb}')
        # d = ((p1[0] - p2[0]) * (p3[1] - p4[1])) - ((p1[1] - p2[1]) * (p3[0] - p3[0]))
        # 平行
        if math.fabs(ka - kb) < 0.05:
            return -1
        x = (ka * p1[0] - p1[1] - kb * p3[0] + p3[1]) / (ka - kb)
        y = (ka * kb * (p1[0] - p3[0]) + ka * p3[1] - kb * p1[1]) / (ka - kb)
        # x = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])) / d
        # y = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])) / d
        return x, y


class ParkingROI(BaseROI):
    def __init__(self, id, vertexes, type, resolution, desc=None):

        super(ParkingROI, self).__init__(id, vertexes, type, desc)
        if len(self.vertexes) != 4:
            raise ValueError('length of vertexes must be 4')

        self.resolution = resolution
        # 停车位划分为3x3的子区域
        self.row = 3
        self.col = 3
        self.weights = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])  # 各区域权重占比
        self.center, self.sub_rects = self._init_parking_space()  # 车位中心点，车位子区域划分
        self.mask = self._init_mask(resolution)  # 车位区域的二值化掩码图
        self.w_area = self._calc_area_with_weight(self.mask, 10, 1)  # 权重的车位区域面积
        self.stat = False  # 是否被占用
        self.occu_rate = 0  # 车位占用面积百分比
        self.car_info = None  # 车位上的占用车辆信息[['车型',置信度,[x1,y1,x2,y2]]]

    def _init_mask(self, resolution):
        mask = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
        for i in range(self.row):
            for j in range(self.col):
                cv2.fillPoly(mask, [self.sub_rects[i, j]], i * self.col + j + 1)
        return mask

    def _calc_area_with_weight(self, mask, min_len, min_value):
        """
        计算带权重的roi面积，以掩码出现次数作为掩码面积
        Args:
            mask:感兴趣区域掩码
            min_len:最小计数长度，mask中的最大值+1
            min_ind:统计大于等于min_value的带权重频次

        Returns:带权重的面积
        """
        xmin, ymin = np.min(self.vertexes[:, 0]), np.min(self.vertexes[:, 1])
        xmax, ymax = np.max(self.vertexes[:, 0]), np.max(self.vertexes[:, 1])
        ps_rect = mask[ymin:ymax, xmin:xmax]
        ps_arr = ps_rect.reshape((ps_rect.shape[0] * ps_rect.shape[1]))
        bc = np.bincount(ps_arr, minlength=min_len)
        w_area = 0
        for i, c in enumerate(bc):
            if i < min_value:
                continue
            row = (i - min_value) // self.col
            col = (i - min_value) % self.col
            w_area += self.weights[row, col] * c
        return w_area

    def _init_parking_space(self):
        """
        划分四边形网格
        :return:
        """
        point_num_x = self.col + 1
        point_num_y = self.row + 1
        sub_rects = np.zeros((self.row, self.col, 4, 2))
        points = np.zeros((point_num_y, point_num_x, 2))
        # 四边形中心点
        center = self._cross_point(self.vertexes[0], self.vertexes[2], self.vertexes[1], self.vertexes[3])
        center = np.array(center, dtype=np.int)

        # 四边形边上的分割点计算
        for i in range(point_num_x):
            point = self.vertexes[0] + (self.vertexes[1] - self.vertexes[0]) * i / self.col
            points[0, i, :] = point
        for i in range(point_num_y):
            point = self.vertexes[1] + (self.vertexes[2] - self.vertexes[1]) * i / self.row
            points[i, self.col, :] = point
        for i in range(point_num_x):
            point = self.vertexes[3] + (self.vertexes[2] - self.vertexes[3]) * i / self.col
            points[self.row, i, :] = point
        for i in range(point_num_y):
            point = self.vertexes[0] + (self.vertexes[3] - self.vertexes[0]) * i / self.row
            points[i, 0, :] = point
        # 四边形内部分割点计算
        for i in range(1, self.row):
            for j in range(1, self.col):
                point_ij = self._cross_point(points[i, 0, :], points[i, self.col, :],
                                             points[0, j, :], points[self.row, j, :])
                points[i, j, :] = point_ij

        for i in range(self.row):
            for j in range(self.col):
                sub_rects[i, j, 0] = points[i, j]
                sub_rects[i, j, 1] = points[i, j + 1]
                sub_rects[i, j, 2] = points[i + 1, j + 1]
                sub_rects[i, j, 3] = points[i + 1, j]
        sub_rects = sub_rects.astype(np.int)
        # points = points.astype(np.int)
        return center, sub_rects  # , points

    def _calc_weighted_occupacy(self, rect_mask):
        """
        计算停车位带权重的占用率
        :param rect:(xmin,ymin,xmax,ymax)
        :param size:图像分辨率
        :return:
        """

        # weights_area = 0
        box_mask = self.mask + rect_mask
        w_area = self._calc_area_with_weight(box_mask, 19, 10)
        # xmin, ymin = np.min(self.vertexes[:, 0]), np.min(self.vertexes[:, 1])
        # xmax, ymax = np.max(self.vertexes[:, 0]), np.max(self.vertexes[:, 1])
        # ps_rect = box_mask[ymin:ymax, xmin:xmax]
        # ps_arr = ps_rect.reshape((ps_rect.shape[0] * ps_rect.shape[1]))
        # bc = np.bincount(ps_arr, minlength=19)
        # w_area = 0
        # for i, c in enumerate(bc):
        #     if i < 10:
        #         continue
        #     row = (i - 10) // self.col
        #     col = (i - 10) % self.col
        #     w_area += self.weights[row, col] * c

        # print('w_area', w_area)
        # for i in range(self.row):
        #     for j in range(self.col):
        #         box_mask = self.masks[i, j]
        #         t2 = time.time()
        #         inters_mask = cv2.bitwise_and(rect_mask, box_mask)
        #         t3 = time.time()
        #
        #         area = np.sum(inters_mask > 0)
        #         t4 = time.time()
        #         weights_area += self.weights[i, j] * area
        # print('weights_area',weights_area)
        # e = time.time()
        # print('初始化车位掩码:', t4 - t0)

        # print('填充车位:', t2 - t1)
        # print('计算逻辑与:', t3 - t2)
        # print('计算面积:', t4 - t3)
        # print('total cost:', e - s)

        return w_area

    def calc_weights_occupacies(self, rect_masks):
        """
        计算停车位与所有rects候选框的带权重面积占用率
        :param rects:
        :param size: 图像分辨率
        :return:
        """
        occu_l = []
        for rect_mask in rect_masks:
            w_occu = self._calc_weighted_occupacy(rect_mask)
            occu_l.append(w_occu / self.w_area)
        return np.array(occu_l)

    def in_space(self, points):
        """
        检查points中的点是否在roi区域内
        :param points:
        :return: list
        """
        in_l = []
        for point in points:
            in_l.append(True if self.mask[point[1], point[0]] > 0 else False)
        return np.array(in_l)
