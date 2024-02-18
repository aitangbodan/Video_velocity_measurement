#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/8月/16 17:40  
# File: sub_work.py  
# Project: TRT_inference    
# Software: PyCharm   
"""
Function:
    
"""
import time
from multiprocessing import Process
from utils.log import logger
from model.models import tracking_from_list


class BaseWork(Process):
    def __init__(self, p):
        super(BaseWork, self).__init__()
        self.p = p


class TrackingWork(BaseWork):
    def __init__(self, tracker, p):
        """
        跟踪算法进程
        :param tracker:跟踪器
        :param p:管道连接端
        """
        super(TrackingWork, self).__init__(p)
        self.tracker = tracker
        self.daemon = True
        self.running = False

    def run(self):
        if not self.running:
            self.running = True
        while self.running:
            data = self.p.recv()
            if data is not None:
                # logger.info(f'子进程{self.tracker}收到消息:{data}')
                t0 = time.time()
                result = tracking_from_list(self.tracker, data)
                # logger.info(f'tracking_from_list cost:{time.time() - t0} ')
                self.p.send(result)
