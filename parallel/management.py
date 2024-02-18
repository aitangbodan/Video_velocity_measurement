#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/8月/16 18:14  
# File: management.py  
# Project: TRT_inference    
# Software: PyCharm   
"""
Function:
    
"""
from multiprocessing import Pipe
from .sub_work import TrackingWork


class WorkReply(object):
    def __init__(self, p):
        self._p = p

    def get(self):
        data = self._p.recv()
        return data


class WorkManager(object):
    def __init__(self):
        self.works = []  # 工作站
        self.main_conn = []  # 当前进程管道收发端
        self.sub_conn = []  # 子任务管道收发端

    def add_work(self, target):
        """
        新增子工作站
        :param work:
        :return:
        """

        p1, p2 = Pipe(duplex=True)  # 创建通信管道
        work = TrackingWork(target, p2)
        self.works.append(work)
        self.main_conn.append(p1)
        self.sub_conn.append(p2)
        work.start()

    def send(self, w_ind, datas):
        self.main_conn[w_ind].send(datas)
        reply = WorkReply(self.main_conn[w_ind])
        return reply
