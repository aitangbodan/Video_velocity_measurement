#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮
# Gitlab: ECCOM
# Creat Time:  2022/6月/23 15:56
# File: server.py
# Project: TRT_inference
# Software: PyCharm
"""
Function:

"""
import sys

#from boto import config

import cv2

sys.path.append('./')

import json
import numpy as np
import time
import os
from pysharemem.ShareMemory import ShareMemory
from utils.log import logger
from application import get_manager


# json 序列化
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def ai_job(img, json_ms):
    # logging.warning('---------------')
    size = json_ms["size"]
    frame_inds = json_ms['frame_inds']
    data_transform, alg_run = 0, 0
    t0 = time.time()
    preds = get_manager().detect([img], frame_inds)
    alg_run += (time.time() - t0)
    # logging.warning(f'preds:{alg_run}')
    result = dict()
    e = time.time()
    result["rec"] = preds
    result["time"] = (e - t0)
    # logging.warning(f'running time:{e - s}')
    return json.dumps(result, cls=NumpyEncoder)


# 共享内存运行的主逻辑
def share_memory_main():
    share = ShareMemory(int(sys.argv[1]), os.getpid())  # 创建共享内存
    share.do(ai_job)  # 添加算法任务函数
    config = share.get_config()
    logger.info(config)
    get_manager(config)
    # 使用config进行相关初始化
    # ======

    # print(share.get_config())
    # print(share.get_config().dump())
    while True:
        logger.info("[Python]: ++++++++++心跳包，表明程序活着++++++++++")
        try:
            os.kill(share.host_pid(), 0)
        except Exception as e:
            exit("[Python]: ++++++++++主进程已意外死亡 - Python自杀退出++++++++++")
        time.sleep(10)


# 启动gRPC服务
if __name__ == '__main__':
    share_memory_main()
    # serve()
