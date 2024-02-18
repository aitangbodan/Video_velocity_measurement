#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 王根一 
# Gitlab: ECCOM  
# Creat Time:  2022/11月/23 17:02  
# File: server.py
# Project: TRT_inference    
# Software: PyCharm   
"""
Function:
    
"""
import os
import sys


sys.path.append('./')

import json
import numpy as np
import time
# 引入grpc依赖
from concurrent import futures
import grpc

import dto_pb2
import dto_pb2_grpc
import cv2

from application import get_manager, get_manager_v1, get_manager_v2
import multiprocessing as mp
from utils import logger
import pycuda.autoinit as cudainit

# import utils.logging
data_transfor, alg_run = 0, 0
import threading
from pysharemem.ShareMemory import ShareMemory


# json 序列化
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
share = ShareMemory(int(sys.argv[1]), os.getpid())

class GRPCServer(dto_pb2_grpc.CVServiceServicer):
    def __init__(self, device):
        super(GRPCServer, self).__init__()
        self.device = device
        self.index = 0
        self.cal_index = None
        self.cal_index_ = None
        self.cal_npy = "../cal.npy"
        self.sof_npy = "../sof.npy"
        self.update_config = None
        self.service_config = None
        self.cal_model = None
        self.cal_model_index = None
        self.inf_model = None
        self.result = []
        self.count = 0

    def imganalyse(self, request, context):
        logger.info('---------------')
        # global data_transfor, alg_run
        import json
        alg_run = 0
        s = time.time()
        t0 = time.time()
        # t0 = time.time()
        imgW = request.scale_imgW
        imgH = request.scale_imgH
        input_param = json.loads(request.json_input)
        cal = input_param["cal"]
        frame_inds = input_param['frame_inds']
        finish = input_param['finish']
        imgs = share.get_data()
        imgs = np.array(imgs, dtype=np.uint8)
       
        # logger.info(f"imgs shape:{imgs.shape}")
        logger.info(f"imgs len:{len(imgs)}")
        preds = []
        # logger.info(f'frame_inds:{frame_inds}')
        if cal:
            if self.cal_model_index == None:
                self.cal_model = get_manager(self.service_config)
                self.cal_model_index = 1
            self.cal_model.calibration(imgs, frame_inds, finish)  # finish=True or 1表示传完了，生成校准数据  
            # self.cal_index = 1
            self.cal_index_ = 0

        else:
            self.cal_model_index = None
            # self.cal_index = 0
            # if self.index == 0:
            #     self.inf_model = get_manager_v1(self.service_config)
            #     self.index += 1
            if self.cal_index_ == 0:
                self.inf_model.cal(self.cal_npy, self.sof_npy, self.service_config)
                self.cal_index_ = 1
            preds = self.inf_model.detect(imgs, frame_inds)

        # preds = self.engine.inference_v2([request.imgdata], imgW, imgH)
        # preds = get_mc().run([request.imgdata], imgH, imgW)
        # logger.info(f'preds:{preds}')
        # if self.cal_index == 1:
        #     self.cal_index_ = 0
        alg_run += (time.time() - t0)
        logger.info(f'preds:{preds}')
        result = dict()
        result["rec"] = preds
        e = time.time()
        self.result.append(result)
        self.count += 1
        if self.count == 7200:
            with open('predict.json', 'w') as json_file:
                json_file.write(json.dumps(self.result, ensure_ascii=False, indent=4))

        reply = dto_pb2.AIReply(
            # 创建python客户端代码
            stream_id=request.stream_id,
            json_out=json.dumps(result, cls=NumpyEncoder),
            send_time=(e - s)
        )
        e = time.time()
        logger.info(f'running time:{e - s}')
        return reply

    def configuration(self, request, context, cal=None):
        logger.info(f'pid:{os.getpid()} tid:{threading.currentThread().ident}')
        t0 = time.time()
        if request.json_input is not None:
            # 初始化算法
            self.service_config = json.loads(request.json_input)
            self.inf_model = get_manager_v1(self.service_config)
            self.inf_model.cfg = self.service_config
            # logger.info(f'config content:{configs}')
        t1 = time.time()
        reply = dto_pb2.AIReply(
            # 创建python客户端代码
            stream_id=request.stream_id,
            json_out=json.dumps({'status': True}, cls=NumpyEncoder),
            send_time=(t1 - t0)
        )
        logger.info(f'configuration running time:{t1 - t0}')
        return reply


class InferenceProcess(mp.Process):
    MAX_MESSAGE_LENGTH = 128 * 1024 * 1024

    def __init__(self, grpc_port, device):
        super(InferenceProcess, self).__init__()
        self.grpc_port = grpc_port
        self.device = device
        self.engine = None

    def run(self):
        self.serve()

    def serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', self.MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', self.MAX_MESSAGE_LENGTH)])
        dto_pb2_grpc.add_CVServiceServicer_to_server(GRPCServer(self.device), server)
        # 使用本地ip和对应端口nvi
        server.add_insecure_port(f'[::]:{self.grpc_port}')
        server.start()
        logger.info(f'process id {os.getpid()}')
        server.wait_for_termination()


# 启动gRPC服务
if __name__ == '__main__':
    # MAX_MESSAGE_LENGTH = 128 * 1024 * 1024
    # device = 1
    # grpc_port = '12345'
    # server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
    #     ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
    #     ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    # servicer = GRPCServer(device)
    # dto_pb2_grpc.add_CVServiceServicer_to_server(servicer, server)
    # # 使用本地ip和对应端口
    # server.add_insecure_port(f'[::]:{grpc_port}')
    # server.start()
    # server.wait_for_termination()

    try:
        mp.set_start_method('spawn')
    except Exception as e:
        logger.error(e)
        sys.exit(-1)
    # grpc ports list
    ports = ['12340']
    devices = [1]
    # proes = []
    for port, device in zip(ports, devices):
        p = InferenceProcess(grpc_port=port, device=device)
        p.start()
        # proes.append(proes)
