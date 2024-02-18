#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/6月/14 15:52  
# File: inference_engine.py  
# Project: TRT_inference    
# Software: PyCharm   
"""
Function: pytorch preprocess + tensorrt inference

"""
import time
import traceback

import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit as cudainit
import tensorrt as trt
import cv2
import torch
import torch.nn.functional as F
import sys


sys.path.append('../model/inference/')
from .general1 import opt_non_max_suppression
# from ..utils.general import opt_non_max_suppression
import logging

logger = logging.getLogger(__name__)
DEBUG = True

pre_, run, post = 0, 0, 0


class EngineInstance(object):
    def __init__(self, engine_file,
                 binding_names,
                 names=None,
                 device=0,
                 half=True,

                 ):
        """
        :param engine_file: trt模型文件路径
        :param binding_names: 输入输出节点名称，必须与输入输出节点索引对应
        :param device: gpu序列号
        :param names: 类别名列表
        :param half: 输入输出是否使用半精度
        """

        assert os.path.exists(engine_file), f'cannot found file \'{engine_file}\''
        self.device = device
        self.names = names
        self.half = half
        self.logger = trt.Logger(trt.Logger.INFO)
        self._cuda_init(self.device)
        # 加载推理引擎
        self.engine, self.context = self._load_engine(engine_file)

        # 创建处理流，分配显存
        self.input_bindings, self.output_bindings, \
        self.input_tensors, self.output_tensors, self.stream = \
            self._momery_allocate(self.engine, binding_names, self.device)

    def _cuda_init(self, device_id):
        cuda.init()
        dev = cuda.Device(device_id)
        ctx = dev.make_context()

        return ctx

    def _load_engine(self, engine_file):
        """
        Deserialize the TensorRT engine from specified plan file
        :param engine_file:
        :return:trt inference engine instance
        """
        with torch.cuda.device(self.device):
            with open(engine_file, mode='rb') as f, trt.Runtime(self.logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                logger.info('engine init finish....')
                context = engine.create_execution_context()
                logger.info('context init finish....')
                return engine, context

    def _momery_allocate(self, engine, binding_names, device):
        """
        存储空间分配
        :param engine:inference engine
        :param binding_names:list of binding tensor names
        :param device: gpu device index
        """

        # initialize cuda
        max_batch_size = engine.max_batch_size
        stream = cuda.Stream()  # self.device
        input_bindings = []
        output_shapes, output_bindings = [], []

        input_tensors, output_tensors = [], []
        for binding_name in binding_names:
            binding_shape = engine.get_binding_shape(engine[binding_name])
            binding_shape[0] = max_batch_size
            # print('binding shape', binding_shape)
            torch.rand(*binding_shape).cuda(device)

            # print(tmp)
            if engine.binding_is_input(engine[binding_name]):
                input_bindings.append(engine[binding_name])
                input_tensor = torch.cuda.ByteTensor(binding_shape[0], binding_shape[2], binding_shape[3],
                                                     binding_shape[1]).zero_()
                input_tensors.append(input_tensor)
            else:
                output_shapes.append(binding_shape)
                output_bindings.append(engine[binding_name])
                output_tensor = torch.cuda.FloatTensor(*binding_shape).zero_().half() if self.half \
                    else torch.cuda.FloatTensor(*binding_shape).zero_()
                output_tensors.append(output_tensor)
        # print('allocate output:', output_tensors[0].device)
        return input_bindings, output_bindings, input_tensors, output_tensors, stream

    def deallocate(self):
        """
        memory release
        :return:
        """
        # for input_device_mem in self.input_device_mems:
        #     input_device_mem.free()
        # for output_device_mem in self.output_device_mems:
        #     output_device_mem.free()
        # for output_host_mem in self.output_host_mems:
        #     output_host_mem.base.free()
        torch.cuda.empty_cache()
        # self.ctx.pop()
        # clear_context_caches()

    def inference_v2(self, *args, **kwargs):
        """
        inference interface
        """

        raise NotImplementedError()

    def test_inference_cuda(self, *args, **kwargs):
        # # 预处理
        input_imgs_b = args[0]
        logger.warning(len(input_imgs_b[0]))
        imgw, imgh = kwargs['imgw'], kwargs['imgh']
        stream = cuda.Stream()
        input_datas = np.array(bytearray(input_imgs_b[0])).reshape(8, imgh, imgw, 3)
        input_datas = input_datas[..., ::-1].transpose(0, 3, 1, 2)
        input_datas = input_datas.astype(np.float16)
        input_datas = np.ascontiguousarray(input_datas)
        input_datas = input_datas / 255.0
        self.context.set_optimization_profile_async(0, stream.handle)
        self.context.set_binding_shape(self.engine.get_binding_index('images'),
                                       (8, 3, imgh, imgw))
        input_memory = cuda.mem_alloc(input_datas.nbytes)
        out_type = trt.nptype(self.engine.get_binding_dtype(self.engine['output']))
        logger.info(f'output_type:{out_type}')
        output_buffer = cuda.pagelocked_empty(8 * 6, out_type)
        output_memory = cuda.mem_alloc(output_buffer.nbytes)

        cuda.memcpy_htod_async(input_memory, input_datas)
        self.context.execute_async_v2(bindings=[int(input_memory), int(output_memory)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        stream.synchronize()
        logger.info(output_buffer)
        return output_buffer

    def pre_process(self, batch_image, imgw, imgh, batch_image_ts=None, func=None, args=[]):

        batch_size = int(len(batch_image) / (imgw * imgh * 3))
        if batch_image_ts is not None:
            cuda.memcpy_htod_async(int(batch_image_ts.storage().data_ptr()), batch_image)
            batch_image_ts = batch_image_ts[:batch_size]
        else:
            batch_image_ts = torch.ByteTensor(torch.ByteStorage.from_buffer(batch_image)).to(self.device)
            batch_image_ts = batch_image_ts.view(batch_size, 3, imgh, imgw)
        if func is not None:
            return func(batch_image_ts, self.half, *args)
        else:
            return batch_image_ts

    def post_process(self, *args, **kwargs):
        raise NotImplementedError()

    def run(self, input_datas):
        """
        通用trt模型推理方法,可适应多输入多输出模型
        :param input_datas:list, input_datas中的每个元素为torch.Tensor类型,
        :return:list,输入列表中的每个元素为torch.Tensor类型
        """
        bs = input_datas[0].shape[0]
        bindings = []
        self.context.set_optimization_profile_async(0, self.stream.handle)
        # binding input
        for i, (binding, input_data) in enumerate(zip(self.input_bindings, input_datas)):
            self.context.set_binding_shape(binding,
                                           input_data.shape)
            bindings.append(input_data.storage().data_ptr())
        # binding output
        for i, output_tensor in enumerate(self.output_tensors):
            bindings.append(output_tensor.storage().data_ptr())

        self.context.execute_async_v2(bindings=bindings
                                      , stream_handle=self.stream.handle)  #
        self.stream.synchronize()
        return [output_tensor[:bs] for output_tensor in self.output_tensors]


class DetectEngine(EngineInstance):
    def __init__(self, engine_file,
                 binding_names,
                 names=[],
                 device=0,
                 half=True
                 ):
        """
        :param engine_file: trt模型文件路径
        :param binding_names: 输入输出节点名称，必须与输入输出节点索引对应
        :param device: gpu序列号
        :param names: 类别名列表
        :param half: 输入输出是否使用半精度
        """

        super(DetectEngine, self).__init__(engine_file, binding_names,
                                           device=device, half=half)
        self.names = names

    def post_process(self, pred, input_ts_shape=None, src_img_shape=None, names=[]):

        out_pred = opt_non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, agnostic=True)
        ret_pred = []
        if out_pred is not None:
            # t0 = time.time()
            out_pred[..., (0, 2)].clamp_(0, src_img_shape[1] - 1)
            out_pred[..., (1, 3)].clamp_(0, src_img_shape[0] - 1)
            out_pred = out_pred.cpu()  # [n,6+1]
            for i in range(pred.shape[0]):
                batch_ind = out_pred[:, -1] == i
                # det = out_pred[]
                det = out_pred[batch_ind]

                if torch.sum(batch_ind) > 0:
                    xyxy_label = det[..., :4].int()
                    confs = det[..., 4].tolist()
                    cls_ids = det[..., 5].int().tolist()
                    cls = []
                    for j, (conf, c) in enumerate(zip(confs, cls_ids)):
                        # xyxy_label[i].append(names[c.item()])
                        # xyxy_label[i].append(round(conf.item(), 2))
                        # xyxy_label[j].extend([names[c], round(conf, 2)])
                        cls.append(names[c])
                    ret_pred.append([xyxy_label, cls, confs])
                else:
                    ret_pred.append([])
        return ret_pred

    def inference_v2(self, input_imgs, imgw, imgh, ret_box_ts=False):
        """
        模型推理
        :param input_imgs:批量图像字节
        :param imgw:图像宽
        :param imgh:图像高
        :param output_box_ts:是否输出目标截取框张量
        :return:list ,list 长度为输入的batch size大小,list中的每个元素包含三个子元素：
        ([[batch_i中目标的坐标张量，batch_i中目标的类别列表,batch_i中目标的置信度张量],...]
        optional[输出bbox在输入张量中的截取子图],
        optional[截取子图对应的batch索引])
        """
        pre_, run, post = 0, 0, 0
        t0 = time.time()

        # 预处理
        def pre_func(imgs_ts, half, *other_args):
            batch_image_ts = imgs_ts[..., (2, 1, 0)].permute(0, 3, 1, 2)
            batch_image_ts = batch_image_ts.contiguous()
            batch_image_ts = batch_image_ts.half() if half else batch_image_ts.float()
            batch_image_ts /= 255.0
            return batch_image_ts

        input_datas = [self.pre_process(input_imgs[0], imgw, imgh,
                                        batch_image_ts=self.input_tensors[0], func=pre_func)]
        pre_ += (time.time() - t0)

        # logger.info(f'det pre:{pre_}')
        t0 = time.time()
        outputs = self.run(input_datas)
        run += (time.time() - t0)
        # logger.info(f'det inference:{run}')
        t0 = time.time()
        xyxy_cls_conf = self.post_process(outputs[0], input_ts_shape=input_datas[0].shape,
                                          src_img_shape=(imgh, imgw), names=self.names)
        # batch_inds = []
        crop_inp_ts_l = []
        # 是否输出检测目标对应的输入张量子图
        if ret_box_ts:
            for i, batch_i_boxes in enumerate(xyxy_cls_conf):
                # batch_inds.extend([i] * len(batch_i_boxes))
                crop_inp_tss = []
                for box in batch_i_boxes[0]:
                    crop_inp_ts = input_datas[0][i, :, box[1]:box[3], box[0]:box[2]]
                    crop_inp_tss.append(crop_inp_ts)
                crop_inp_ts_l.append(crop_inp_tss)
        post += (time.time() - t0)
        # logger.info(f'det post :{post}')
        return (xyxy_cls_conf, crop_inp_ts_l) if ret_box_ts else xyxy_cls_conf


class FeatureExtractEngine(EngineInstance):
    def __init__(self, engine_file,
                 binding_names,
                 names=[],
                 device=0,
                 half=True
                 ):
        super(FeatureExtractEngine, self).__init__(engine_file,
                                                   binding_names,
                                                   device=device,
                                                   half=half)
        self.names = names

    def post_process(self, pred, names=[]):
        # logger.info(f'pred:{pred.shape}')
        # pred = torch.argmax(pred, dim=1)
        # pred = pred.cpu().int().tolist()
        # # logger.info(f'pred:{pred}')
        # out_lb = []
        # for p in pred:
        #     label = names[p]
        #     # logger.info(f'prediction:{names[p]}')
        #     out_lb.append(label)
        # logger.info(f'feature:{pred.shape}')
        # logger.info(f'norm? {torch.sum(torch.square(pred), dim=1)}')
        return pred.type(torch.float16).cpu().numpy()  # .cpu()
        # return torch.clone(pred)

    def inference_v2(self, input_imgs, imgw, imgh, pre_proc=True):
        if input_imgs is None or not len(input_imgs):
            # logger.info('empty inputs..')
            return [],

        if pre_proc:
            def img_prefunc(imgs_ts, half, *args):
                batch_image_ts = imgs_ts[..., (2, 1, 0)].permute(0, 3, 1, 2)
                batch_image_ts = batch_image_ts.contiguous()
                batch_image_ts /= 255.0
                batch_image_ts = batch_image_ts.half() if half else batch_image_ts.float()
                return batch_image_ts

            input_datas = [
                self.pre_process(input_imgs[0], imgw, imgh, batch_image_ts=self.input_tensors[0], func=img_prefunc, )]
        else:
            if isinstance(input_imgs, torch.Tensor):
                input_datas = [input_imgs]
            elif isinstance(input_imgs, list):
                input_imgs_l = []
                for input_img in input_imgs:
                    # mean = torch.tensor([0.406, 0.456, 0.485]).resize(3, 1, 1).to(self.device)
                    # std = torch.tensor([0.225, 0.224, 0.229]).resize(3, 1, 1).to(self.device)
                    # input_img = (input_img - mean) / std
                    input_img = input_img.unsqueeze(0)
                    input_imgs_l.append(F.interpolate(input_img, (imgh, imgw)))
                input_datas = [torch.cat(input_imgs_l, dim=0).half() if self.half else
                               torch.cat(input_imgs_l, dim=0).float()]
                # logger.info(f'input_datas is contiguous: {input_datas[0].is_contiguous()}')
                # logger.info(f'cls input_datas:{len(input_datas[0])},{input_datas[0].shape}')
            else:
                traceback.print_exc()
                raise TypeError('error accured where building input...')
        outputs = self.run(input_datas)
        res = self.post_process(outputs[0], self.names)
        return res


def run_time(func, args, func_name='func_name'):
    if DEBUG:
        t0 = time.time()
    ret = func(*args)
    if DEBUG:
        print(f'{func_name} runtime :', time.time() - t0)
    return ret


if __name__ == '__main__':
    import pycuda.autoinit as cudainit

    batch_size = 8
    size = (640, 640)
    img_path = './test_images/test3.png'
    det_engine = DetectEngine('./model/engine_file/best_dynamic_batch.engine',
                              ['images', 'output'],
                              names=['car', 'suv', 'truck',
                                     'light truck', 'van',
                                     'chemicals vehicle', 'bus'],
                              half=True,
                              device=0)
    # cls_engine = FeatureExtractEngine('./model/engine_file/reid_fp32.engine',
    #                                   ['images', 'output'],
    #                                   # names=['car', 'suv', 'truck', 'light truck', 'van', 'chemicals vehicle'],
    #                                   half=False,
    #                                   device=device
    #                                   )
    t0 = time.time()
    input_datas = []
    for i in range(batch_size):
        # input_img = pre_process(img_path, (size[-1], size[0]), dtype=dtype)
        input_img = cv2.imread(img_path)
        input_img = cv2.resize(input_img, (size[-1], size[0]))
        input_datas.append(input_img)
    input_datas = np.stack(input_datas)
    input_datas = input_datas.tobytes()
    for _ in range(10):
        t0 = time.time()
        xyxy_cls_conf, boxes_ts_l = \
            det_engine.inference_v2([input_datas], imgw=size[0], imgh=size[1], ret_box_ts=True)
        t1 = time.time()
        # cls_res = cls_engine.inference_v2(boxes_ts_l[0], 64, 64, pre_proc=False)[0]
        t2 = time.time()
        # logger.info(f'result len {len(cls_res)},{cls_res.shape}')
        # logger.info(xyxy_cls_conf)
        logger.info(f'total cost:{t2 - t0},det cost:{t1 - t0},cls cost:{t2 - t1}')
        logger.info('--------------------')
    det_engine.deallocate()
    # cls_engine.deallocate()
