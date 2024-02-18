#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/2月/22 15:42  
# File: cuda_gpu_man.py  
# Project: cuda_gpu_man    
# Software: PyCharm   
"""
Function:
    
"""
import torch
import pycuda.driver as cuda
from pycuda.tools import clear_context_caches
from pycuda.tools import DeviceData

def get_gpu_memory_info():
    """
    torch cuda context will build when executing the first cuda operation,
    it will consumes a certain amount of GPU memory,
    and how much GPU memory spend on it depends on torch version and cuda version

    Returns:(list,list,list)
        (list of total memory of all available devices,
        list of used memory of all available devices,
        list of free memory of all available devices)
    """
    if torch.cuda.is_available():
        # if device is None:
        #     device = torch.device('cuda:0')
        # else:
        #     device = torch.device(device) if isinstance(device, int) or isinstance(device, str) else device
        #
        # torch.FloatTensor([1.0]).to(device)
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, \
            nvmlDeviceGetCount
        nvmlInit()
        n = nvmlDeviceGetCount()
        totals = []
        useds = []
        frees = []
        for ni in range(n):
            handle = nvmlDeviceGetHandleByIndex(ni)
            info = nvmlDeviceGetMemoryInfo(handle)
            totals.append(info.total)
            useds.append(info.used)
            frees.append(info.free)
        # CUDA out of memory
        nvmlShutdown()
        return totals, useds, frees

    else:
        return None
        # raise RuntimeError('no device available')


def get_proc_gpu_memory_reserved(device, func, args=[]):
    """
    calculate GPU memory reseved to func
    Args:
        device: str or int
        func:callable object
        args: arguments of func

    Returns:int
        gpu memory reserved to func
    """
    func(*args)
    reserved = torch.cuda.memory_reserved(device)
    return reserved


def get_max_memory_free_device():
    """
    get id of device ,which hold the maximum memory,
    return None if no gpu device available
    Returns:
            device id
    """
    info = get_gpu_memory_info()
    if info is None:
        return info
    totals, useds, frees = info
    max_free = 0
    max_free_ind = -1
    for i, free in enumerate(frees):
        if free > max_free:
            max_free = free
            max_free_ind = i
    return None if max_free_ind == -1 else max_free_ind


def device_init(device_id):
    cuda.init()
    dev = cuda.Device(device_id)
    ctx = dev.make_context()
    return ctx, dev
