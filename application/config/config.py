#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2021/11月/08 16:56  
# File: config.py  
# Project: smart_service_area    
# Software: PyCharm   
"""
Function:
    
"""
import json


def parse_config(cfg_file):
    """
    解析json文件
    :param cfg_file:
    :return:
    """
    if not cfg_file.endswith('json'):
        raise ValueError(f'\'cfg_file\' must be a json file path')
    with open(cfg_file, mode='r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg

