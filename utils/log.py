#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/7月/11 11:40  
# File: log.py
# Project: TRT_inference    
# Software: PyCharm   
"""
Function:
    
"""
import logging

logging.basicConfig(level=logging.DEBUG, format=r'[%(asctime)s]-[%(filename)s]-%(lineno)d %(levelname)s: %(message)s')

logger = logging.getLogger()
