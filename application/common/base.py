#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/10月/11 12:24  
# File: base.py  
# Project: waterlevel_detect    
# Software: PyCharm   
"""
Function:
    
"""


class BaseDetection(object):
    def __init__(self):
        super(BaseDetection, self).__init__()

    def detect(self, **kwargs):
        raise NotImplementedError
