#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 张晏玮 
# Gitlab: ECCOM  
# Creat Time:  2022/7月/12 11:01  
# File: container.py  
# Project: TRT_inference    
# Software: PyCharm   
"""
Function:
    
"""


class FeedBackInfo(object):
    def __init__(self, alg_id=None, alg_name=None, date=None, msg='', trace_back='', err_code=0, data=[]):
        self.alg_id = alg_id
        self.alg_name = alg_name
        self.date = date
        self.msg = msg
        self.trace_back = trace_back
        self.err_code = err_code
        self.data = data

    def get(self):
        return self.__dict__
