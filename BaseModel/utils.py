#!/user/bin/env python
# coding=utf-8
"""
@file: utils.py
@author: zwt
@time: 2020-10-31 11:35
@desc: 
"""
import pickle

with open('../data/x_seq.pkl', 'rb') as f:
    x_seq = pickle.load(f)
    print(x_seq)