# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
import sklearn
from scipy.optimize import minimize
import numpy as np
import time
import torch
from functorch import grad
from functools import partial
import traceback

class Scaler(ABC):
    """
    Base class for scalers
    """

    def __init__(self, time_range=[1,5], feature_scale=1, rescale_features=True):
        self.time_scaler = sklearn.preprocessing.MinMaxScaler()
        self.traj_scale  = None
        self.feature_scale = feature_scale
        self.time_scale = (time_range[1]-time_range[0])
        self.time_shift = time_range[0]
        self.rescale_features = rescale_features

    def fit(self, time, trajectory):
        self.time_scaler.fit(time.reshape(-1,1))
        self.traj_scale = trajectory[0]
        self.traj_scale[self.traj_scale==0] = 1

    def transform(self, time, trajectory):
        scaled_time = self.time_scaler.transform(time.reshape(-1,1))*self.time_scale+self.time_shift
        if self.rescale_features: 
            scaled_traj = self.feature_scale * trajectory/(self.traj_scale.reshape(1,-1))
        else: scaled_traj = trajectory
        return scaled_time[:,0], scaled_traj
        
    def fit_transform(self, time, trajectory):
        self.fit(time, trajectory)
        scaled_time, scaled_trajectory = self.transform(time, trajectory)
        return scaled_time, scaled_trajectory
    
    def get_params(self):
        scale = self.feature_scale/self.traj_scale
        val_min, val_max = self.time_scaler.data_min_[0], self.time_scaler.data_max_[0]
        a_t, b_t = self.time_scale/(val_max-val_min), -self.time_scale*val_min/(val_max-val_min)+self.time_shift
        return (a_t, b_t, scale)

    def rescale_function(self, env, tree, a_t, b_t, scale):
        nodes = tree.prefix().split("|")
        if len(nodes)>len(scale): 
            return tree
        for dim, node in enumerate(nodes):
            nodes[dim] = f"mul,{1/scale[dim]},"+nodes[dim].lstrip(',').rstrip(',')
        prefix = ",|,".join(nodes).split(",")
        idx = 0
        while idx < len(prefix):
            if (prefix[idx].startswith("x_") and self.rescale_features) or prefix[idx] == "t":
                if prefix[idx].startswith("x_"):
                    dim = int(prefix[idx][-1])
                    if dim>=len(scale): 
                        return tree
                    a = str(scale[dim])
                    prefix_to_add = ["mul", a, prefix[idx]]
                else:
                    a, b = str(a_t), str(b_t)
                    prefix_to_add = ["add", b, "mul", a, prefix[idx]]
                prefix = prefix[:idx] + prefix_to_add + prefix[min(idx + 1, len(prefix)):]
                idx += len(prefix_to_add)
            else:
                idx+=1
                continue
        rescaled_tree = env.word_to_infix(prefix, is_float=False, str_array=False)
        return rescaled_tree