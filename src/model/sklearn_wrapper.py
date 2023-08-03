# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import copy
import time as _time
import numpy as np
import torch
from collections import defaultdict
from symbolicregression.metrics import compute_metrics
from sklearn.base import BaseEstimator
import symbolicregression.model.utils_wrapper as utils_wrapper
from symbolicregression.model.mixins import PredictionIntegrationMixin
import traceback
from sklearn import feature_selection 
from symbolicregression.envs.generators import integrate_ode
from symbolicregression.envs.utils import *
import warnings
import scipy

def exchange_node_values(tree, dico):
    new_tree = copy.deepcopy(tree)
    for (old, new) in dico.items():
        new_tree.replace_node_value(old, new)
    return new_tree

class SymbolicTransformerRegressor(BaseEstimator, PredictionIntegrationMixin):

    def __init__(self,
                model=None,
                from_pretrained=False,
                max_input_points=10000,
                rescale=True,
                params=None,
                model_kwargs={},
                ):

        self.max_input_points = max_input_points
        self.model = model
        self.rescale = rescale
        self.params = params
        if from_pretrained:
            self.load_pretrained()
        for kwarg, val in model_kwargs.items():
            setattr(self.model, kwarg, val)

    def load_pretrained(self):
        import gdown
        model_path = "odeformer.pt" 
        if not os.path.exists(model_path):
            url = "https://drive.google.com/uc?id=18CwlutaFF_tAOObsIukrKVZMPmsjwNwF"
            gdown.download(url, model_path, quiet=False)
        model = torch.load(model_path)
        print("Loaded pretrained model")
        self.model = model

    def set_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self, arg), "{} arg does not exist".format(arg)
            setattr(self, arg, val)

    def fit(
        self,
        times,
        trajectories,
        sort_candidates=True,
        sort_metric="snmse",
        rescale=None,
        verbose=False,
    ):

        if not rescale: rescale = self.rescale
        self.rescale = rescale

        assert not (self.model.average_trajectories and self.rescale), "Cannot average trajectories and rescale at the same time"
        #assert not (self.params is None and self.rescale), "Need to know the time and feature range to rescale to"
        if not self.params:
            feature_scale = 1
            time_range = 10
        else:
            feature_scale = self.params.init_scale
            time_range = self.params.time_range

        if not isinstance(times, list):
            times = [times]
            trajectories = [trajectories]
        n_datasets = len(times)
        
        # rescale time and features
        scaler = utils_wrapper.Scaler(time_range=[1, time_range], feature_scale=feature_scale) if self.rescale else None 
        scale_params = {}
        if scaler is not None:
            scaled_times = []
            scaled_trajectories = []
            for i, (time, trajectory) in enumerate(zip(times, trajectories)):
                scaled_time, scaled_trajectory = scaler.fit_transform(time, trajectory)
                scaled_times.append(scaled_time)
                scaled_trajectories.append(scaled_trajectory)
                scale_params[i]=scaler.get_params()
        else:
            scaled_times = times.copy()
            scaled_trajectories = trajectories.copy()

        # permute trajectories so that when bagging the model doesn't get chunks
        for i, (scaled_time, scaled_trajectory) in enumerate(zip(scaled_times, scaled_trajectories)):
            permutation = np.random.permutation(len(scaled_time))
            scaled_times[i] = scaled_time[permutation]
            scaled_trajectories[i] = scaled_trajectory[permutation]

        # split into bags of size max_input_points
        inputs, inputs_ids = [], []
        for seq_id in range(len(scaled_times)):
            for seq_l in range(len(scaled_times[seq_id])):
                y_seq = scaled_trajectories[seq_id]
                if len(y_seq.shape)==1:
                    y_seq = np.expand_dims(y_seq,-1)
                if seq_l%self.max_input_points == 0:
                    inputs.append([])
                    inputs_ids.append(seq_id)
                inputs[-1].append([scaled_times[seq_id][seq_l], y_seq[seq_l]])
            # inputs.append([])
            # inputs_ids.append(seq_id)

        # Forward transformer
        forward_time=_time.time()
        outputs = self.model(inputs)  ##Forward transformer: returns predicted functions
        if verbose: print("Finished forward in {} secs".format(_time.time()-forward_time))

        all_candidates = defaultdict(list)
        assert len(inputs) == len(outputs), "Problem with inputs and outputs"
        for i in range(len(inputs)):
            input_id = inputs_ids[i]
            candidates = outputs[i]
            if not candidates: all_candidates[input_id].append(None)
            for candidate in candidates:
                if scaler is not None:
                    candidate = scaler.rescale_function(self.model.env, candidate, *scale_params[input_id])   
                    candidate = self.model.env.simplifier.simplify_tree(candidate)
                all_candidates[input_id].append(candidate)
        #assert len(all_candidates.keys())==n_datasets
    
        if sort_candidates:
            for input_id in all_candidates.keys():
                all_candidates[input_id] = self.sort_candidates(times[input_id], trajectories[input_id], all_candidates[input_id], metric=sort_metric, verbose=verbose)

        return all_candidates

    @torch.no_grad()
    def evaluate_tree(self, tree, times, trajectory, metric):
        earliest = np.argmin(times)
        try: pred_trajectory = self.integrate_prediction(times, trajectory[earliest], prediction=tree)
        except: return np.nan
        metrics = compute_metrics(pred_trajectory, trajectory, predicted_tree=tree, metrics=metric)
        return metrics[metric][0]

    @torch.no_grad()
    def sort_candidates(self, times, trajectory, candidates, metric="snmse", verbose=False):
        if "r2" in metric: 
            descending = True
        else: 
            descending = False
        scores = []
        for candidate in candidates:
            score = self.evaluate_tree(candidate, times, trajectory, metric)
            if math.isnan(score): 
                score = -np.infty if descending else np.infty
            scores.append(score)
        sorted_idx = np.argsort(scores)  

        if descending: sorted_idx= list(reversed(sorted_idx))
        candidates = [candidates[i] for i in sorted_idx]

        scores = [scores[i] for i in sorted_idx]

        if verbose: 
            print(scores, candidates)
            for score, candidate in zip(scores, candidates):
                print(f'{score}:{candidate}')

        return candidates
