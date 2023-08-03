# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math, time, copy
import numpy as np
import torch
from collections import defaultdict
from symbolicregression.metrics import compute_metrics
from sklearn.base import BaseEstimator
import symbolicregression.model.utils_wrapper as utils_wrapper
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

class SymbolicTransformerRegressor(BaseEstimator):

    def __init__(self,
                model=None,
                max_input_points=10000,
                max_number_bags=-1,
                stop_refinement_after=1,
                n_trees_to_refine=1,
                rescale=False,
                average_trajectories=False,
                params=None
                ):

        self.max_input_points = max_input_points
        self.max_number_bags = max_number_bags
        self.model = model
        self.stop_refinement_after = stop_refinement_after
        self.n_trees_to_refine = n_trees_to_refine
        self.rescale = rescale
        self.params = params
        self.average_trajectories = average_trajectories
        self.model.average_trajectories = average_trajectories

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
        average_trajectories=None,
        rescale=None,
        verbose=False,
    ):
        self.start_fit = time.time()

        if not average_trajectories: average_trajectories = self.average_trajectories
        self.model.average_trajectories = average_trajectories
        if not rescale: rescale = self.rescale
        self.rescale = rescale

        assert not (self.average_trajectories and self.rescale), "Cannot average trajectories and rescale at the same time"
        assert not (self.params is None and self.rescale), "Need to know the time and feature range to rescale to"

        if not isinstance(times, list):
            times = [times]
            trajectories = [trajectories]
        n_datasets = len(times)

        # take finite differences
        if self.params:
            if self.params.differentiate:
                for i in range(len(times)):
                    trajectories[i], times[i] = np.diff(trajectories[i], axis=0), times[i][1:]
        
        scaler = utils_wrapper.Scaler(time_range=[1, self.params.time_range], feature_scale=self.params.init_scale) if self.rescale else None 
        scale_params = {}
        if scaler is not None:
            scaled_times = []
            scaled_trajectories = []
            for i, (time_, trajectory) in enumerate(zip(times, trajectories)):
                scaled_time, scaled_trajectory = scaler.fit_transform(time_, trajectory)
                scaled_times.append(scaled_time)
                scaled_trajectories.append(scaled_trajectory)
                scale_params[i]=scaler.get_params()
        else:
            scaled_times = times
            scaled_trajectories = trajectories

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

        if self.max_number_bags>0:
            inputs = inputs[:self.max_number_bags]
            inputs_ids = inputs_ids[:self.max_number_bags]

        # Forward transformer
        forward_time=time.time()
        outputs = self.model(inputs)  ##Forward transformer: returns predicted functions
        if verbose: print("Finished forward in {} secs".format(time.time()-forward_time))

        all_candidates = defaultdict(list)
        assert len(inputs) == len(outputs), "Problem with inputs and outputs"
        for i in range(len(inputs)):
            input_id = inputs_ids[i]
            candidates = outputs[i]
            if not candidates: all_candidates[input_id].append(None)
            for candidate in candidates:
                if scaler is not None:
                    candidate = scaler.rescale_function(self.model.env, candidate, *scale_params[input_id])                    
                all_candidates[input_id].append(candidate)
        #assert len(all_candidates.keys())==n_datasets

        if sort_candidates:
            for input_id in all_candidates.keys():
                all_candidates[input_id] = self.sort_candidates(scaled_times[input_id], scaled_trajectories[input_id], all_candidates[input_id], metric=sort_metric)
            
        self.trees = all_candidates

        return all_candidates

    @torch.no_grad()
    def evaluate_tree(self, tree, times, trajectory, metric):
        earliest = np.argmin(times)
        pred_trajectory = self.predict(times, trajectory[earliest], tree=tree)
        metrics = compute_metrics(pred_trajectory, trajectory, predicted_tree=tree, metrics=metric)
        return metrics[metric][0]

    @torch.no_grad()
    def sort_candidates(self, times, trajectory, candidates, metric="snmse"):
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
        if descending: sorted_idx= reversed(sorted_idx)
        candidates = [candidates[i] for i in sorted_idx]

        return candidates

    @torch.no_grad()
    def predict(self, times, y0, tree=None):   

        if tree is None:
            return None

        # integrate the ODE
        if self.params:
            ode_integrator = self.params.ode_integrator
        else:
            ode_integrator = "solve_ivp"
        trajectory = integrate_ode(y0, times, tree, ode_integrator=ode_integrator)
        
        return trajectory