from collections import defaultdict
import copy
import numpy as np
import torch
from boolformer.metrics import compute_metrics
import torch.nn.functional as F
import sklearn
from sklearn.base import BaseEstimator
import pickle
from abc import ABC, abstractmethod
import numpy as np
import math
from boolformer.envs.utils import *
from boolformer.envs import build_env
from boolformer.model import build_modules
from boolformer.model.utils_wrapper import *
import re
import time

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class SymbolicTransformerRegressor(BaseEstimator):

    def __init__(self,
                env=None,
                embedder=None,
                encoder=None,
                decoder=None,
                beam_length_penalty=1,
                beam_size=1,
                max_generated_output_len=200,
                validation_metrics="r2",
                beam_selection_metrics=1,
                beam_early_stopping=True,
                max_input_points=10000,
                scale_type=None,
                refinements_types="method=BFGS_batchsize=256_metric=/_mse",
                use_nevergrad=False
                ):

        self.beam_length_penalty = beam_length_penalty
        self.beam_size = beam_size
        self.max_generated_output_len = max_generated_output_len
        self.validation_metrics = validation_metrics
        self.beam_selection_metrics = beam_selection_metrics
        self.beam_early_stopping = beam_early_stopping
        self.max_input_points = max_input_points
        self.scale_type = scale_type
        self.env = env
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.refinements_types = refinements_types
        self.use_nevergrad = use_nevergrad

    def set_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self, arg), "{} arg does not exist".format(arg)
            setattr(self, arg, val)

    def fit(
        self,
        X,
        Y,
        required_operators=None,
        verbose=False,
       # threshold_constant=0.,
    ):

        """
        max_input_points: length of each sequence paquet. -1 means no joint decoding
        """
        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        selection_metric = self.validation_metrics
        if not isinstance(X, list):
            X = [X]
            Y = [Y]
        n_inputs = len(X)

        if self.scale_type is None: scaler = None
        elif self.scale_type == "standard": scaler = StandardScaler()
        elif self.scale_type == "minmax": scaler = MinMaxScaler()
        else: raise NotImplementedError

        params = []
        if scaler is not None:
            scaled_X = []
            for x in X:
                scaled_X.append(scaler.fit_transform(x))
                params.append(scaler.get_params())
            unscaled_X = X
        else:
            scaled_X = X
            unscaled_X = X

        x_encoder, x_ids = [], []
        for seq_id in range(len(scaled_X)):
            if len(scaled_X[seq_id][0]) > env.generator.max_input_dimension: return self
            for seq_l in range(len(scaled_X[seq_id])):
                y_seq = Y[seq_id]
                if len(y_seq.shape)==1:
                    y_seq = np.expand_dims(y_seq,-1)
                if seq_l%self.max_input_points == 0:
                    x_encoder.append([])
                    x_ids.append(seq_id)
                x_encoder[-1].append([scaled_X[seq_id][seq_l], y_seq[seq_l]])

        generate_beam = True #self.beam_size!=1 TO MAKE SURE WE KEEP SAME CONDITIONS AS BEFORE
        B, T = len(x_encoder), max([len(xi) for xi in x_encoder])
        
        _generations=[[] for _ in range(n_inputs)]

        with torch.no_grad():
            for chunk in chunks(np.arange(B), int(10000/T)):
                x_ids_chunk = [x_ids[idx] for idx in chunk]
                x, x_len = embedder([x_encoder[idx] for idx in chunk])
                bs = x_len.shape[0]
                encoded = encoder("fwd", x=x, lengths=x_len, causal=False)

                if generate_beam:
                    _, _, generations = decoder.generate_beam(
                        encoded.transpose(0, 1),
                        x_len,
                        beam_size=self.beam_size,
                        length_penalty=self.beam_length_penalty,
                        max_len=self.max_generated_output_len,
                        early_stopping=self.beam_early_stopping,
                        group_ids=torch.arange(len(x_ids_chunk))   # torch.tensor(x_ids)
                        ) 
                    generations = [sorted([hyp for hyp in generations[i].hyp], key=lambda s: s[0], reverse=True) for i in range(bs)] 
                    generations = [[hyp.cpu().tolist() for (_, hyp) in generations[i]] for i in range(bs)]
                    
                else:
                    generations, x_len = decoder.generate(
                        encoded.transpose(0, 1),
                        x_len,
                        max_len=self.max_generated_output_len,
                        group_ids=torch.arange(len(x_ids_chunk))#torch.tensor(x_ids)
                        )  
                    generations = generations.transpose(0,1).unsqueeze(1).cpu().tolist()
                if verbose: print("finished forward of chunk:{}".format(chunk))
            
                for i in range(bs):
                    _generations[x_ids_chunk[i]].extend(generations[i])

        generations = _generations
        bs=len(generations)
        assert bs == n_inputs, "problems with shapes of BS and X"

        refinements_types = self.refinements_types.split(",") if self.refinements_types is not None else []
        refinement = ScipyRefinement()
        nevergrad_refinement = NevergradRefinement()

        self.tree = [None for i in range(bs)]
        self.XYs = [None for i in range(bs)]

        for i in range(bs):
            gens = []
    
            for j, hyp in enumerate(generations[i]):
                _hyp = hyp[1:]
                if generate_beam is False: _hyp=_hyp[:-1]
                predicted_tree = env.idx_to_infix(
                    _hyp, is_float=False, str_array=False
                )

                if (
                    predicted_tree is None
                    or max(
                        [
                            int(xi.split("_")[-1]) if xi.startswith("x_") else 0
                            for xi in predicted_tree.prefix().split(",")
                        ]
                    )
                    + 1
                    > scaled_X[i].shape[-1]
                ):
                    continue
            
                if required_operators is not None:
                    for op in required_operators.split(","):
                        if op not in predicted_tree.prefix().split(","):
                            continue

                rescaled_predicted_tree = predicted_tree
                if not env.params.use_skeleton:
                    if scaler is not None:
                        a, b = params[i]
                        rescaled_predicted_tree = scaler.rescale_function(env, predicted_tree, a, b)
                    else:
                        rescaled_predicted_tree = predicted_tree
                    gens.append(
                        {
                            "refinement_type": "NoRef",
                            "i": i,
                            "predicted_tree": rescaled_predicted_tree,
                        }
                    )

                if self.use_nevergrad:
                    print("using nevergrad")
                    skeleton_tree, constants = env.generator.function_to_skeleton(rescaled_predicted_tree)
                    constants = np.array(constants)

                    nevergrad_refinement.set_args(env=env, 
                                                  tree=skeleton_tree, 
                                                  X=unscaled_X[i],
                                                  y=Y[i]) 
                    refinement_args = {"budget": 1000, "metric":"r2"}

                    if env.params.use_skeleton:
                        refinement_args["random_init"]="True"
                    else:
                        refinement_args["random_init"]="False"

                    nevergrad_refinement.set_args(**refinement_args)
                    nevergrad_tree = nevergrad_refinement.go(constants)
                    if nevergrad_tree is None:
                        continue
                    gens.append(
                        {
                            "refinement_type": "Nevergrad",
                            "i": i,
                            "predicted_tree": nevergrad_tree,
                        }
                    )
            if verbose: print("start evaluation of candidates")
            
            if len(gens)==0: continue
            subset_idx = np.random.choice(Y[i].shape[0], size=1000)
            for g, gen in enumerate(gens):
                predicted_tree=gen["predicted_tree"]
                start=time.time()
                y_tilde = predicted_tree.val(unscaled_X[i][subset_idx])
                gen["metrics"] = compute_metrics({"true": [Y[i][subset_idx]], "predicted": [y_tilde], "predicted_tree": [predicted_tree]}, metrics=selection_metric)
                if verbose: print("{}/{}, val + metrics: {}, tree length: {}".format(g, len(gens), time.time()-start, len(predicted_tree.prefix().split(","))))
            gens = self.beam_selection_strategy(gens)[:1]
            tree_to_refine = gens[0]["predicted_tree"]
            if tree_to_refine is not None: 
                del gens[0]["metrics"]
            if verbose: print("finished best tree selection \ntree:{}\nlength:{}".format(tree_to_refine,len(tree_to_refine.prefix().split(","))))

            for refinement_type in refinements_types:
                refinement_args = [arg.replace("/","") for arg in re.split("(?<!/)_", refinement_type)]
                refinement_args = [refinement_arg.split("=") for refinement_arg in refinement_args]
                refinement_args = {k: v for [k,v] in refinement_args}

                if "expand" not in refinement_args:
                    expand=False
                else: 
                    expand=refinement_args["expand"]=="True"
                    del refinement_args["expand"]
                    
                simplifier_fns = [("simplify_expr", {})]
                if expand: 
                    simplifier_fns.extend([("expand_expr",{}), ("simplify_expr", {}) ]) 

                if verbose: print("start simplification:{} \n".format(refinement_type))

                refined_tree = env.simplifier.apply_fn(tree_to_refine, simplifier_fns)
                if len(refined_tree.prefix().split(","))>5*len(tree_to_refine.prefix().split(",")): 
                    refined_tree = tree_to_refine

                skeleton_tree, constants = env.generator.function_to_skeleton(refined_tree)
                refinement.set_args(env=env, 
                                    tree=skeleton_tree, 
                                    X=unscaled_X[i],
                                    y=Y[i])
                constants = np.array(constants)
                if env.params.use_skeleton and not self.use_nevergrad:
                    refinement_args["random_init"]="True"
                else:
                    if "random_init" not in refinement_args.keys(): 
                        refinement_args["random_init"]="False"
                refinement.set_args(**refinement_args)

                if constants.shape[0]>0:
                    if verbose: print("start refinement:{} \nskeleton:{}\n#constants:{}".format(refinement_type, skeleton_tree, len(constants)))
                    try:
                        refined_tree = refinement.go(constants)
                    except MyTimeoutError as e:
                        refined_tree = refinement.best_so_far

                if verbose: print("finished refinement:{} \ntree:{}\nlength:{}".format(refinement_type,refined_tree,len(refined_tree.prefix().split(","))))

                gens.append(
                    {
                        "refinement_type": refinement_type,
                        "i": i,
                        "predicted_tree": refined_tree,
                    }
                )
            if verbose:
                for gen in gens:
                    print(
                         gen["refinement_type"], gen["predicted_tree"]
                    )
                    
            if len(gens)>0:
                self.tree[i]=gens
                self.XYs[i]=(unscaled_X[i], Y[i])

        self.add_simplified_models()
        return self

    def beam_selection_strategy(self, gens, key="metrics"):
        if len(gens) == 0:
            return [{"predicted_tree": None}]
        if self.beam_selection_metrics>0:
            gens=sorted(gens, key=lambda x: tuple([v[0]*(1- 2*int(k.startswith("_"))) if not np.isnan(v[0]) else -np.infty
                        for k, v in x[key].items()]), reverse=True)
        return gens

    def __str__(self):
        if hasattr(self, "tree"):
            for tree_idx in range(len(self.tree)):
                for gen in self.tree[tree_idx]:
                    print(gen)
        return "Transformer"

    def add_simplified_models(self):
        simplifier_fns = [("simplify_expr", {}), ("round_expr", {"decimals": 3}), ("float_to_int_expr", {})]
        bs = len(self.tree)
        for i in range(bs):
            tree_i = self.tree[i]
            if tree_i is None: continue
            simplified_tree_i = []
            for tree_i_j in tree_i:
                predicted_tree=tree_i_j["predicted_tree"]
                refinement_type = tree_i_j["refinement_type"]
                try:
                    simplified_tree = self.env.simplifier.apply_fn(predicted_tree, simplifier_fns)
                    simplified_tree_i.append(
                        {
                            "refinement_type": refinement_type+"_simplified",
                            "i": i,
                            "predicted_tree": simplified_tree,
                        })
                except:
                    continue
            self.tree[i].extend(simplified_tree_i)
                
    def retrieve_tree(self, refinement_type=None, tree_idx=0):
        best_trees = []
        bs = len(self.tree)
        selection_metric = self.validation_metrics
        for i in range(bs):
            tree_i = self.tree[i]
            x, y = self.XYs[i]
            if tree_i is None:
                best_trees.append(None)
            else:
                for tree_i_j in tree_i:
                    predicted_tree=tree_i_j["predicted_tree"]
                    if "metrics" not in tree_i_j:
                        y_tilde = predicted_tree.val(x)
                        metrics = compute_metrics({"true": [y], "predicted": [y_tilde], "predicted_tree": [predicted_tree]}, metrics=selection_metric)
                        tree_i_j["metrics"]=metrics

                all_tree_i = copy.deepcopy(tree_i)
                if refinement_type is not None:
                    all_tree_i = list(filter(lambda gen: gen["refinement_type"]==refinement_type, all_tree_i))
                best_tree = self.beam_selection_strategy(all_tree_i, key="metrics")[0]["predicted_tree"]
                best_trees.append(best_tree)
        if tree_idx==-1:
            return best_trees
        else:
            return best_trees[tree_idx]

    def retrieve_refinements_types(self):
        refinements_types = self.refinements_types.split(",") if self.refinements_types is not None else []
        if not self.env.params.use_skeleton:
            refinements_types.append("NoRef")
        if self.use_nevergrad:
            refinements_types.append("Nevergrad")
        simplified_refinement_types = [ref+"_simplified" for ref in refinements_types]
        return refinements_types+simplified_refinement_types

    def predict(self, X, refinement_type=None, tree_idx=0, batch=False, squeeze=True, clean=False):
        res = []
        if batch:
            refined_tree = self.retrieve_tree(refinement_type, tree_idx = -1)
            for tree_idx in range(len(refined_tree)):
                if refined_tree[tree_idx] is None: 
                    res.append(None)
                else:   
                    try:
                        y = refined_tree[tree_idx].val(X[tree_idx])
                        if squeeze:
                            y = y[:, 0]
                        res.append(y)
                    except Exception as e:
                        #print(e)
                        res.append(None)
            return res
        else:
            refined_tree = self.retrieve_tree(refinement_type, tree_idx = tree_idx)
            if refined_tree is not None:
                y = refined_tree.val(X)
                if squeeze:
                    y = y[:, 0]
                return y
            else:
                return None
            
if __name__ == "__main__":
    from boolformer.model import build_modules
    from boolformer.envs import build_env
    from train import get_parser

    parser = get_parser()
    params = parser.parse_args()
    args = params
    env = build_env(args)
    env.rng = np.random.RandomState(0)

    modules = build_modules(env, args)

    X_to_fit = np.random.randn(10, 2)
    Y_to_fit = np.cos(X_to_fit[:, 0]) + 1.0

    for param in modules["embedder"].parameters():
        param.requires_grad=False
    for param in modules["encoder"].parameters():
        param.requires_grad=False
    for param in modules["decoder"].parameters():
        param.requires_grad=False

    est = SymbolicTransformerRegressor(env, modules["embedder"], modules["encoder"], modules["decoder"])

    predicted_tree = est.fit([X_to_fit, X_to_fit], [Y_to_fit, Y_to_fit], verbose=True, required_operators=None, max_input_points=10000, threshold_constant=0.0)

    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(est)

    print(predicted_tree)
