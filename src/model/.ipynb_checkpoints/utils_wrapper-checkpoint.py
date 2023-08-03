from abc import ABC, abstractmethod
import sklearn
from scipy.optimize import minimize, leastsq
from src.metrics import compute_metrics
import numpy as np
import multiprocessing
#multiprocessing.set_start_method("spawn")
import sys
from ..utils import bool_flag, timeout, MyTimeoutError


class Scaler(ABC):
    """
    Base class for scalers
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def rescale_function(self, env, tree, a, b):
        prefix = tree.prefix().split(",")
        idx = 0
        while idx < len(prefix):
            if prefix[idx].startswith("x_"):
                k = int(prefix[idx][-1])
                a_k, b_k = str(a[k]), str(b[k])
                prefix_to_add = ["add", b_k, "mul", a_k, prefix[idx]]
                prefix = prefix[:idx] + prefix_to_add + prefix[min(idx + 1, len(prefix)):]
                idx += len(prefix_to_add)
            else:
                idx+=1
                continue
        rescaled_tree = env.word_to_infix(prefix, is_float=False, str_array=False)
        return rescaled_tree

class StandardScaler(Scaler):
    def __init__(self):
        """
        transformation is: 
        x' =  (x - mean)/std
        """
        self.scaler = sklearn.preprocessing.StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_X = self.scaler.fit_transform(X)
        return scaled_X
    
    def transform(self, X):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        return (X-m)/s

    def get_params(self):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        a, b = 1/s, -m/s
        return (a, b)
    
class MinMaxScaler(Scaler):
    def __init__(self):
        """
        transformation is: 
        x' =  2.*(x-xmin)/(xmax-xmin)-1.
        """
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))

    def fit(self, X):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_X = self.scaler.fit_transform(X)
        return scaled_X

    def transform(self, X):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        return 2*(X-val_min)/(val_max-val_min)-1.

    def get_params(self):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        a, b = 2./(val_max-val_min), -1.-2.*val_min/(val_max-val_min)
        return (a, b)
    
class RefinementAlgorithm(ABC):
    def __init__(self):
        pass

    def set_args(self, **args):
        for k, v in args.items():
            setattr(self, k, v)

    @abstractmethod
    def go(self, X, y, initialization):
        pass

    def wrap_equation_floats(self, constants):
        tree=self.tree
        env=self.env
        prefix = tree.prefix().split(",")
        j = 0
        for i, elem in enumerate(prefix):
            if elem == "CONSTANT":
                prefix[i] = str(constants[j])
                j += 1
        assert j == len(constants), "all constants were not fitted"
        assert "CONSTANT" not in prefix, "tree {} got constant after wrapper {}".format(tree, constants)
        tree_with_constants = env.word_to_infix(prefix, is_float=False, str_array=False)
        return tree_with_constants



class ScipyRefinement(RefinementAlgorithm):
    def __init__(self, **args) :
        self.options = {"BFGS": 
                            {"gtol": 1e-05,
                            'norm': np.inf, 
                            'eps': 1.4901161193847656e-08, 
                            'maxiter': None, 
                            'disp': False, 
                            'return_all': False, 
                            'finite_diff_rel_step': None        
                            },
                        "Nelder-Mead":
                            {
                            'maxiter': None, 
                            'maxfev': None, 
                            'disp': False, 
                            'return_all': False, 
                            'initial_simplex': None, 
                            'xatol': 0.0001, 
                            'fatol': 0.0001, 
                            'adaptive': False}   
                        }

        for k, v in args.items():
           for method in self.options.keys(): 
               if k in self.options[method]:
                   self.options[method][k]=v

    def set_args(self, **args):
        for k, v in args.items():
            if k in ["method", "X", "y", "env", "tree", "metric", "random_init", "batchsize"]:
                setattr(self, k, v)
            else:
                for method in self.options.keys(): 
                    if k in self.options[method]:
                        try:
                            v_float = float(v)
                            if v_float.is_integer():
                                self.options[method][k]=int(v_float)
                            else:
                                self.options[method][k]=v_float
                        except:
                            self.options[method][k]=v

    @timeout(5)
    def go(self, initialization):
        method=self.method
        random_init=self.random_init=="True"
        assert method in ["BFGS", "Nelder-Mead"]
        self.loss_history=[]
        if random_init:
            initialization=np.random.randn(*initialization.shape)
        self.best_so_far, self.best_so_far_loss = self.wrap_equation_floats(initialization), self.objective(initialization)
        minimize(fun=self.objective, x0=initialization, method=method, options=self.options[method], callback=self.callback)
        return  self.best_so_far

    def callback(self, constants):
        loss = self.objective(constants)
        if loss < self.best_so_far_loss:
            self.best_so_far_loss=loss
            self.best_so_far=self.wrap_equation_floats(constants)
        self.loss_history.append(loss)

    def objective(self, constants):
        X=self.X
        y=self.y
        idx = np.random.choice(X.shape[0], size=int(getattr(self, "batchsize", 256)))
        X_batch, y_batch = X[idx], y[idx]
        metric=self.metric
        tree_with_constants = self.wrap_equation_floats(constants)
        assert tree_with_constants is not None
        y_tilde = tree_with_constants.val(X_batch)
        metrics = compute_metrics({"true": [y_batch], "predicted": [y_tilde]}, metrics=metric)
        loss = metrics[metric][0]
        if not metric.startswith("_"):
            loss=-loss
        return loss

class LSQRefinement(RefinementAlgorithm):
    def __init__(self, **args) :
        
        for k, v in args.items():
           for method in self.options.keys(): 
               if k in self.options[method]:
                   self.options[method][k]=v

    def set_args(self, **args):
        for k, v in args.items():
            if k in ["method", "X", "y", "env", "tree", "metric", "random_init"]:
                setattr(self, k, v)
            
    def go(self, initialization):
        random_init=self.random_init=="True"
        if random_init:
            initialization=np.random.randn(*initialization.shape)
        self.best_so_far, self.best_so_far_loss = self.wrap_equation_floats(initialization), self.residual(initialization)
        out = leastsq(func=self.residual, x0=initialization)
        final_constants = out[0]
        return  self.wrap_equation_floats(final_constants)

    def residual(self, constants):
        X=self.X
        y=self.y
        tree_with_constants = self.wrap_equation_floats(constants)
        assert tree_with_constants is not None
        y_tilde = tree_with_constants.val(X)
        loss = (y-y_tilde)**2
        loss = loss[:,0]
        return loss


class NevergradRefinement(RefinementAlgorithm):
    def __init__(self, **args) :


        for k, v in args.items():
            setattr(self, k, v)

    def set_args(self, **args):
        for k, v in args.items():
            setattr(self, k, v)
            
    def callback(self, constants):
        loss = self.objective(constants)
        if loss < self.best_so_far_loss:
            self.best_so_far_loss=loss
            self.best_so_far=self.wrap_equation_floats(constants)
        self.loss_history.append(loss)

    def go(self, initialization):
        import nevergrad as ng
        from concurrent import futures

        random_init=self.random_init=="True"
        if random_init:
            initialization=np.random.randn(*initialization.shape)
            self.best_so_far, self.best_so_far_loss = self.wrap_equation_floats(initialization), self.objective(initialization)

        # optimizers to try: NGOpt, CMA, TwoPointsDE, RandomSearch
        optims = ["NGOpt", "CMA", "DE", "TwoPointsDE"] #* 8
        #with futures.ProcessPoolExecutor(max_workers=len(optims)) as executor:
        #with futures.ProcessPoolExecutor(max_workers=len(optims)) as executor:
        with futures.ThreadPoolExecutor(max_workers=len(optims)) as executor:
            jobs = []
            for optim in optims:
                parametrization = ng.p.Array(init=initialization).set_mutation(sigma=1)
                #parametrization.mutate()
                optimizer = ng.optimizers.registry[optim](parametrization=parametrization, budget=self.budget, num_workers=1)
                # jobs.append(executor.submit(optimizer.minimize, self.objective))
                jobs.append(executor.submit(run_optim, optimizer, self.objective))#self.objective))
        recommendation = min((j.result() for j in jobs), key=lambda r: self.objective(r.value))
        #recommendation = optimizer.minimize(self.objective, executor=None)
        final_constants = recommendation.value
        return  self.wrap_equation_floats(final_constants)

    def objective(self, constants):
        X=self.X
        y=self.y
        metric=self.metric
        tree_with_constants = self.wrap_equation_floats(constants)
        assert tree_with_constants is not None
        y_tilde = tree_with_constants.val(X)
        metrics = compute_metrics({"true": [y], "predicted": [y_tilde]}, metrics=metric)
        loss = metrics[metric][0]
        if not metric.startswith("_"):
            loss=-loss
        return loss


def run_optim(optimizer, objective):
    return optimizer.minimize(objective)

def func(x):
    return sum(x)