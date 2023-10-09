from distutils.log import INFO
from logging import getLogger
import os
import io
import sys
import copy
import json
import traceback

# import math
import numpy as np
import boolformer.envs.encoders as encoders
import boolformer.envs.generators as generators
import boolformer.envs.simplifiers as simplifiers
from boolformer.envs.generators import all_operators

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import collections
from ..utils import bool_flag, timeout, MyTimeoutError

CLEAR_SYMPY_CACHE_FREQ = 10000
SPECIAL_WORDS = ["EOS", "PAD", "(", ")", "SPECIAL", "OOD_unary_op", "OOD_binary_op", "OOD_constant", "False", "True"]
logger = getLogger()


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class BooleanEnvironment(object):

    TRAINING_TASKS = {"recurrence"}

    def __init__(self, params):
        
        params.max_vars = params.max_active_vars + params.max_inactive_vars

        self.params = params
        self.float_tolerance = params.float_tolerance
        self.additional_tolerance = [
            float(x) for x in params.more_tolerance.split(",") if len(x) > 0
        ]
        
        self.generator = generators.RandomBooleanFunctions(params)
        self.input_encoder = encoders.Boolean(params)
        self.input_words = SPECIAL_WORDS+sorted(list(set(self.input_encoder.symbols)))
        self.output_words = sorted(list(set(self.generator.symbols)))
        self.output_words = SPECIAL_WORDS+self.output_words
        self.output_encoder = encoders.Equation(params, self.output_words)
        self.generator.equation_encoder = self.output_encoder

        self.simplifier = simplifiers.Simplifier(self.output_encoder, self.generator)
    
        # number of words / indices
        self.input_id2word = {i: s for i, s in enumerate(self.input_words)}
        self.output_id2word = {i: s for i, s in enumerate(self.output_words)}
        self.input_word2id = {s: i for i, s in self.input_id2word.items()}
        self.output_word2id = {s: i for i, s in self.output_id2word.items()}
        
        assert len(self.input_words) == len(set(self.input_words))
        assert len(self.output_words) == len(set(self.output_words))
        self.n_words = params.n_words = len(self.output_words)
        self.eos_index = params.eos_index = self.output_word2id["EOS"]
        self.pad_index = params.pad_index = self.output_word2id["PAD"]

        logger.info(f"vocabulary: {len(self.input_word2id)} input words, {len(self.output_word2id)} output_words")
        # logger.info(f"output words: {self.output_word2id.keys()}")
    

    def input_to_infix(self, lst, str_array=True):
        m = self.input_encoder.decode(lst)
        if m is None:
            return "Invalid"
        if str_array:
            return np.array2string(np.array(m))
        else:
            return np.array(m)


    def output_to_infix(self, lst, str_array=True):
        m = self.output_encoder.decode(lst)
        if m is None:
            return "Invalid"
        return m.infix()

    #@timeout(3)
    def gen_expr(self, 
                 train, 
                 n_inactive_vars=None,
                 n_active_vars=None, 
                 n_ops=None, 
                 n_points=None):
        
        if n_active_vars is None:
            if self.params.use_controller:
                n_active_vars = self.rng.randint(self.params.min_active_vars, self.params.max_active_vars + 1)

        n_rejected = 0
        while True:
            try:
                res = self._gen_expr(
                    train,
                    n_active_vars=n_active_vars,
                    n_inactive_vars=n_inactive_vars,
                    n_ops=n_ops,
                    n_points=n_points,
                )
                if res is None:
                    if self.params.debug: 
                        print(traceback.format_exc())
                    assert False
                inputs, outputs, val_inputs, val_outputs, tree, info = res
                #print(n_active_vars, 'rejected',n_rejected)
                return inputs, outputs, val_inputs, val_outputs, tree, info
            except (AssertionError, MyTimeoutError):
                n_rejected += 1
                continue
            except:
                if self.params.debug:
                    print(traceback.format_exc())
                continue

    #@timeout(1)
    def _gen_expr(self, 
                 train, 
                 n_active_vars=None,
                 n_inactive_vars=None, 
                 n_ops=None, 
                 n_points=None,
                 ):

        if n_active_vars and not n_ops:
            n_ops = self.rng.randint(n_active_vars-1, self.params.max_ops + 1)
        if n_ops is not None:
            assert n_ops >= n_active_vars - 1
        
        tree, n_active_vars, n_inactive_vars = self.generator.generate(rng=self.rng, 
                                               n_active_vars=n_active_vars, 
                                               n_inactive_vars=n_inactive_vars,
                                               n_ops=n_ops)
        n_vars = n_active_vars + n_inactive_vars

        # check num vars
        # original_data = self.rng.choice([True,False], size=(100, n_vars))
        # is_active = np.zeros(n_vars, dtype=bool)
        # for i in range(n_vars):
        #     perturbed_data = original_data.copy()
        #     perturbed_data[:,i] = np.logical_not(perturbed_data[:,i])
        #     if not np.all(tree.val(original_data) == tree.val(perturbed_data)):
        #         is_active[i] = True
        # n_active_vars_effective = np.sum(is_active)
        #if n_active_vars_effective < n_active_vars:
        #    return None
        #n_active_vars = n_active_vars_effective
            
        if self.params.simplify_form != "none":
            if n_active_vars > 6 and self.params.simplify_form in ['cnf','dnf','shortest']: simplify_form = "basic" # too slow above 6
            else: simplify_form = self.params.simplify_form
            tree.simplify()
            tree = self.simplifier.simplify_tree(tree, simplify_form=simplify_form)
            tree = self.simplifier.simplify_tree(tree, simplify_form=simplify_form)
            tree.simplify()

        if len(tree.prefix().split(',')) > self.params.max_output_len:
            return None

        n_ops = tree.get_n_ops()
        n_binary_ops = tree.get_n_binary_ops()
        n_active_vars = tree.get_n_vars()
        #if n_active_vars < self.params.min_active_vars:
        #    return None
        
        if self.params.input_truth_table:
            if n_points is None:
                n_points = 2**n_vars

            # compute truth table   
            if n_points <= self.params.max_points:  
                truth_table = np.array(np.meshgrid(*([np.array([True, False])] * n_vars))).T.reshape(-1, n_vars)
                inputs  = truth_table
                val_inputs = truth_table
            else:
                n_points = self.params.max_points
                truth_table = self.rng.choice([True, False], size=(10*n_points, n_vars))
                truth_table = np.unique(truth_table, axis=0)
                inputs = truth_table[:n_points]
                val_inputs = truth_table[n_points:2*n_points]
            trajectory_flip_prob = 0
        else:
            n_points = self.rng.choice(np.linspace(self.params.min_points, self.params.max_points+1,5).astype(int))
            trajectory_flip_prob = self.rng.choice(np.linspace(0, 1/4, 10))
            trajectory     = self.rng.choice([True, False], size=(1, n_vars))
            for _ in range(2*n_points):
                noise = self.rng.choice([True,False], size=(1, n_vars), p=[trajectory_flip_prob, 1-trajectory_flip_prob])
                trajectory = np.vstack([trajectory, np.logical_xor(trajectory[-1], noise)])
            inputs = trajectory[:n_points]
            val_inputs = trajectory[n_points:2*n_points]

        if self.params.gotu and train:
            inputs[:,0] = True

        outputs = tree.val(inputs)
        if sum(outputs)/len(outputs) in [0,1]:
            return None
        val_outputs = tree.val(val_inputs)

        # add noise
        flip_prob = self.rng.choice(np.linspace(0, self.params.max_flip_prob, 5))
        if flip_prob > 0:
            input_noise = self.rng.choice([True,False], size=(n_points, n_vars), p=[flip_prob, 1-flip_prob])
            inputs = np.logical_xor(inputs, input_noise)
            noise = self.rng.choice([True,False], size=(n_points), p=[flip_prob, 1-flip_prob])
            outputs = np.logical_xor(outputs, noise)
        
        info = {"n_ops": n_ops, 
                "n_binary_ops": n_binary_ops, 
                "n_vars":n_vars, 
                "n_active_vars":n_active_vars, 
                "n_inactive_vars":n_inactive_vars,
                "n_points":n_points,
                "flip_prob":flip_prob,
                "trajectory_flip_prob":trajectory_flip_prob,
        }
        
            
        return inputs, outputs, val_inputs, val_outputs, tree, info
    

    def code_class(self, tree):
        return {"ops": tree.get_n_ops()}
        
    def decode_class(self, n_ops):
        return n_ops

    def create_train_iterator(self, task, data_path, params, args={}):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")
        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
            **args
        )
        return DataLoader(
            dataset,
            timeout=0 if params.num_workers == 0 else 1800,
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size, input_length_modulo, **args
    ):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][int(data_type[5:])]
            ),
            size=size,
            type=data_type,
            input_length_modulo=input_length_modulo,
            **args
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        
        parser.add_argument("--simplify_form", type=str, default="boolean_package",
                            choices = ["none", "basic", "shortest", "cnf", "dnf", "boolean_package"],
                            help="Whether to use further sympy simplification")

        #generator 
        parser.add_argument("--min_ops", type=int, default=1,
                            help="Number of unary or binary operators")
        parser.add_argument("--max_ops", type=int, default=15,
                            help="Number of unary or binary operators")  
        parser.add_argument("--max_output_len", type=int, default=200,
                        help="Max output length")
        parser.add_argument("--unary_prob", type=float, default=.5,
                            help="Proba of generating NOT operators")    
        parser.add_argument("--min_inactive_vars", type=int, default=0,
                            help="Number of variables")
        parser.add_argument("--max_inactive_vars", type=int, default=0,
                            help="Number of variables")
        parser.add_argument("--max_flip_prob", type=float, default=0.,
                            help="Number of variables")
        parser.add_argument("--min_active_vars", type=int, default=3,
                            help="Number of variables")
        parser.add_argument("--max_active_vars", type=int, default=16,
                            help="Number of variables")
    
        parser.add_argument("--input_truth_table", type=bool_flag, default=True,
                            help="Whether to input whole truth table")
        parser.add_argument("--gotu", type=bool_flag, default=False,
                            help="Whether to hide part of the input distribution")
        parser.add_argument("--min_points", type=int, default=4,
                            help="Number of variables")
        parser.add_argument("--max_points", type=int, default=1024,
                            help="Number of points")                            
        parser.add_argument("--min_points_eval", type=int, default=None,
                            help="Number of variables")
        parser.add_argument("--max_points_eval", type=int, default=None,
                            help="Number of points")
        parser.add_argument("--use_controller", type=bool_flag, default=True,
                            help="Whether to enforce same number of examples per support")  
        parser.add_argument("--operators_to_use", type=str, 
                            default="not,and,or",
                            help="Op probas to downsample")  
        parser.add_argument("--max_value", type=int, default=10000,
                            help="Maximal value of the constant")
        parser.add_argument("--special_const_proba", type=int, default=3,
                            help="How much more common special constants (e,pi) are than integers")  
  
        # evaluation
        parser.add_argument("--float_tolerance", type=float, default=1e-10,
                            help="error tolerance for float results")
        parser.add_argument("--more_tolerance", type=str, default="0.0,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1", 
                            help="additional tolerance limits")
        
        # debug
        parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                            help="Debug multi-GPU / multi-node within a SLURM job")
        parser.add_argument("--debug", help="Enable all debug flags",
                            action="store_true")


class EnvDataset(Dataset):
    def __init__(self, env, task, train, params, path, size=None, type=None,input_length_modulo=-1, **args):
        super(EnvDataset).__init__()
        self.env = env
        self.params = env.params
        if 'n_ops_prob' in args:
            self.env.generator.n_ops_prob=args['n_ops_prob']
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        self.type = type
        self.input_length_modulo=input_length_modulo

        if "test_env_seed" in args:
            self.test_env_seed = args["test_env_seed"]
        else:
            self.test_env_seed = None 
        if "env_info" in args:
            self.env_info=args["env_info"]
        else:
            self.env_info=None

        assert task in BooleanEnvironment.TRAINING_TASKS
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0
        self.remaining_data=0
        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        self.batch_load = params.batch_load
        self.reload_size = params.reload_size
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path), "{} not found".format(path)
            if params.batch_load and self.train:
                self.load_chunk()
            else:
                logger.info(f"Loading data from {path} ...")
                with io.open(path, mode="r", encoding="utf-8") as f:
                    # either reload the entire file, or the first N lines
                    # (for the training set)
                    if not train:
                        lines = []
                        for i, line in enumerate(f):
                            lines.append(json.loads(line.rstrip()))
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            if i == params.reload_size:
                                break
                            if i % params.n_gpu_per_node == params.local_rank:
                                #lines.append(line.rstrip())
                                lines.append(json.loads(line.rstrip()))
                #self.data = [xy.split("=") for xy in lines]
                #self.data = [xy for xy in self.data if len(xy) == 3]
                self.data=lines
                logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 10000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def load_chunk(self):
        self.basepos = self.nextpos
        logger.info(
            f"Loading data from {self.path} ... seekpos {self.seekpos}, "
            f"basepos {self.basepos}"
        )
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            for i in range(self.reload_size):
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip().split("|"))
            self.seekpos = 0 if endfile else f.tell()

        self.data = [xy.split("\t") for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, "
            f"nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        inputs, outputs, val_inputs, val_outputs, trees, infos = zip(*elements)
        info_tensor = {info_type: torch.Tensor([info[info_type] for info in infos]) for info_type in infos[0].keys()} 
        

        trees = [self.env.output_encoder.encode(tree) for tree in trees]
        trees = [torch.LongTensor([self.env.output_word2id[w] for w in seq]) for seq in trees]
        trees, trees_len = self.batch_output_sequences(trees)

        res = []
        for (x_arr, y_arr) in zip(inputs, outputs):
            seq_toks = []
            for (x,y) in zip(x_arr, y_arr):
                x_toks = self.env.input_encoder.encode(x)
                y_toks = self.env.input_encoder.encode(y)
                x_toks = [*x_toks, *["PAD" for _ in range(self.params.max_vars - len(x_toks))]]
                toks = [*x_toks, *y_toks]
                seq_toks.append([self.env.input_word2id[tok] for tok in toks])
            res.append(torch.LongTensor(seq_toks))
        data = res
        data, data_len = self.batch_input_sequences(data)


        samples = {
            "data": data, # (slen, n, dim)
            "data_len": data_len, # (n)
            "val_inputs": val_inputs,
            "val_outputs": val_outputs,
            "inputs": inputs,
            "outputs": outputs,
            "trees": trees,
            "trees_len": trees_len,
            "infos": info_tensor,
        }
        return samples#(data, data_len), (trees, trees_len), info_tensor
    
    def batch_input_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n, dim) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) for s in sequences])
        bs, slen = len(lengths), max(lengths)
        dim = sequences[0].shape[-1]

        sent = torch.LongTensor(slen, bs, dim).fill_(self.env.pad_index)
        for i, seq in enumerate(sequences):
            sent[0 : len(seq), i, :] = seq

        return sent, torch.LongTensor(lengths)
    
    def batch_output_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.env.pad_index)
        #assert lengths.min().item() > 2

        sent[0] = self.env.eos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.env.eos_index

        return sent, lengths
    

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [worker_id, self.global_rank, self.env_base_seed]
            if self.env_info is not None:
                seed+=[self.env_info]
            self.env.rng = np.random.RandomState(seed)
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{seed} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = self.test_env_seed  if "valid" in self.type else 0
            self.env.rng = np.random.RandomState(seed)
            logger.info(
                "Initialized {} generator, with seed {} (random state: {})".format(self.type, seed, self.env.rng)    
            )
            #print(self.generate_sample())

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """s
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index
        x = self.data[idx]
        x1 = x["x1"].split(" ")
        x2 = x["x2"].split(" ")
        infos = {}
        for col in x:
            if col not in ["x1", "x2", "tree"]:
                infos[col]=int(x[col])
        return x1, x2, infos 
        
        #x, y = self.data[idx]
        #x = x.split()
        #y = y.split()
        #assert len(x) >= 1 and len(y) >= 1
        #return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """

        while True:
            if self.task == "recurrence":
                output = self.env.gen_expr(self.train)
            else:
                raise Exception(f"Unknown data type: {self.task}")
            if output is None:
                continue # discard problematic 
            break

        # if CLEAR_SYMPY_CACHE_FREQ > 0 and self.count % CLEAR_SYMPY_CACHE_FREQ == 0:
        #     logger.warning(f"Clearing SymPy cache (worker {self.get_worker_id()})")
        #     clear_cache()
        x, y, val_x, val_y, tree, info = output
        return x, y, val_x, val_y, tree, info
