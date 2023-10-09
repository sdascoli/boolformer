import json
import random
import argparse
import numpy as np
import torch
import os
import pickle
import setproctitle

import boolformer
from boolformer.slurm import init_signal_handler, init_distributed_mode
from boolformer.utils import bool_flag, initialize_exp
from boolformer.model import check_model_params, build_modules
from boolformer.envs import BooleanEnvironment, build_env
from boolformer.trainer import Trainer
from boolformer.evaluator import Evaluator

import wandb


np.seterr(all='raise')


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Recurrence prediction", add_help=False)

    # main parameters
    parser.add_argument("--use_wandb", type=bool_flag, default=False,
                        help="Log to wandb")
    parser.add_argument("--dump_path", type=str, default="",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--print_freq", type=int, default=50,
                        help="Print every n steps")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model parameters
    parser.add_argument("--activation", type=str, default='silu',
                        help="Activation function")
    parser.add_argument("--emb_emb_dim", type=int, default=64,
                        help="Encoder embedding layer size")
    parser.add_argument("--emb_expansion_factor",type=int,default=1,
                        help="Expansion factor for embedder")
    parser.add_argument("--enc_emb_dim", type=int, default=256,
                        help="Encoder embedding layer size")
    parser.add_argument("--dec_emb_dim", type=int, default=None,
                        help="Decoder embedding layer size")
    parser.add_argument("--n_emb_layers", type=int, default=1,
                        help="Number of Transformer layers in the encoder")
    parser.add_argument("--n_enc_layers", type=int, default=4,
                        help="Number of Transformer layers in the encoder")
    parser.add_argument("--n_dec_layers", type=int, default=None,
                        help="Number of Transformer layers in the decoder")
    parser.add_argument("--n_enc_heads", type=int, default=16,
                        help="Number of Transformer encoder heads")
    parser.add_argument("--n_dec_heads", type=int, default=None,
                        help="Number of Transformer decoder heads")
    parser.add_argument("--n_enc_hidden_layers", type=int, default=1,
                        help="Number of FFN layers in Transformer encoder")
    parser.add_argument("--n_dec_hidden_layers", type=int, default=1,
                        help="Number of FFN layers in Transformer decoder")
    parser.add_argument("--enc_positional_embeddings",type=str,default=None,
                        help="Use none/learnable/sinusoidal/alibi embeddings",)
    parser.add_argument("--dec_positional_embeddings",type=str,default="learnable",
                        help="Use none/learnable/sinusoidal/alibi embeddings",)

    parser.add_argument("--norm_attention", type=bool_flag, default=False,
                        help="Normalize attention and train temperaturee in Transformer")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")

    # training parameters
    
    parser.add_argument("--curriculum_n_ops", type=bool, default=False,
                    help="Whether we use a curriculum strategy for the number of ops during training")
    parser.add_argument("--env_base_seed", type=int, default=-1,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--test_env_seed", type=int, default=1,
                        help="Test seed for environments")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Number of sentences per batch")
    parser.add_argument("--batch_size_eval", type=int, default=None,
                        help="Number of sentences per batch during evaluation (if None, set to 1.5*batch_size)")
    parser.add_argument("--optimizer", type=str, default="adam_cosine,warmup_updates=10000,init_period=150000,period_mult=1.5,lr_shrink=0.5,lr=0.0002",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=1,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=300000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=1000,
                        help="Number of epochs")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of CPU workers for DataLoader")

    # export data / reload it
    parser.add_argument("--export_data", type=bool_flag, default=False,
                        help="Export data and disable training.")
    parser.add_argument("--reload_data", type=str, default="",
                        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)")
    parser.add_argument("--reload_size", type=int, default=-1,
                        help="Reloaded training set size (-1 for everything)")
    parser.add_argument("--batch_load", type=bool_flag, default=False,
                        help="Load training set by batches (of size reload_size).")

    # tasks
    parser.add_argument("--tasks", type=str, default="recurrence",
                        help="Tasks")

    # beam search configuration
    parser.add_argument("--beam_eval", type=bool_flag, default=True, 
                        help="Evaluate with beam search decoding.")
    parser.add_argument("--beam_eval_train", type=int, default=0,
                        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--beam_length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--beam_early_stopping", type=bool_flag, default=True,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")
    parser.add_argument("--beam_type", type=str, default="search", 
                        help="Beam search or sampling")
    parser.add_argument("--beam_temperature", type=float, default=0.1, 
                        help="Beam temperature in case of sampling")

    # reload pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # evaluation
    parser.add_argument("--eval_size", type=int, default=10000,
                        help="Size of valid and test samples")
    parser.add_argument("--train_noise", type=float, default=0,
                        help="Amount of noise at train time")
    parser.add_argument("--eval_noise", type=float, default=0,
                        help="Amount of noise at test time")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_data", type=str, default="",
                        help="Path of data to eval")
    parser.add_argument("--eval_verbose", type=int, default=0,
                        help="Export evaluation details")
    parser.add_argument("--eval_verbose_print", type=bool_flag, default=False,
                        help="Print evaluation details")
    parser.add_argument("--eval_input_length_modulo", type=int, default=-1,
                        help="Compute accuracy for all input lengths modulo X. -1 is equivalent to no ablation")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False,
                        help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--windows", type=bool_flag, default=True,
                        help="Windows version (no multiprocessing for eval)")
    parser.add_argument("--nvidia_apex", type=bool_flag, default=False,
                        help="NVIDIA version of apex")
    
    BooleanEnvironment.register_args(parser)

    return parser


def main(params):

    setproctitle.setproctitle(params.exp_id)

    if params.use_wandb:
        wandb.login() 
        wandb.init(
        # set the wandb project where this run will be logged
        project="sr-for-booleans",
        group=params.exp_name,
        name=params.exp_id,
        # track hyperparameters and run metadata
        config=params.__dict__,
        resume=True
        )

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    boolformer.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None: params.batch_size_eval = int(params.batch_size)
    env = build_env(params)

    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    # training
    if params.reload_data!="":
        data_types = ["valid{}".format(i) for i in range(1,len(trainer.data_path["recurrence"]))]
    else:
        data_types = ["valid1"]
    evaluator.set_env_copies(data_types)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(data_types)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    scores = evaluator.run_all_evals(data_types)
    if params.is_master:
        logger.info("__log__:%s" % json.dumps(scores))

    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_equations = 0

        while trainer.n_equations < trainer.epoch_size:

            # training steps
            for task_id in np.random.permutation(len(params.tasks)):
                task = params.tasks[task_id]
                if params.export_data:
                    trainer.export_data(task)
                else:
                    trainer.enc_dec_step(task)
                trainer.iter()
        logger.info("============ End of epoch %i ============" % trainer.epoch)

        trainer.save_best_model(scores)
        trainer.save_periodic()

        # evaluate perplexity
        if not params.export_data:
            scores = evaluator.run_all_evals(data_types)
            if params.is_master:
                logger.info("__log__:%s" % json.dumps(scores))
                if params.use_wandb:
                    wandb.log({metric:score for metric,score in scores.items() if 'info' not in metric})
                
            if params.curriculum_n_ops:
                neg_accuracy_per_n_ops = {int(measure.split("_")[-1]): 1.-acc/100. for measure, acc in scores.items() if "n_ops" in measure and "valid1" in measure}
                min_neg_accuracy_per_n_ops = min(neg_accuracy_per_n_ops.values())
                for op in range(1,params.max_ops+1):
                    if op not in neg_accuracy_per_n_ops:
                        neg_accuracy_per_n_ops[op]=min_neg_accuracy_per_n_ops
                neg_accuracy_per_n_ops = {key : neg_accuracy_per_n_ops[key] for key in sorted(neg_accuracy_per_n_ops.keys())}
                probabilities = np.array(list(neg_accuracy_per_n_ops.values()))
                probabilities = probabilities[:params.max_ops]
                probabilities /= probabilities.sum()
                trainer.set_new_train_iterator_params({"n_ops_prob": probabilities, "env_info": trainer.epoch})
                
        # end of epoch
        trainer.end_epoch(scores)


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    if params.eval_only:
        # read params from pickle
        pickle_file = params.reload_checkpoint + "/params.pkl"
        assert os.path.isfile(pickle_file)
        pk = pickle.load(open(pickle_file, 'rb'))
        pickled_args = pk.__dict__
        del pickled_args['exp_id']
        for p in params.__dict__:
            if p in pickled_args and p not in ["eval_only", "dump_path", "reload_checkpoint", "batch_size_eval", "beam_size", "beam_selection_metric", "use_wandb", "eval_size"]:
                params.__dict__[p] = pickled_args[p]
        params.is_slurm_job = False
        params.local_rank = -1
        params.master_port = -1
        params.num_workers = 1

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        if params.exp_id == '':
            params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True

    # check parameters
    check_model_params(params)

    # run experiment
    main(params)
