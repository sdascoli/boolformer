def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import glob
import os
import string
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import sys
import copy
from pathlib import Path
from sympy import *
import pickle
from collections import defaultdict, OrderedDict
import math
import scipy.special
import warnings
from sklearn.manifold import TSNE
from IPython.display import display
from importlib import reload  # Python 3.4+
import importlib.util
import subprocess
import pandas as pd
import sympy 
import tqdm
import time
import re
import gdown

from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#from tabpfn import TabPFNClassifier

try:
    user = os.getlogin()
    BASE_PATH = '/data/rcp'
except:
    BASE_PATH = '/sb_u0621_liac_scratch'

def get_most_free_gpu():
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
    free_memory = [int(x) for x in output.decode().strip().split('\n')]
    most_free = free_memory.index(max(free_memory))
    # set visible devices to the most free gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(most_free)
    return most_free

def module_from_file(module_name, file_path):
    print(file_path, module_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def import_file(full_path_to_module):
    module_dir, module_file = os.path.split(full_path_to_module)
    module_name, module_ext = os.path.splitext(module_file)
    save_cwd = os.getcwd()
    os.chdir(module_dir)
    module_obj = __import__(module_name)
    module_obj.__file__ = full_path_to_module
    globals()[module_name] = module_obj
    os.chdir(save_cwd)
    return module_obj

############################ GENERAL ############################


def find(array, value):
    idx= np.argwhere(np.array(array)==value)[0,0]
    return idx

def select_runs(runs, params, constraints):
    selected_runs = []
    for irun, run in enumerate(runs):
        keep = True
        for k,v in constraints.items():
            if type(v)!=list:
                v=[v]
            if (not hasattr(run['args'],k)) or (getattr(run['args'],k) not in v):
                keep = False
                break
        if keep:
            selected_runs.append(run)
    selected_params = copy.deepcopy(params)
    for con in constraints:
        selected_params[con]=[constraints[con]]
    return selected_runs, selected_params

def group_runs(runs, finished_only=True):
    runs_grouped = defaultdict(list)
    for run in runs:
        seedless_args = copy.deepcopy(run['args'])
        del(seedless_args.seed)
        del(seedless_args.name)
        if str(seedless_args) not in runs_grouped.keys(): 
            runs_grouped[str(seedless_args)].append(run) # need at least one run
        else:
            if run['finished'] or not finished_only:
                runs_grouped[str(seedless_args)].append(run)
    runs_grouped = list(runs_grouped.values())
    return runs_grouped

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def permute(array, indices):
    return [array[idx] for idx in indices]

def ordered_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(map(float,labels))[k])] )
    #ax.legend(handles, labels, **kwargs)
    return handles, labels

def legend_no_duplicates(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[i+1:]]
    ax.legend(*zip(*unique), **kwargs)
    
def load_run(params, new_params={}, epoch=None):

    final_params = params
    for arg, val in new_params.items():
        setattr(final_params,arg,val)
    final_params.multi_gpu = False
        
    sys.path.append(os.path.join(BASE_PATH, 'boolean'))
    import boolformer

    print(boolformer)
    from boolformer.model import build_modules, Boolformer
    from boolformer.envs import BooleanEnvironment, build_env
    from boolformer.trainer import Trainer
    from boolformer.evaluator import Evaluator, idx_to_infix, calculate_error
    from boolformer.envs.generators import RandomBooleanFunctions
    
    env = build_env(final_params)
    modules = build_modules(env, final_params)
        
    #trainer = Trainer(modules, env, final_params)
    #evaluator = Evaluator(trainer)
    
    checkpoint_path = os.path.join(final_params.dump_path,f'checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    for k, module in modules.items():
        weights = {k:v for k,v in checkpoint[k].items()}
        module.load_state_dict(weights)
        module.eval()

    embedder = modules["embedder"]
    encoder = modules["encoder"]
    decoder = modules["decoder"]
    embedder.eval()
    encoder.eval()
    decoder.eval()

    boolformer_model = Boolformer(
        env=env,
        embedder=embedder,
        encoder=encoder,
        decoder=decoder,
    )
    
    return boolformer_model

def get_logical_circuit(function_name='multiplexer'):
    to_bin = lambda x : int(''.join([str(int(digit)) for digit in x]),2)
    to_decimal = lambda x : sum(x[::-1].astype(int)*2**np.arange(n_vars//2))

    if function_name.startswith('multiplexer'):
        order=int(function_name.split("_")[-1])
        n_vars = order + 2**order
        tree = lambda x : x[order:][sum(x[:order]*2**np.arange(order)).astype(int)]
    if function_name.startswith('majority'):
        n_vars=int(function_name.split("_")[-1])
        tree = lambda x : sum(x)>=n_vars/2
    elif function_name.startswith('parity'):
        n_vars=int(function_name.split("_")[-1])
        tree = lambda x : 1-sum(x)%2
    elif function_name.startswith('all') or function_name.startswith('any'):
        name, n_vars = function_name.split('_')[0], int(function_name.split('_')[1])
        tree = lambda x : getattr(np,name)(x)
    elif function_name.startswith("comparator"):
        n_vars = 2*int(function_name.split("_")[-1])
        def tree(x):
            factor1, factor2 = to_decimal(x[:n_vars//2]), to_decimal(x[n_vars//2:])
            return 1 if factor1>factor2 else 0
    elif function_name.startswith("multiadder"):
        n_vars = int(function_name.split("_")[-1])
        def tree(x):
            tmp = sum(x.astype(int))
            n_output = int(np.log2(n_vars))+1
            return ['{:0{n_output}b}'.format(tmp, n_output=n_output)[-i] for i in range(n_output)]
    elif function_name.startswith("adder"):
        n_vars = 2*int(function_name.split("_")[-1])
        def tree(x):
            factor1, factor2 = to_bin(x[:n_vars//2]), to_bin(x[n_vars//2:])
            n_output = n_vars//2+1
            return [bin(factor1+factor2)[2:].zfill(n_output)[-i] for i in range(n_output)]
    elif function_name.startswith("multiplier"):
        n_vars = 2*int(function_name.split("_")[-1])
        def tree(x):
            factor1, factor2 = to_bin(x[:n_vars//2]), to_bin(x[n_vars//2:])
            n_output = n_vars
            return [bin(factor1*factor2)[2:].zfill(n_output)[-i] for i in range(n_output)]
    elif function_name.startswith("decoder"):
        n_vars = int(function_name.split("_")[-1])
        def tree(x):
            index = sum(x.astype(int)*2**np.arange(n_vars))
            #index = 4*int(x[0])+2*int(x[1])+int(x[2])
            tmp = [False for i in range(2**n_vars)]
            tmp[index] = True
            return tmp
    elif function_name.startswith("random"):
        n_vars = int(function_name.split("_")[-1])
        truth_table = np.array(np.meshgrid(*([np.array(['0','1'])] * n_vars))).T.reshape(-1, n_vars)
        dict = {''.join(input):np.random.choice([True,False]) for input in truth_table}
        tree = lambda x : dict[''.join(x.astype(int).astype(str))]
    return tree, n_vars

def get_boolnet(idx, env, verbose=False):
    path = os.path.join(BASE_PATH,"pyboolnet/pyboolnet/repository/")
    dataset_names = [os.path.basename(x) for x in glob.glob(path+"*") if "README" not in x and ".py" not in x]
    dataset_name = dataset_names[idx]
    file_path = os.path.join(path, dataset_name, dataset_name+'.bnet')
    with open(file_path, 'r') as f:
        text = f.read().splitlines()
    text = [line for line in text if line and  '#' not in line][1:]
    variables = {line.split(',')[0].strip():f"x_{i}" for i, line in enumerate(text)}
    variables = {k:variables[k] for k in sorted(variables.keys(), key=lambda x : -len(x))}
    trees = [line.split(',')[1].strip() for line in text]
    for k,v in variables.items():
        trees = [tree.replace(k, v) for tree in trees]
    trees = [tree.replace('!','~') for tree in trees]
    trees = [sympy.parse_expr(tree) for tree in trees]
    trees = [env.simplifier.sympy_to_tree(tree) for tree in trees]
    if verbose: print(trees)
    res = lambda x: np.array([tree(x)[0] for tree in trees])
    return res, len(variables)

def generate_data(tree, n_vars=None, n_points=None):
    inputs, outputs = [], []
    val_inputs, val_outputs = [], []
    truth_table = get_truth_table(n_vars)
    # shuffle rows
    if n_points: truth_table = truth_table[np.random.choice(truth_table.shape[0], size=truth_table.shape[0], replace=False)]

    if not n_points:
        for input in truth_table:
            inputs.append(input)
            val_inputs.append(input)
            outputs.append(tree(input))
            val_outputs.append(tree(input))
    else:
        for i in range(n_points):    
            input  = truth_table[i]
            if i < n_points:    
                inputs.append(input)
                outputs.append(tree(input))
            else:
                val_inputs.append(input)
                val_outputs.append(tree(input))

    inputs, outputs = np.array(inputs).astype(bool), np.array(outputs).astype(bool)
    val_inputs, val_outputs = np.array(inputs).astype(bool), np.array(outputs).astype(bool)
    if len(outputs.shape)==1:
        inputs, outputs = [inputs], [outputs]
        val_inputs, val_outputs = [val_inputs], [val_outputs]
    else:
        output_dim = outputs.shape[1]
        inputs, outputs = [inputs for i in range(output_dim)], [outputs[:,i] for i in range(output_dim)]
        val_inputs, val_outputs = [val_inputs for i in range(output_dim)], [val_outputs[:,i] for i in range(output_dim)]
    return inputs, outputs, val_inputs, val_outputs

def get_embeddings(embedder, inputs, outputs):
    x, x_len = embedder.batch(inputs, outputs)
    x = embedder(x)
    return x.squeeze().cpu().detach().numpy()
            
def tsne_plot_2d(embeddings, n_words=10000):
    
    tsne_model = TSNE(n_components=2, init='pca', n_iter=1000, random_state=0)
    new_values = tsne_model.fit_transform(embeddings)

    x = []; y = []
    for value in new_values:
        x.append(value[0]); y.append(value[1])
        
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 8)) 
    for i in range(len(x))[:n_words]:
        plt.scatter(x[i],y[i], color=cmap(i/min(n_words, len(x))))
        #if i%(len(x)//10)==0 or i==len(x)-1:
        #    plt.annotate(f"{i:010b}"[2:], xy=(x[i], y[i]), ha='center', va='center')
    plt.axis(False)
    plt.tight_layout()
    return fig

def plot_attention(model, inputs, outputs):

    env = model.env
    args = env.params
    
    encoder, decoder = model.encoder, model.decoder
    encoder.eval()
    encoder.STORE_OUTPUTS = True
    num_heads = encoder.n_heads
    num_layers = encoder.n_layers
    
    new_args = copy.deepcopy(args)
    new_args.series_length = 15
    pred_trees, error_arr, complexity_arr = model.fit(inputs, outputs, verbose=False, beam_size=10, store_attention=True)
        
    fig, axarr = plt.subplots(num_layers, num_heads, figsize=(2*num_heads,2*num_layers), constrained_layout=True)        
        
    for l in range(num_layers):
        module = encoder.attentions[l]
        scores = module.outputs[0]
        
        for head in range(num_heads):                  
            axarr[l][head].matshow(scores[head])
            
            axarr[l][head].set_xticks([]) 
            axarr[l][head].set_yticks([]) 
                
    cols = [r'Head {}'.format(col+1) for col in range(num_heads)]
    rows = ['Layer {}'.format(row+1) for row in range(num_layers)]
    for icol, col in enumerate(cols):
        axarr[0][icol].set_title(col, fontsize=18, pad=10)
    for irow, row in enumerate(rows):
        axarr[irow][0].set_ylabel(row, fontsize=18, labelpad=10)

    return fig, axarr

def get_truth_table(n_vars):
    table = []
    for i in range(2**n_vars):
        table.append([bool(int(x)) for x in list(bin(i)[2:].zfill(n_vars))])
    return np.array(table)
 
def get_data_drug_discovery(seed=0, data_path='.', problem="TOX", num_points=500, num_test_points=100, num_features=None, balance=True, latex_format=False):

    np.random.seed(seed)

    # read data
    np.random.seed()
    if problem=="BBB":
        dataset_name = 'BBBP_MACCS.csv'
    elif problem=="BBB2":
        dataset_name = 'BBB_ecfp4.csv'
    elif problem=="HIV":
        dataset_name = 'HIV_MACCS.csv'
    elif problem=="HIV2":
        dataset_name = 'HIV_ecfp4.csv'
    elif problem=="TOX":
        dataset_name = 'TOX_MACCS.csv'
    elif problem=="TOX2":
        dataset_name = 'TOX_ecfp4.csv'
    elif problem.startswith("smell"):
        dataset_name = 'Leffingwell_dataset.csv'
        smell = problem.split("_")[1]
    df = pd.read_csv(os.path.join(data_path, dataset_name))
    df = df.drop('smiles', axis=1)
    if problem.startswith("smell"):
        df['activity'] = df['labels'].str.contains(smell)
        df.drop('labels', axis=1, inplace=True)

    # sort features
    if num_features is not None:
        positive, negative = df[df['activity']==1], df[df['activity']==0]
        covar = positive.mean(axis=0) - negative.mean(axis=0)
        feature_order = covar.abs().sort_values(ascending=False).index
        df = df.reindex(feature_order, axis=1)
        # keep only first num_features
        df = df.iloc[:,:num_features]
    df = df.drop_duplicates()

    #num_total = num_points + num_test_points
    num_test_points = min(num_test_points, len(df)-num_points)

    df = df.sample(frac=1, random_state=0)
    val_df = df[:num_test_points]
    df = df.drop(val_df.index)
    # printnumber of positives
    #print(len(df[df['activity']==1]), len(val_df[val_df['activity']==1]))
    # rebalance data if necessary
    if balance:
        num_positives = num_points//2 #min(num_points//2, len(df[df['activity']==1]))
        num_negatives = num_points - num_positives
        replace_positives = num_positives>len(df[df['activity']==1])
        replace_negatives = num_negatives>len(df[df['activity']==0])
        positives = df[df['activity']==1].sample(n=num_positives, replace=replace_positives, random_state=seed)
        negatives = df[df['activity']==0].sample(n=num_negatives, replace=replace_negatives, random_state=seed)
        df = pd.concat([positives, negatives])
    else:
        df = df.sample(n=num_points, replace=True, random_state=seed)

    outputs, val_outputs = df['activity'], val_df['activity']

    df = df.drop('activity', axis=1)
    val_df = val_df.drop('activity', axis=1)

    inputs, val_inputs = df.values, val_df.values
    outputs, val_outputs = outputs.values, val_outputs.values

    dictionary = {} #{i:name for (i,name) in enumerate(df.columns)}
    df_labels = pd.read_csv(os.path.join(data_path, 'Key_MACCS.csv'), header=0)
    for i in range(len(df.columns)):
        bit_idx = int(df.columns[i].replace('bit',''))-1
        #print(i, bit_idx, len(df.columns), len(df_labels))
        try: label = df_labels.iloc[bit_idx]['KeyDescription']
        except: print('fail'); continue
        label = label.replace("Is there","Presence of").replace("Are there","Presence of")
        label = label.replace("?","").replace('/',' or ').replace('-', ' ').replace('( ','(').replace(' )',')').replace('_','')
        if latex_format:
            label = split_latex_text(label)
        else:
            label = split_text(label)
        dictionary[i] = label

    return inputs, outputs, val_inputs, val_outputs, dictionary

def get_data_pmlb(dataset, train_ratio=.8, max_points=600, seed=0, latex_format=False, binarize_categorical=True, max_features=100, verbose=False):

    np.random.seed(seed)

    df = fetch_data(dataset, return_X_y=False, local_cache_dir='pmlb_cache')
    outputs, df = df.iloc[:,-1].values, df.drop(df.columns[-1], axis=1)

    if verbose: 
        print(dataset)
        print("Shape : ",df.shape)

    if binarize_categorical:
        # binarize features
        df = binarize_all_features(df)
        if len(df.columns)>max_features:
            print('Too many features, skipping')
            return None, None, None, None, None
    else:
        # remove non binary features
        indices_to_keep = []
        for i, col in enumerate(df.columns):
            if len(df[col].unique())==2: indices_to_keep.append(i)
        df = df.drop(df.columns[np.logical_not(np.isin(np.arange(len(df.columns)), indices_to_keep))], axis=1)
    if verbose: 
        print("Shape after binarization : ",df.shape)
        print("Attributes : ",df.columns)

    inputs = df.values
    shuffle_idx = np.random.permutation(len(inputs)) ; inputs, outputs = inputs[shuffle_idx], outputs[shuffle_idx]
    n_samples = min(max_points,int(train_ratio*len(inputs)))#min(n_points, len(inputs)-20)

    if set(np.unique(outputs)) != {0,1}: # binarize features
        values = np.unique(outputs)
        if not len(values)==2:
            print('Too many output classes, skipping')
            return None, None, None, None, None
        mapping = {values[0]:0, values[1]:1} # then map to new values
        outputs = np.vectorize(mapping.get)(outputs)

    inputs, val_inputs, outputs, val_outputs = inputs[:n_samples].astype(int), inputs[n_samples:].astype(int), outputs[:n_samples].astype(int), outputs[n_samples:].astype(int)

    dictionary = {i:name for (i,name) in enumerate(df.columns)}
    for key, value in dictionary.items():
        if latex_format:
            value = split_latex_text(value)
        else:
            value = split_text(value)
        value = value.replace('_',' ')
        dictionary[key] = value

    return inputs, outputs, val_inputs, val_outputs, dictionary

def get_data_uci(dataset, class_idx, binarize_categorical=False):

    import uci_dataset
    df = getattr(uci_dataset, dataset)()
    outputs, df = df.iloc[:,class_idx].values, df.drop(df.columns[class_idx], axis=1)
    if binarize_categorical:
        df = binarize_all_features(df)
    else:
        # remove non binary features
        indices_to_keep = []
        for i, col in enumerate(df.columns):
            if len(df[col].unique())==2: indices_to_keep.append(i)
        df = df.drop(df.columns[np.logical_not(np.isin(np.arange(len(df.columns)), indices_to_keep))], axis=1)
    
    inputs = df.values
    # drop columns which are not to keep
    print('\n-------------------------------\n')
    print(dataset, inputs.shape, df.columns)

    shuffle_idx = np.random.permutation(len(inputs)) ; inputs, outputs = inputs[shuffle_idx], outputs[shuffle_idx]
    dictionary = {i:name for (i,name) in enumerate(df.columns)}
    n_samples = int(min(1024, .8*len(inputs)))

    for i in range(inputs.shape[1]):
        if set(np.unique(inputs[:,i])) != {0,1}: # binarize features
            values = np.unique(inputs[:,i]); assert len(values)==2, np.unique(inputs)
            mapping = {values[0]:0, values[1]:1} # then map to new values
            inputs[:,i] = np.vectorize(mapping.get)(inputs[:,i])
    values = np.unique(outputs)
    mapping = {values[0]:0, values[1]:1} # then map to new values   
    outputs = np.vectorize(mapping.get)(outputs)

    inputs, val_inputs, outputs, val_outputs = inputs[:n_samples].astype(int), inputs[n_samples:].astype(int), outputs[:n_samples].astype(int), outputs[n_samples:].astype(int)

    return inputs, outputs, val_inputs, val_outputs, dictionary


def binarize_all_features(df):
    # transform each categorical feature into a set of binary features
    
    for col in df.columns:
        num_values = len(df[col].unique())
        if num_values ==2: continue
        if num_values > 5: # drop this feature
            df = df.drop(col, axis=1)
            continue
        binary_features = pd.get_dummies(df[col], prefix=col, prefix_sep='=')
        df = pd.concat([df, binary_features], axis=1)
        df = df.drop(col, axis=1)
    return df

        
def run_models(boolformer_model, inputs, outputs, val_inputs, val_outputs, beam_size=10, n_estimators=1, verbose=False, seed=0):

    accs, f1s = {}, {}
    for method in ['RandomForestClassifier_1', 'RandomForestClassifier_100', 'LogisticRegression', 'Boolformer']:# 'TabPFNClassifier']:
        fit_kwargs, predict_kwargs = {}, {}
        if method == 'Boolformer': 
            clf = boolformer_model
            fit_kwargs = {'beam_size':beam_size}
            predict_kwargs = {'n_estimators':n_estimators}
        elif method.startswith('RandomForestClassifier'):
            n_trees = int(method.split('_')[-1])
            clf = RandomForestClassifier(random_state=seed, n_estimators=n_trees)
        elif method == 'TabPFNClassifier':
            clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=1)
        else:
            clf = eval(method)(random_state=seed)
            
        clf.fit(inputs, outputs, **fit_kwargs)
        preds = clf.predict(val_inputs, **predict_kwargs).squeeze()

        acc, f1 = accuracy_score(val_outputs, preds), f1_score(val_outputs, preds)
        precision = precision_score(val_outputs, preds)
        recall = recall_score(val_outputs, preds)
        #print(method)
        if verbose: print(f"{method} & {acc:.3f} & {f1:.3f} & {precision:.3f} & {recall:.3f}")
        accs[method] = acc
        f1s[method] = f1

    pred_tree = boolformer_model.estimators[0][0]

    if verbose:
        display(boolformer_model.env.simplifier.get_simple_infix(pred_tree, simplify_form='basic'))
    return accs, f1s, pred_tree

def split_text(text, chunk_size=4):
    text = text.split(' ')
    text = [" ".join(text[i:i+chunk_size]) for i in range(0, len(text), chunk_size)]
    text = "\n".join([chunk for chunk in text])
    return text

def split_latex_text(text, chunk_size=2):
    text = text.replace('-',' ')
    text = text.split(' ')
    text = [" ".join(text[i:i+chunk_size]) for i in range(0, len(text), chunk_size)]
    text = " \\\\ ".join(['\\text{'+chunk+'}' for chunk in text])
    text = "\\substack{"+text+"}"  
    return text