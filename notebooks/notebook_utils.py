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

from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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

def latex_float(f, precision=3):
    float_str = f"%.{precision}g"%f
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

############################ RECURRENCE ############################

def readable_infix(tree):
    infix = tree.infix()
    infix = infix.replace('x_0_','x').replace('x_1_','y').replace('x_2_','z')
    infix = infix.replace('add','+').replace('sub','-').replace('mod','%')
    infix = infix.replace('mul','*').replace('idiv','/').replace('div','/').replace('fabs','abs')
    infix = infix.replace('inv','1/').replace('euler_gamma', 'g').replace('rand','w')
    return infix

def sympy_infix(tree):
    infix = readable_infix(tree)
    for i in range(6):
        exec('x{}'.format(i)+'='+"symbols('x{}'.format(i))")
        exec('y{}'.format(i)+'='+"symbols('y{}'.format(i))")
        exec('z{}'.format(i)+'='+"symbols('z{}'.format(i))")
    n, w, e, g = symbols('n w e g')
    init_printing(use_unicode=True)
    return simplify(eval(infix))

def read_run(path):
    
    run = {}
    args = pickle.load(open(path+'/params.pkl', 'rb'))
    run['args'] = args
    if 'use_sympy' not in args:
        setattr(args,'use_sympy',False)
    if 'mantissa_len' not in args:
        setattr(args,'mantissa_len',1)
    if 'train_noise' not in args:
        setattr(args,'train_noise', 0)
    setattr(args, 'extra_constants', '')
    run['logs'] = []
    run['num_params'] = []
    logfile = path+'/train.log'
    f = open(logfile, "r")
    for line in f.readlines():
        if '__log__' in line:
            log = eval(line[line.find('{'):].rstrip('\n'))
            if not run['logs']: run['logs'].append(log)
            else: 
                if log['valid1_recurrence_beam_acc'] != run['logs'][-1]['valid1_recurrence_beam_acc']: run['logs'].append(log)
    f.close()
    args.output_dir = Path(path)
    return run
    
def load_run(params, new_params={}, epoch=None):
    
    #try: del boolformer
    #except: pass
    #path = '/private/home/sdascoli/recur/src'

    final_params = params
    for arg, val in new_params.items():
        setattr(final_params,arg,val)
    final_params.multi_gpu = False
        
    #path = final_params.dump_path+'/src'
    #src = import_file(path)
    sys.path.append(os.path.join(BASE_PATH, 'boolean'))
    import src

    print(src)
    from src.model import build_modules, Boolformer
    from src.envs import ENVS, build_env
    from src.trainer import Trainer
    from src.evaluator import Evaluator, idx_to_infix, calculate_error
    from src.envs.generators import RandomRecurrence
    
    env = build_env(final_params)
    modules = build_modules(env, final_params)
        
    trainer = Trainer(modules, env, final_params)
    evaluator = Evaluator(trainer)
    
    if epoch is not None:
        print(f"Reloading epoch {epoch}")
        checkpoint_path = os.path.join(final_params.dump_path,f'periodic-{epoch}.pth')
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        new_checkpoint = {}
        for k, module in modules.items():
            weights = {k.partition('.')[2]:v for k,v in checkpoint[k].items()}
            module.load_state_dict(weights)
            module.eval()

    embedder = modules["embedder"]
    encoder = modules["encoder"]
    decoder = modules["decoder"]
    embedder.eval()
    encoder.eval()
    decoder.eval()

    boolformer = Boolformer(
        env=env,
        embedder=embedder,
        encoder=encoder,
        decoder=decoder,
    )
    
    return env, modules, trainer, evaluator, boolformer

def eval_run(run, new_args=None):
    
    env, modules, trainer, evaluator = load_run(run, new_args=new_args)
    data_types = ["valid1"]
    evaluator.set_env_copies(data_types)
    scores = evaluator.run_all_evals(data_types)   
    return scores

      
def predict(env, modules, inputs, outputs, 
            verbose=False, 
            store_attention=False, 
            sort_by='error', 
            beam_size=1, 
            beam_type='search', 
            beam_temperature=.1, 
            complexity_metric='n_ops_arbitrary_fan_in'):
    
    encoder, decoder, embedder = modules["encoder"], modules["decoder"],  modules["embedder"]
       
    if hasattr(outputs, 'shape'):
        if len(outputs.shape)==1:
            inputs, outputs = [inputs], [outputs]
    elif not isinstance(inputs, list) or not isinstance(outputs, list):
        inputs, outputs = [inputs], [outputs]
        
    x, x_len = embedder.batch(inputs, outputs)
    x = embedder(x)

    if not env.params.cpu:
        x, x_len = x.cuda(), x_len.cuda()
    encoded = encoder("fwd", x=x, lengths=x_len, causal=False, store_outputs=store_attention).transpose(0,1)
    bs = encoded.shape[0]
    if beam_type == 'search':
        _, _, gens = decoder.generate_beam(
                    encoded,
                    x_len,
                    beam_size=beam_size,
                    length_penalty=env.params.beam_length_penalty,
                    early_stopping=env.params.beam_early_stopping,
                    max_len = env.params.max_output_len)
        gens = [sorted([hyp for (_, hyp) in gens[i].hyp], key=lambda s: s[0], reverse=True) for i in range(bs)]
    elif beam_type == 'sampling':
        num_samples = beam_size
        encoded = encoded.unsqueeze(1).expand((bs, num_samples) + encoded.shape[1:]).contiguous().view((bs * num_samples,) + encoded.shape[1:])
        x_len = x_len.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)
        gens, _, _ = decoder.generate(
                    encoded,
                    x_len,
                    sample_temperature=beam_temperature)
        gens = gens.unsqueeze(-1).view(gens.shape[0], bs, num_samples).transpose(0, 1).transpose(1, 2).cpu()#.tolist()   
    else:
        raise
    
    best_trees, best_errors, best_complexities = [], [], []
    
    for problem_idx, gen in enumerate(gens):

        ratio = sum(outputs[problem_idx])/len(outputs[problem_idx])
        if ratio in [0,1]:
            if ratio == 1:
                best_trees.append(env.output_encoder.decode(['True']))
            else:
                best_trees.append(env.output_encoder.decode(['False']))
            best_errors.append(0)
            best_complexities.append(0)
            continue

        pred_trees, errors, complexities = [], [], []
        for hyp in gen:
            tokens = [env.output_id2word[wid] for wid in hyp.tolist()[1:]]
            pred_tree = env.output_encoder.decode(tokens)
            if pred_tree is None: continue
            pred_tree = env.simplifier.simplify_tree(pred_tree, env.params.simplify_form)
            pred_tree.simplify()
            #pred_tree.to_arbitrary_fan_in()
            pred_trees.append(pred_tree)
            pred = pred_tree(inputs[0]).flatten()
            true = outputs[problem_idx]
            error = 1.-sum(pred==true)/len(pred)
            errors.append(error)
            if complexity_metric == 'n_ops_arbitrary_fan_in':
                complexity = pred_tree.get_n_ops_arbitrary_fan_in()
            elif complexity_metric == 'n_ops':
                complexity = pred_tree.get_n_ops()
            else:
                complexity = len(pred_tree.get_variables())
            complexities.append(complexity)

        # sort by error and complexity
        if sort_by=='error':
            idx = sorted(range(len(errors)), key=lambda k: (errors[k], complexities[k]))
        else:
            idx = sorted(range(len(errors)), key=lambda k: (complexities[k], errors[k]))
        pred_trees = [pred_trees[i] for i in idx]
        errors = [errors[i] for i in idx]
        complexities = [complexities[i] for i in idx]
        if verbose:
            print('Errors      ', '   '.join(['{:.3f}'.format(error) for error in errors]))
            print('Complexities', '   '.join(['{:>5d}'.format(complexity) for complexity in complexities]))
        if pred_trees:
            best_trees.append(pred_trees[0])
            best_errors.append(errors[0])
            best_complexities.append(complexities[0])
        else:
            best_trees.append(None)
            best_errors.append(None)
            best_complexities.append(None)
    
    return best_trees, best_errors, best_complexities

def get_logical_circuit(function_name='multiplexer'):
    to_bin = lambda x : int(''.join([str(int(digit)) for digit in x]),2)
    to_decimal = lambda x : sum(x[::-1].astype(int)*2**np.arange(n_vars//2))

    if function_name.startswith('multiplexer'):
        order=int(function_name[-1])
        n_vars = order + 2**order
        tree = lambda x : x[order:][sum(x[:order]*2**np.arange(order)).astype(int)]
    if function_name.startswith('majority'):
        n_vars=int(function_name[-1])
        tree = lambda x : sum(x)>=n_vars/2
    elif function_name.startswith('parity'):
        n_vars=int(function_name[-1])
        tree = lambda x : 1-sum(x)%2
    elif function_name.startswith('all') or function_name.startswith('any'):
        name, n_vars = function_name.split('_')[0], int(function_name.split('_')[1])
        tree = lambda x : getattr(np,name)(x)
    elif function_name.startswith("comparator"):
        n_vars = 2*int(function_name[-1])
        def tree(x):
            factor1, factor2 = to_decimal(x[:n_vars//2]), to_decimal(x[n_vars//2:])
            return 1 if factor1>factor2 else 0
    elif function_name.startswith("multiadder"):
        n_vars = int(function_name[-1])
        def tree(x):
            tmp = sum(x.astype(int))
            n_output = int(np.log2(n_vars))+1
            return ['{:0{n_output}b}'.format(tmp, n_output=n_output)[-i] for i in range(n_output)]
    elif function_name.startswith("adder"):
        n_vars = 2*int(function_name[-1])
        def tree(x):
            factor1, factor2 = to_bin(x[:n_vars//2]), to_bin(x[n_vars//2:])
            n_output = n_vars//2+1
            return [bin(factor1+factor2)[2:].zfill(n_output)[-i] for i in range(n_output)]
    elif function_name.startswith("multiplier"):
        n_vars = 2*int(function_name[-1])
        def tree(x):
            factor1, factor2 = to_bin(x[:n_vars//2]), to_bin(x[n_vars//2:])
            n_output = n_vars
            return [bin(factor1*factor2)[2:].zfill(n_output)[-i] for i in range(n_output)]
    elif function_name.startswith("decoder"):
        n_vars = int(function_name[-1])
        def tree(x):
            index = sum(x.astype(int)*2**np.arange(n_vars))
            #index = 4*int(x[0])+2*int(x[1])+int(x[2])
            tmp = [False for i in range(2**n_vars)]
            tmp[index] = True
            return tmp
    elif function_name.startswith("random"):
        n_vars = int(function_name[-1])
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
    truth_table = truth_table[np.random.choice(truth_table.shape[0], size=truth_table.shape[0], replace=False)]

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

def plot_attention(args, env, modules, inputs, outputs):
    
    encoder, decoder = modules["encoder"], modules["decoder"]
    encoder.eval()
    encoder.STORE_OUTPUTS = True
    num_heads = encoder.n_heads
    num_layers = encoder.n_layers
    
    new_args = copy.deepcopy(args)
    new_args.series_length = 15
    pred_trees, error_arr, complexity_arr = predict(env, modules, inputs, outputs, verbose=False, beam_size=10, store_attention=True)
        
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

def getTargetGenesEvalExpressions(bool_expressions):  
	target_genes = [] 
	eval_expressions = []  
	for k in range(0, len(bool_expressions)):  
		expr = bool_expressions[k]   
		gene_num = int(re.search(r'\d+', expr[:expr.find(" = ")]).group())
		eval_expr =  expr[expr.find("= ") + 2:]
		target_genes.append(gene_num)   
		eval_expressions.append(eval_expr) 
	return target_genes, eval_expressions

def getBooleanExpressions(model_path):
	bool_expressions = []
	with open(model_path) as f:
		bool_expressions = [line.replace("!"," not ").replace("&"," and ").replace("||", " or ").strip() for line in f]  
	return bool_expressions 

def evalBooleanModel(model_path, test_series): 
    rows, columns = test_series.shape 
    simulations = test_series.iloc[[0]].copy()  #set initial states          
    bool_expressions = getBooleanExpressions(model_path)       
    target_genes, eval_expressions = getTargetGenesEvalExpressions(bool_expressions)        

	#intialize genes to false
    for k in range(0, columns):   
        gene_num = k + 1    
        exec("Gene" + str(gene_num) + " = False", globals())

    for time_stamp in range(1, rows):
		#dynamically allocate variables  
        for k in range(0, len(target_genes)):    
            gene_num = target_genes[k]   
            exec("Gene" + str(gene_num) + " = " + str(simulations.iat[time_stamp - 1, gene_num - 1]))    
		
		#initialize simulation to false  
        ex_row = [0]*columns   
		#evaluate all expression  
        for k in range(0, len(bool_expressions)):      
            gene_num = target_genes[k]   
            eval_expr = eval_expressions[k]     
            #print(eval_expr, eval(eval_expr))
            ex_row[gene_num - 1] = int(eval(eval_expr))	 
        simulations = simulations._append([ex_row], ignore_index = True)    

    erros = simulations.sub(test_series) 
    return np.absolute(erros.to_numpy()).sum()  

def run_benchmark(env, modules, verbose=True, network_size=16, stop_at=-1,  beam_size=1, organism='Ecoli', model_name='Boolformer', max_points=None, batch_size = 4):
    base_path = os.path.join(BASE_PATH, 'reviewAndAssessment')
    network_size = network_size
    network_num = 10
    data_path = os.path.join(base_path,'results',organism,str(network_size))
    results_method_path = os.path.join(data_path, model_name)
    for network_id in tqdm.tqdm(range(1, network_num+1)):
        data_file = organism + "-" + str(network_id) + "_dream4_timeseries.tsv" 
        df = pd.read_csv(os.path.join(data_path,data_file), sep='\t', header=None)

        rows, columns = df.shape  
        seriesSize = rows    
        test_size = 56         
        crossIterations = int(seriesSize/test_size) 
        dynamic_errors, execution_times = [], []

        variable_counts = defaultdict(int)

        for series_id in range(crossIterations):
            drop_rows = range(series_id*test_size, min((series_id + 1)*test_size, seriesSize))    
            test_series = df.iloc[drop_rows]    
            test_series = test_series.reset_index(drop=True)           
            infer_series = df.drop(drop_rows)     
            infer_series = infer_series.reset_index(drop=True)     
            #test_series, infer_series = infer_series, test_series

            n_vars = len(infer_series.columns)

            inputs = infer_series.values[None,:,:].repeat(n_vars, axis=0)
            outputs = np.array([inputs[var, 1:, var] for var in range(n_vars)])
            #inputs = np.array([np.concatenate((inputs[var,:,:var],inputs[var,:,var+1:]), axis=-1) for var in range(n_vars)])
            for var in range(n_vars):
                inputs[var,:,var] = np.random.choice([0,1], size=inputs[var,:,var].shape, p=[0.5, 0.5])
            inputs = inputs[:, :-1, :]
            if max_points is not None:
                #indices = np.random.choice(range(inputs.shape[1]), max_points, replace=False)
                #inputs, outputs = inputs[:,indices,:], outputs[:,indices]
                inputs, outputs = inputs[:,:max_points,:], outputs[:,:max_points]
            val_inputs = test_series.values[None,:,:].repeat(n_vars, axis=0)
            val_outputs = np.array([val_inputs[var, 1:, var] for var in range(n_vars)])
            val_inputs = val_inputs[:, :-1, :]
            num_datasets = len(inputs)
            num_batches = num_datasets//batch_size
            
            start = time.time()  
            pred_trees, error_arr, complexity_arr = [], [], []   
            for batch in range(num_batches):
                inputs_, outputs_ = inputs[batch*batch_size:(batch+1)*batch_size], outputs[batch*batch_size:(batch+1)*batch_size]
                pred_trees_, error_arr_, complexity_arr_ = predict(env, 
                                                                modules, 
                                                                inputs_, 
                                                                outputs_, 
                                                                verbose=False, 
                                                                beam_size=beam_size,
                                                                sort_by='error')
                pred_trees.extend(pred_trees_), error_arr.extend(error_arr_), complexity_arr.extend(complexity_arr_)
            end = time.time()
            elapsed = (end - start)/num_datasets

            test_error_arr = []
            for iout, pred_tree in enumerate(pred_trees):
                if pred_tree is None: 
                    test_error_arr.append(.5)
                    continue
                preds = pred_tree(val_inputs[iout])
                test_error = 1.-sum(preds==val_outputs[iout])/len(preds)
                test_error_arr.append(test_error)

            if verbose: 
                try:
                    print(f"AVG Error, test error: {np.nanmean(error_arr)}, {np.nanmean(test_error_arr)}")
                except: print('error')

            dynamics_path = os.path.join(results_method_path, organism + "-" + str(network_id) + "_" + str(series_id) + "_dynamics.tsv")
            structure_path = os.path.join(results_method_path, organism + "-" + str(network_id) + "_" + str(series_id) + "_structure.tsv") 
            # make directory if it doesn't exist
            if not os.path.exists(os.path.dirname(dynamics_path)):
                os.makedirs(os.path.dirname(dynamics_path))
            if not os.path.exists(os.path.dirname(structure_path)):
                os.makedirs(os.path.dirname(structure_path))
            dynamics_file = open(dynamics_path, 'w')
            structure_file = open(structure_path, 'w')
            for idx, pred_tree in enumerate(pred_trees):
                pred_tree.increment_variables()
                used_variables = pred_tree.get_variables()
                for var in used_variables:
                    variable_counts[var] += 1
                line = f'Gene{idx+1} = {pred_tree.infix()}' 
                line = line.replace('x_', 'Gene').replace('and', '&').replace('or', '||').replace('not', '!')
                line += '\n'
                dynamics_file.write(line)
                for var in used_variables:
                    var_idx = int(var.split('_')[-1])
                    influence = f'{idx+1} <- {var_idx}' + '\n'
                    structure_file.write(influence)
            dynamics_file.close()
            structure_file.close()

            try:
                errs = evalBooleanModel(dynamics_path, test_series)
            except:
                errs = np.inf
            dynamic_errors.append(errs)
            execution_times.append(elapsed)

            if network_id==stop_at: return pred_trees, inputs, outputs

        # print top 10 variables sorted by count
        print(sorted(variable_counts.items(), key=lambda x : -x[1])[:10])

        rslt_df = pd.DataFrame(list(zip(execution_times, dynamic_errors)), columns=["time", "errors"])  
        results_file = os.path.join(results_method_path, "results_network_" + str(network_id) + ".tsv") 
        rslt_df.to_csv(results_file, index=False, sep="\t", float_format='%.2f')

def run_drug_discovery(model, data_path='.', problem="TOX", num_points=500, num_test_points=500, num_features=None, beam_size=10, verbose=True, balance=True):

    env = model.env

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
    df = pd.read_csv(os.path.join(data_path, dataset_name))
    df = df.drop('smiles', axis=1)

    #num_total = num_points + num_test_points
    num_test_points = min(num_test_points, len(df)-num_points)
    while True:
        df = df.sample(frac=1)
        val_df = df[:num_test_points]
        if len(val_df[val_df['activity']==1])>10: break
    df = df.drop(val_df.index)
    # printnumber of positives
    #print(len(df[df['activity']==1]), len(val_df[val_df['activity']==1]))
    # rebalance data if necessary
    if balance:
        num_positives = num_points//2 #min(num_points//2, len(df[df['activity']==1]))
        num_negatives = num_points - num_positives
        positives = df[df['activity']==1].sample(n=num_positives, replace=True)
        negatives = df[df['activity']==0].sample(n=num_negatives, replace=True)
        df = pd.concat([positives, negatives])

    outputs, val_outputs = df['activity'], val_df['activity']
    # sort features
    df = df.drop('activity', axis=1)
    val_df = val_df.drop('activity', axis=1)
    feature_order = df.var().sort_values(ascending=False).index
    df = df.reindex(feature_order, axis=1)
    val_df = val_df.reindex(feature_order, axis=1)

    dictionary = {i:name for (i,name) in enumerate(df.columns)}
    inputs, val_inputs = df.values, val_df.values
    outputs, val_outputs = outputs.values, val_outputs.values

    accs, f1s = {}, {}
    for method in ['GradientBoostingClassifier', 'LogisticRegression']:
        clf = eval(method)(random_state=0).fit(inputs, outputs)
        preds = clf.predict(val_inputs)
        acc, f1 = accuracy_score(val_outputs, preds), f1_score(val_outputs, preds)
        precision = precision_score(val_outputs, preds)
        recall = recall_score(val_outputs, preds)
        #print(method)
        if verbose: print(f"{acc:.3f} & {f1:.3f} & {precision:.3f} & {recall:.3f}")
        accs[method] = acc
        f1s[method] = f1

    # rank features by importance
    #if data.shape[1]>1000:
    #    index = abs(clf.coef_[0]).argsort()[::-1]
    #    data, val_data = data[:,index], val_data[:,index]

    inputs, val_inputs = inputs[:,:num_features], val_inputs[:,:num_features]

    pred_trees, error_arr, complexity_arr = model.predict(inputs, outputs, verbose=False, beam_size=beam_size, sort_by='error')
    pred_tree = pred_trees[0]
    preds = pred_tree.val(val_inputs)
    targets = val_outputs
    acc = sum(preds==targets)/len(preds)
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    accs['Boolformer'] = acc
    f1s['Boolformer'] = f1

    pred_tree.relabel_variables(dictionary)
    #print(f"Accuracy {1-error_arr[0]}")
    if verbose:
        print(f"{acc:.3f} & {f1:.3f} & {precision:.3f} & {recall:.3f}")
        display(env.simplifier.get_simple_infix(pred_trees[0], simplify_form='basic'))
        fancy_tree = tree_to_latex(pred_tree)
        fancy_tree = format_drug_discovery(fancy_tree, key_path = os.path.join(data_path, 'Key_MACCS.csv'), problem=problem)
        print(fancy_tree)

    return accs, f1s, pred_tree

def tree_to_latex(tree, problem=None):
    bool_text = tree.forest_prefix()
    return bool_text

def format_drug_discovery(bool_text, key_path, problem=None):

    if problem == "BBBP":
        bool_text = bool_text.replace("y_0", "\\text{Can pass the BBB}")
    elif problem == "HIV":
        bool_text = bool_text.replace("y_0", "\\text{Anti-HIV}")
    elif problem == "TOX":
        bool_text = bool_text.replace("y_0", "\\text{Toxic}")
    else:
        pass
    bits = re.findall(r'bit\d+', bool_text)

    df1 = pd.read_csv(key_path, header=0) 
    df1.head()

    for bit in bits:
        bit = bit.replace("bit","")
        bit = int(bit)
        new_text = df1[df1["KeyID"]==bit].values[0][1]
        new_text = new_text.replace("Is there","Presence of").replace("Are there","Presence of")
        new_text = new_text.replace("?","").replace('/',' or ').replace('-', ' ').replace('( ','(').replace(' )',')').replace('_','')
        new_text = split_latex_text(new_text)
        bool_text = bool_text.replace(f'bit{bit}', new_text)

    return bool_text

def split_latex_text(text):
    text = text.split(' ')
    chunk_size = 2
    text = [" ".join(text[i:i+chunk_size]) for i in range(0, len(text), chunk_size)]
    text = " \\\\ ".join(['\\text{'+chunk+'}' for chunk in text])
    text = "\\substack{"+text+"}"  
    return text

def run_pmlb(env, modules, dataset, beam_size=20, n_points=500):
    df = fetch_data(dataset, return_X_y=False, local_cache_dir='pmlb_cache')
    # remove non binary features
    indices_to_keep = []
    for i, col in enumerate(df.columns[:-1]):
        if len(df[col].unique())==2: indices_to_keep.append(i)
    df = df.drop(df.columns[:-1][np.logical_not(np.isin(np.arange(len(df.columns[:-1])), indices_to_keep))], axis=1)
    dictionary = {i:name for (i,name) in enumerate(df.columns)}

    inputs, outputs = fetch_data(dataset, return_X_y=True, local_cache_dir='pmlb_cache')
    shuffle_idx = np.random.permutation(len(inputs)) ; inputs, outputs = inputs[shuffle_idx], outputs[shuffle_idx]
    print('\n-------------------------------\n')
    print(dataset, inputs.shape, df.columns)
    n_samples = min(n_points, len(inputs)-20)

    inputs = inputs[:,indices_to_keep]
    if set(np.unique(inputs)) != {0,1}: # binarize features
        values = np.unique(inputs); assert len(values)==2
        mapping = {values[0]:0, values[1]:1} # then map to new values
        inputs = np.vectorize(mapping.get)(inputs)

    inputs, val_inputs, outputs, val_outputs = inputs[:n_samples].astype(int), inputs[n_samples:].astype(int), outputs[:n_samples].astype(int), outputs[n_samples:].astype(int)
    print(inputs.shape)
    #inputs, outputs, val_inputs, val_outputs = [inputs], [outputs], [val_inputs], [val_outputs]

    accs, f1s = {}, {}
    for method in ['GradientBoostingClassifier', 'LogisticRegression', 'MLPClassifier''']:
        clf = eval(method)(random_state=0).fit(inputs, outputs)
        preds = clf.predict(val_inputs)
        acc, f1 = accuracy_score(val_outputs, preds), f1_score(val_outputs, preds)
        precision = precision_score(val_outputs, preds)
        recall = recall_score(val_outputs, preds)
        accs[method] = acc
        f1s[method] = f1

        try: pred_trees, error_arr, complexity_arr = predict(env, modules, inputs, outputs, verbose=False, beam_size=beam_size)
        except: import traceback; traceback.print_exc(); return

    pred_tree = pred_trees[0]
    preds = pred_tree.val(val_inputs)
    targets = val_outputs
    acc = sum(preds==targets)/len(preds)
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    accs['Boolformer'] = acc
    f1s['Boolformer'] = f1

    print(f1s)
    for pred_tree in pred_trees:
        display(env.simplifier.get_simple_infix(pred_tree, simplify_form='basic'))
        for key, value in dictionary.items():
            dictionary[key] = split_latex_text(value)
        pred_tree.relabel_variables(dictionary)
        dataset = dataset.replace('_','')
        print(r'\begin{forest}')
        print('[$\\text{'+dataset+'}$ '+tree_to_latex(pred_tree)+']')
        f1_1, f1_2, f1_3 = f1s['Boolformer'], f1s['LogisticRegression'], f1s['GradientBoostingClassifier']
        print(r'\end{forest}')
        print('\\caption{'+dataset
              +'. F1: '+f'{f1_1:.3f}.'
              +' LogReg: '+f'{f1_2:.3f}.'
              +' GradientBoost: '+f'{f1_3:.3f}.'
              +'}')

def run_uci(env, modules, dataset):
    import uci_dataset
    df = uci_dataset.load_early_stage_diabetes_risk()
    inputs, outputs = df.iloc[:,:-1].values, df.iloc[:,-1].values
    # remove non binary features
    indices_to_keep = []
    for i, col in enumerate(df.columns[:-1]):
        if len(df[col].unique())==2: indices_to_keep.append(i)
    # drop columns which are not to keep
    df = df.drop(df.columns[:-1][np.logical_not(np.isin(np.arange(len(df.columns[:-1])), indices_to_keep))], axis=1)

    shuffle_idx = np.random.permutation(len(inputs)) ; inputs, outputs = inputs[shuffle_idx], outputs[shuffle_idx]
    dictionary = {i:name for (i,name) in enumerate(df.columns)}
    print('\n-------------------------------\n')
    print(dataset, inputs.shape, df.columns)
    n_samples = min(500, len(inputs)-20)

    inputs = inputs[:,indices_to_keep]
    print(indices_to_keep)
    for col in range(inputs.shape[1]):
        mapping = {'Yes':1, 'No':0, 'Male':1, 'Female':0}
        inputs[:,col] = np.vectorize(mapping.get)(inputs[:,col])
    mapping = {'Negative':0, 'Positive':1}
    outputs = np.vectorize(mapping.get)(outputs)

    inputs, val_inputs, outputs, val_outputs = inputs[:n_samples].astype(int), inputs[n_samples:].astype(int), outputs[:n_samples].astype(int), outputs[n_samples:].astype(int)
    print(inputs.shape)
    #inputs, outputs, val_inputs, val_outputs = [inputs], [outputs], [val_inputs], [val_outputs]

    accs, f1s = {}, {}
    for method in ['GradientBoostingClassifier', 'LogisticRegression', 'MLPClassifier''']:
        clf = eval(method)(random_state=0).fit(inputs, outputs)
        preds = clf.predict(val_inputs)
        acc, f1 = accuracy_score(val_outputs, preds), f1_score(val_outputs, preds)
        precision = precision_score(val_outputs, preds)
        recall = recall_score(val_outputs, preds)
        accs[method] = acc
        f1s[method] = f1

        pred_trees, error_arr, complexity_arr = predict(env, modules, inputs, outputs, verbose=False, beam_size=20)

    pred_tree = pred_trees[0]
    preds = pred_tree.val(val_inputs)
    targets = val_outputs
    acc = sum(preds==targets)/len(preds)
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    accs['Boolformer'] = acc
    f1s['Boolformer'] = f1

    print(f1s)
    for pred_tree in pred_trees:
        display(env.simplifier.get_simple_infix(pred_tree, simplify_form='basic'))
        for key, value in dictionary.items():
            if value=="Gender": value = "Male"
            dictionary[key] = split_latex_text(value)
        pred_tree.relabel_variables(dictionary)
        dataset = dataset.replace('_','')
        print('[.$\\text{'+dataset+'}$ '+tree_to_latex(pred_tree)+']')
        f1_1, f1_2 = f1s['Boolformer'], f1s['LogisticRegression']
        print('\\caption{'+dataset
                +'. F1: '+f'{f1_1:.3f}.'
                +' Logistic regression: '+f'{f1_2:.3f}'
                +'.}')


if __name__ == "__main__":

    def get_most_free_gpu():
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
        free_memory = [int(x) for x in output.decode().strip().split('\n')]
        most_free = free_memory.index(max(free_memory))
        # set visible devices to the most free gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(most_free)
        print(f"Running on GPU {most_free}")
    get_most_free_gpu()

    parser = argparse.ArgumentParser(description='Boolformer')
    parser.add_argument('--organism', type=str, default='Ecoli')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--network_size', type=int, default=16)
    parser.add_argument('--model_name', type=str, default='Boolformer')
    benchmark_args = parser.parse_args()

    #path = os.path.join(BASE_PATH, "boolean/experiments/bnet/exp_max_flip_prob_0.1_max_inactive_vars_60/")
    #path = os.path.join(BASE_PATH, "boolean/experiments/bnet_traj/exp_enc_emb_dim_512_n_enc_layers_4/")
    paths = glob.glob(os.path.join(BASE_PATH, "boolean/experiments/bnet_hard/*"))[:1]
    for i, path in enumerate(paths):
        
        model_name = benchmark_args.model_name
        if len(paths)>1: model_name+=f"_{i}"
        args = pickle.load(open(os.path.join(path,'params.pkl'), 'rb'))
        new_args =  {
            'eval_size':0,
            'dump_path':args.dump_path.replace('/sb_u0621_liac_scratch',BASE_PATH),
        }
        env, modules, trainer, evaluator = load_run(args, new_args)
        run_benchmark(env, modules, 
                    network_size=benchmark_args.network_size,
                    beam_size=benchmark_args.beam_size, 
                    organism=benchmark_args.organism,
                    model_name=model_name,
                    )