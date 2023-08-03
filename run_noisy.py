import subprocess
from subprocess import DEVNULL
import os
import re
from time import sleep
import itertools
from pathlib import Path
import shutil
from distutils import dir_util
#user = os.getlogin()

exp_folder = 'bnet_hard'

#dump_path = f'/home/{user}/odeformer/experiments'
dump_path = f'/sb_u0621_liac_scratch/boolean/experiments'
Path(dump_path).mkdir(exist_ok=True)

extra_args = {
    "batch_size":256,
    "use_wandb": True,
    #"enc_emb_dim":512,
    #"dec_emb_dim":512,
    #"max_active_vars":12,
    "max_ops":15,
    "min_active_vars":1,
    "min_points":30,
    "max_points":300,
    "input_truth_table":False,
    "max_flip_prob":0.1,
    "enc_emb_dim":512,
    "n_enc_layers":8,
    #"max_vars":20,
    #"max_ops":20,
    #"max_points":200,
    #"max_active_vars":6,
    #"max_inactive_vars":80,
    }

grid = {
    #"max_flip_prob":["0.","0.1"],#, "not,and,or,xor"],
    "max_inactive_vars":["80", "120"],
    "max_active_vars":["6","10"],
    #"n_enc_layers":["4"],
    #"operators_to_use":["not,and,or", "not,and,or,xor"],
    #"max_active_vars": ["10","20","30"],
    #"max_points":[800,400],
    #"dec_emb_dim":[512,1024]
    #"n_enc_layers":[4,8],
    #"n_dec_layers":[4,8],
    #"max_points":[200, 400],
    # "max_vars":[10,20],
    # "max_ops":[10,20],
    #"float_descriptor_length":[3]
    #'use_sympy':[True],
    #"masked_output":[0,0.3,0.6],
    #"sign_as_token":[False],
    #"use_two_hot":[False]
    #"fixed_init_scale":[True,False],
    # "ode_integrator": ["odeint","solve_ivp"],
    # "max_dimension":[2,4]
    #"use_cross_attention":[True,False],
    #"enc_positional_embeddings": ["none","learnable"],
    #"optimizer": ['adam_cosine,warmup_updates=5000,init_period=50000,period_mult=1.5,lr_shrink=0.5'],
}

def get_free_gpus():
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
    free_memory = [int(x) for x in output.decode().strip().split('\n')]
    free_gpus = [i for i, memory in enumerate(free_memory) if memory > 10000]  # Change the threshold based on your needs
    free_gpus = sorted(free_gpus, key=lambda i: free_memory[i], reverse=True)
    return free_gpus

def get_most_free_gpu():
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
    free_memory = [int(x) for x in output.decode().strip().split('\n')]
    most_free = free_memory.index(max(free_memory))
    # set visible devices to the most free gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(most_free)
    return free_gpus

# Get the list of free GPUs
free_gpus = get_free_gpus()
print("Free GPUs: ",free_gpus)
if not free_gpus:
    print("No free GPUs available!")
    exit()

# Path to your PyTorch script
pytorch_script = "train.py"

# Function to run the PyTorch script with a specific learning rate on a specific GPU
def run_experiment(gpu_id, args, logfile):
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command = ["python", pytorch_script]
    for arg, value in args.items():
        command.append(f"--{arg}")
        command.append(str(value))
    with open(logfile, 'a') as f:
        subprocess.Popen(command, env=env_vars, stdout=DEVNULL, stderr=f)

def dict_product(d):
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))

for params in dict_product(grid):
    if not free_gpus:
        break
    exp_id = 'exp_'+'_'.join(['{}_{}'.format(k,v) for k,v in params.items()])
    params['dump_path'] = dump_path
    params['exp_name'] = exp_folder
    params['exp_id'] = exp_id

    #params['dec_emb_dim'] = params['enc_emb_dim']
    #params['n_dec_layers'] = params['n_enc_layers']
    
    job_dir = Path(os.path.join(dump_path, exp_folder, exp_id))
    job_dir.parent.mkdir(exist_ok=True)
    job_dir.mkdir(exist_ok=True)

    for arg, value in extra_args.items():
        if arg not in params:
            params[arg] = value

    for f in os.listdir():
        if f.endswith('.py'):
            shutil.copy2(f, job_dir)
    dir_util.copy_tree('src', os.path.join(job_dir,'src'))
    os.chdir(job_dir)

    logfile = os.path.join(job_dir,'train.log')
    gpu_id = free_gpus.pop(0)
    print(f"Starting experiment {exp_id} on GPU: {gpu_id}")
    run_experiment(gpu_id, params, logfile)
    sleep(1)

print("All experiments started.")
