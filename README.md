# Boolformer: symbolic regression of Boolean functions with transformers

This repository contains code for the paper Boolformer: symbolic regression of Boolean functions with transformers.

## Installation
This package is installable via pip:

```pip install boolformer```

## Demo 

We include a small notebook that loads a pre-trained model you can play with: `Boolformer_demo.ipynb`.

## Usage

Import the model in a few lines of code:
```python 
from boolformer import load_boolformer
boolformer_noiseless = load_boolformer('noiseless')
boolformer_noisy     = load_boolformer('noisy')
```

Using the model:
```python
import numpy as np
inputs = np.array([  
    [False, False],
    [False, True ],
    [True , False],
    [True , True ],
])
outputs1 = np.array([False, False, False, True])
outputs2 = np.array([True, False, False, True])
inputs = [inputs, inputs]
outputs = [outputs1, outputs2]
pred_trees, errors, complexities = boolformer_noiseless.fit(inputs, outputs, verbose=False, beam_size=10, beam_type="search")

for pred_tree in pred_trees:
    print(pred_tree)
```


## Training and evaluation

To launch a model training with additional arguments (arg1,val1), (arg2,val2):
```python train.py --arg1 val1 --arg2 --val2```

All hyper-parameters related to training are specified in ```train.py```, and those related to the environment are in ```envs/environment.py```.

To retrain the models of the paper, use the scripts in the ```scripts``` folder.
