[![PyPI](https://img.shields.io/pypi/v/boolformer.svg)](
https://pypi.org/project/boolformer/)
[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/github/sdascoli/boolformer/blob/main/Boolformer_demo.ipynb)


# Boolformer: symbolic regression of Boolean functions with transformers

This repository contains code for the paper [Boolformer: symbolic regression of Boolean functions with transformers](https://arxiv.org/pdf/2309.12207.pdf).

## Installation
This package is installable via pip:

```pip install boolformer```

## Demo 

We include a small notebook that loads a pre-trained model you can play with here:

[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/github/sdascoli/boolformer/blob/main/Boolformer_demo.ipynb)

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

All hyper-parameters related to training are specified in ```parsers.py```, and those related to the environment are in ```envs/environment.py```.

## Citation

If you want to reuse this material, please considering citing the following:
```
@misc{dascoli2023boolformer,
      title={Boolformer: Symbolic Regression of Logic Functions with Transformers}, 
      author={Stéphane d'Ascoli and Samy Bengio and Josh Susskind and Emmanuel Abbé},
      year={2023},
      eprint={2309.12207},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

This repository is licensed under MIT licence.
