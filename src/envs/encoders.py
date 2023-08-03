from abc import ABC, abstractmethod
import numpy as np
import math
from .generators import all_operators, Node

def to_base(val, base=10):
    assert(val>=0 and base > 1)
    if val == 0:
        return ['0']
    res = []
    v = val
    while v > 0:
        res.append(str(v % base))
        v = v // base
    return res[::-1]

class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self, params):
        pass

    @abstractmethod
    def encode(self, val):
        pass

    @abstractmethod
    def decode(self, lst):
        pass

class Boolean(Encoder):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.symbols = ['0','1','|']

    def encode(self, data):
        #res = []
        # if len(data.shape)==2:
        #     for sample in data:
        #         for var in sample:
        #             res.append(str(var))
        #         res.append('|')
        #     return res[:-1] # remove final separator
        #elif len(data.shape)==1:
        data = np.array(data, dtype=bool)
        if len(data.shape)==0:
                return str(int(data))
        elif len(data.shape)==1:
            return [str(int(x)) for x in data]
        else:
            raise ValueError("Boolean encoder expects 1D or 0D data")
            

    def decode(self, data):
        # if '|' in data:
        #     res = [[]]
        #     for val in data:
        #         if val=='|':
        #             res.append([])
        #         else:
        #             res[-1].append(bool(eval(val)))
        #     return np.array(res, dtype=bool)
        # else:
        return [bool(eval(val)) for val in data]


class Equation(Encoder):
    def __init__(self, params, symbols):
        super().__init__(params)
        self.params = params
        self.symbols = symbols
        self.operators = all_operators

    def encode(self, tree):
        res = []
        for elem in tree.prefix().split(','):
            try:
                val=float(elem) 
                if (val).is_integer():
                    res.extend(self.write_int(int(elem)))
                else:
                    res.append("OOD_constant")
            except ValueError:
                res.append(elem)
        return res

    def _decode(self, lst):
        if len(lst)==0:
            return None, 0
        if (lst[0] not in self.symbols) and (not lst[0].lstrip('-').isdigit()):
            return None, 0
        if "OOD" in lst[0]:
            return None, 0
        if lst[0] in self.operators.keys():
            res = Node(lst[0], self.params)
            arity = all_operators[lst[0]]
            pos = 1
            for i in range(arity):
                child, length = self._decode(lst[pos:])
                if child is None:
                    return None, pos
                res.push_child(child)
                pos += length
            return res, pos
        else: # other leafs
            return Node(lst[0], self.params), 1

    def decode(self, lst):
        return self._decode(lst)[0]
    
