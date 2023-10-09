from abc import ABC, abstractmethod
from ast import parse
from operator import length_hint
import numpy as np
from graphviz import Digraph

import math
import scipy.special
import copy
from collections import defaultdict
from treelib import Tree

operators = {
    'and': 2,
    'or' : 2,
    'xor': 2,
    #'nand':2,
    #'nor': 2,
    'not': 1,
}


operators_extra = {
}

all_operators = {**operators, **operators_extra}

def get_random_id():
    return '%08i' % np.random.randint(0, 100000000)

class Node():
    def __init__(self, value, params, children=None):
        self.value = value
        self.children = children if children else []
        self.params = params
        self.siblings = None

    def push_child(self, child):
        self.children.append(child)

    def prefix(self):
        s = str(self.value)
        for c in self.children:
            s += ',' + c.prefix()
        return s
    
    def infix(self):
        nb_children = len(self.children)
        if nb_children <= 1:
            s = str(self.value)
            if nb_children == 1:
                s = s + '(' + self.children[0].infix() + ')'
            return s
        s = '(' + self.children[0].infix()
        for c in self.children[1:]:
            s = s + ' ' + str(self.value) + ' ' + c.infix()
        return s + ')'
    
    def to_arbitrary_fan_in(self):
        for c in self.children:
            c.to_arbitrary_fan_in()
            if c.value == self.value:
                for cc in c.children:
                    self.children.append(cc)
                self.children.remove(c)
    
    def prefix_arbitrary_fan_in(self, parent=None):
        s = str(self.value)
        if s == parent:
            s = ""
            for c in self.children:
                s += ',' + c.prefix_arbitrary_fan_in(self.value)
        else:
            for c in self.children:
                s += ',' + c.prefix_arbitrary_fan_in(self.value)
        return s

    # export to latex forest
    def forest_prefix(self, parent=None):
        s = str(self.value)
        if s == parent:
            s = ""
            for c in self.children:
                s += c.forest_prefix(self.value)
        else:
            if s in all_operators:
                s = '\mathrm{'+s+'}'
            s = "[$" + s + "$ "
            for c in self.children:
                s += c.forest_prefix(self.value)
            s += "]"
        return s
    
    def print_latex(self):
        print(r'\centering')
        print(r'\begin{forest}')
        print(self.forest_prefix())
        print(r'\end{forest}')

    def get_latex(self):
        res = '\centering\n'
        res += '\\begin{forest}\n'
        res += self.forest_prefix()
        res += '\n\\end{forest}'
        return res
        
    def treelib(self, tree = None, parent_id = None):
        if not tree:
            tree = Tree()
        id = get_random_id()
        tree.create_node(self.value, id, parent=parent_id)
        for c in self.children:
            c.treelib(tree, parent_id=id) 
        return tree

    def graphviz(self, tree=None, parent=None, id_parent=None):
        if tree is None:
            tree = Digraph()
        id = get_random_id()
        if self.value == parent:
            for c in self.children:
                c.graphviz(tree, self.value, id_parent)
        else:
            tree.node(name=id, label=self.value.replace('x_', 'x'))
            if id_parent is not None:
                tree.edge(id_parent, id)
            for c in self.children:
                c.graphviz(tree, self.value, id) 
        return tree
    
    def check_leaves(self):
        if not self.children:
            return True
        if len(self.children) == 1:
            return self.children[0].check_leaves()
        else:
            c1, c2 = self.children
            if not c1.children and not c2.children and c1.value == c2.value:
                return False
            return c1.check_leaves() and c2.check_leaves()

    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        # infix a default print
        return self.infix()
    
    def __repr__(self):
        # infix a default print
        return self.prefix()
    
    def __call__(self, x):
        return self.val(x)
    
    def val(self, x):
        if len(x.shape) < 2:
            x = x.reshape(1, -1)
        if len(self.children) == 0:
            if self.value.startswith('x_'):
                dimension = int(self.value.split('_')[-1])
                if not 0 <= dimension < x.shape[1]:
                    res = np.zeros(x.shape[0]); res.fill(np.nan)
                    return res
                return x[:, dimension] 
            elif str(self.value) in ['1','True']:
                return np.ones(x.shape[0], dtype=bool)
            elif str(self.value) in ['0','False']:
                return np.zeros(x.shape[0], dtype=bool)

        elif self.value == 'and':
            #return np.logical_and(self.children[0].val(x), self.children[1].val(x))
            return np.logical_and.reduce([c.val(x) for c in self.children])
        elif self.value == 'or':
            #return np.logical_or(self.children[0].val(x), self.children[1].val(x))
            return np.logical_or.reduce([c.val(x) for c in self.children])
        elif self.value == 'nand':
            return np.logical_not(np.logical_and(self.children[0].val(x), self.children[1].val(x)))
        elif self.value == 'nor':
            return np.logical_not(np.logical_or(self.children[0].val(x), self.children[1].val(x)))
        elif self.value == 'xor':
            return np.logical_xor(self.children[0].val(x), self.children[1].val(x))
        elif self.value == 'not':
            return np.logical_not(self.children[0].val(x))
        else:
            raise Exception(f"Unknown operator {self.value}")
            
    def get_n_binary_ops(self):
        if self.value in all_operators:
            if self.value == 'not':
                return self.children[0].get_n_binary_ops()
            else:
                return 1 + sum([child.get_n_binary_ops() for child in self.children])
        else: 
            return 0  
        
    def get_n_ops_arbitrary_fan_in(self):
        prefix = self.prefix_arbitrary_fan_in().split(',')
        # count number of operators
        n_ops = 0
        for token in prefix:
            if token in all_operators and token != 'not':
                n_ops += 1
        return n_ops


    def get_n_ops(self):
        if self.value in all_operators:
            return 1 + sum([child.get_n_ops() for child in self.children])
        else: 
            return 0   

    def get_n_vars(self):
        counts = defaultdict(int)
        for token in self.prefix().split(','):
            if token.startswith('x_'):
                counts[token] += 1
        return len(counts.keys())
    
    def get_variables(self):
        variables = set()
        for token in self.prefix().split(','):
            if token not in all_operators and token not in ['1','0','True','False']:
                variables.add(token)
        return variables

    def replace_node_value(self, old_value, new_value):
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)

    def increment_variables(self):
        # make variables names start from 1 instead of 0
        if self.value.startswith('x_'):
            self.value = 'x_' + str(int(self.value.split('_')[-1]) + 1)
        for child in self.children:
            child.increment_variables()

    def decrement_variables(self):
        # make variables names start from 1 instead of 0
        if self.value.startswith('x_'):
            self.value = 'x_' + str(int(self.value.split('_')[-1]) - 1)
        for child in self.children:
            child.decrement_variables()

    def relabel_variables(self, dictionary):
        if self.value not in all_operators and self.value not in ['1','0','True','False']:
            dim = int(self.value.split('_')[-1])
            self.value = dictionary[dim]
        for child in self.children:
            child.relabel_variables(dictionary)

    def relabel_operators(self, dictionary):
        if self.value in all_operators:
            self.value = dictionary[self.value]
        for child in self.children:
            child.relabel_operators(dictionary)

    def skeletonize(self):
        var_dictionary = {}
        for variable in self.get_variables():
            dimension = int(variable.split('_')[-1])
            var_dictionary[dimension] = 'x'
        self.relabel_variables(var_dictionary)
        #op_dictionary = {}
        #for operator in all_operators:
        #    op_dictionary[operator] = "op"
        #self.relabel_operators(op_dictionary)

    def simplify(self):
        for c in self.children:
            c.simplify()
        if self.value == 'not' and self.children[0].value=='not':
            self.value = self.children[0].children[0].value
            self.children = self.children[0].children[0].children
        if self.value in ['and', 'or']:
            num_not = sum([c.value=='not' for c in self.children])
            if num_not > len(self.children)//2:
                if self.value == 'and':
                    self.value = 'or'
                else:
                    self.value = 'and'
                for child in self.children:
                    child.negate()
                self.negate()

    def negate(self):
        if self.value == 'not':
            self.value = self.children[0].value
            self.children = self.children[0].children
        else:
            self.children = [Node(self.value, self.params, self.children)]
            self.value = 'not'
            
class Generator(ABC):
    def __init__(self, params):
        pass
        
    @abstractmethod
    def generate(self, rng):
        pass

    
class RandomBooleanFunctions(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.operators = copy.deepcopy(all_operators)
        
        self.unaries = [o for o in self.operators.keys() if np.abs(self.operators[o]) == 1 and o in params.operators_to_use]
        self.binaries =  [o for o in self.operators.keys() if np.abs(self.operators[o]) == 2 and o in params.operators_to_use]
        self.unary = False #len(self.unaries) > 0
        self.distrib = self.generate_dist(2 * self.params.max_ops)

        self.variables = [f"x_{i}" for i in range(0, self.params.max_active_vars+self.params.max_inactive_vars)]
        self.symbols = list(self.operators) + self.variables 
        
    def generate_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n- 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D = []
        D.append([0] + ([1 for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        return D

    def generate_leaf(self, rng, variable_counts):
        unused_variables = [v for v in variable_counts if variable_counts[v] == 0]
        if unused_variables:
            return rng.choice(unused_variables)
        else:
            return rng.choice(list(variable_counts.keys()))

    def generate_ops(self, rng, arity):
        if arity==1:
            ops=self.unaries
        else:
            ops=self.binaries
        return rng.choice(ops)

    def sample_next_pos(self, rng, nb_empty, n_ops):
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `nb_empty` - 1}.
        """
        assert nb_empty > 0
        assert n_ops > 0
        probs = []
        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[n_ops - 1][nb_empty - i])
        for i in range(nb_empty):
            probs.append(self.distrib[n_ops - 1][nb_empty - i + 1])
        probs = [p / self.distrib[n_ops][nb_empty] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(len(probs), p=probs)
        arity = 1 if self.unary and e < nb_empty else 2
        e %= nb_empty
        return e, arity
    
    def fill_empty_nodes(self, rng, empty_nodes, variable_counts):
        for i,n in enumerate(empty_nodes):
            while True:
                leaf = self.generate_leaf(rng, variable_counts)
                if len(variable_counts)>1 and i>0 and empty_nodes[i-1].parent_id == n.parent_id:
                    brother = empty_nodes[i-1]
                    if brother.value == leaf:
                        continue
                variable_counts[leaf] += 1
                n.value = leaf
                break
            
    def generate_tree(self, rng, n_active_vars, n_inactive_vars, n_ops):
        tree = Node(0, self.params)
        variable_counts = {variable:0 for variable in rng.choice(self.variables[:n_active_vars+n_inactive_vars], 
                                                                 n_active_vars, 
                                                                 replace=False)}
        empty_nodes = [tree]
        next_en = 0
        nb_empty = 1
        while n_ops > 0:
            next_pos, arity = self.sample_next_pos(rng, nb_empty, n_ops)
            self.fill_empty_nodes(rng, empty_nodes[next_en:next_en + next_pos], variable_counts)
            next_en += next_pos
            op = self.generate_ops(rng, arity)
            parent_id = get_random_id()
            empty_nodes[next_en].value = op
            for _ in range(arity):
                e = Node(0, self.params)
                e.parent_id = parent_id
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            nb_empty += arity - 1 - next_pos
            n_ops -= 1
            next_en += 1
        self.fill_empty_nodes(rng, empty_nodes[next_en:], variable_counts)
        
        return tree

        
    def generate(self, rng, n_active_vars=None, n_inactive_vars=None, n_ops=None):
        #rng = rng
        #rng.seed() 

        """prediction_points is a boolean which indicates whether we compute prediction points. By default we do not to save time. """
        trees = []
        if n_active_vars is None:
            n_active_vars = rng.randint(self.params.min_active_vars, self.params.max_active_vars + 1)
        else:
            assert n_active_vars <= self.params.max_active_vars 
        if n_inactive_vars is None:
            n_inactive_vars = rng.randint(self.params.min_inactive_vars, self.params.max_inactive_vars + 1)
        else:
            assert n_inactive_vars <= self.params.max_inactive_vars
        if n_ops is None:
            n_ops = rng.randint(n_active_vars-1, self.params.max_ops+1)
        else:
            assert n_ops <= self.params.max_ops

        tree = self.generate_tree(rng, n_active_vars=n_active_vars, n_inactive_vars=n_inactive_vars, n_ops=n_ops)
        tree = self.add_unaries(rng, tree)

        # if not tree.check_leaves():
        #     print('bad leaves', tree)
        #     tree = None
            
        return tree, n_active_vars, n_inactive_vars
    
    def add_unaries(self, rng, tree):
        prefix = self._add_unaries(rng, tree)
        prefix = prefix.split(",")
        tree = self.equation_encoder.decode(prefix)
        return tree

    def _add_unaries(self, rng, tree):
        s = str(tree.value)
        if rng.rand() < self.params.unary_prob:
            s = f"{rng.choice(self.unaries)}," + s
        for c in tree.children:
            s += f"," + self._add_unaries(rng, c)
        return s

    def chunks_idx(self, step, min, max):
        curr=min
        while curr<max:
            yield [i for i in range(curr, curr+step)]
            curr+=step

