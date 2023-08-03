import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from .generators import all_operators, Node
from ..utils import timeout, MyTimeoutError
import boolean

class InvalidPrefixExpression(BaseException):
    pass


class Simplifier():
    
    def __init__(self, encoder, generator):
        
        self.encoder = encoder
        self.params = generator.params
        self.operators = generator.operators
        self.local_dict = {
            'n': sp.Symbol('n', real=True, nonzero=True, positive=True, integer=True),
            'e': sp.E,
            'pi': sp.pi,
            'euler_gamma': sp.EulerGamma,
            'arcsin': sp.asin,
            'arccos': sp.acos,
            'arctan': sp.atan,
            'sign': sp.sign,
            'step': sp.Heaviside,
        }

    @timeout(1)
    def simplify_tree(self, tree, simplify_form="boolean_package"):
        prefix = tree.prefix().split(',')
        simplified_prefix = self.simplify_prefix(prefix, simplify_form=simplify_form)
        simplified_tree = self.encoder.decode(simplified_prefix)
        if simplified_tree is None: 
            return tree
        else:
            return simplified_tree
    
    def simplify_prefix(self, prefix, simplify_form="boolean_package"):
        infix = self.prefix_to_infix(prefix)
        sympy_infix = self.infix_to_sympy(infix, simplify_form=simplify_form)
        #print('hi', sympy_infix, type(sympy_infix))
        simplified_prefix = self.sympy_to_prefix(sympy_infix)    
        return simplified_prefix
    
    def sympy_to_tree(self, expr):
        return self.encoder.decode(self.sympy_to_prefix(expr))
    
    def get_simple_infix(self, tree, simplify_form="boolean_package"):
        infix = self.prefix_to_infix(tree.prefix().split(','))
        if simplify_form == "none": return infix
        return self.infix_to_sympy(infix, simplify_form=simplify_form)
    
    def get_fourier_tree(self, tree):
        infix = self.prefix_to_infix(tree.prefix().split(','), fourier=True)
        expr = self.infix_to_sympy(infix)
        expr = sp.expand(expr)
        return expr
    
    def get_leap(self, tree):
        polynomial = sp.Poly(self.get_fourier_tree(tree))
        def calc_leap(mon1, mon2):
            return sum([1 for i in range(len(mon1)) if (mon1[i] and not mon2[i])])
        leap = 0
        monomials = list(reversed(polynomial.monoms()))
        seen_monomials = set()
        seen_monomials.add(tuple((0 for _ in range(len(monomials[0])))))
        for monom in monomials:
            curr_leap = min([calc_leap(monom, seen) for seen in seen_monomials])
            seen_monomials.add(monom)
            leap = max(curr_leap,leap)
        return leap
        
    def write_infix(self, token, args, fourier=False):
        """
        Infix representation.
                    if self.value == 'and':
                s = x + '*' + y
            elif self.value == 'or':
                s = f"0.5 * ({x} + {y} + {x} * {y} - 1)"
        """
        if fourier:
            if token == 'and':
                return f'(1+{args[0]})*(1+{args[1]})/4'
            elif token == 'or':
                return f'1-((1-{args[0]})*(1-{args[1]}))/4'
            elif token == 'xor':
                return f'({args[0]})*({args[1]})'
            elif token == 'not':
                return f'-({args[0]})'
            else:
                return token            
        else:
            if token == 'and':
                return f'({args[0]})&({args[1]})'
            elif token == 'or':
                return f'({args[0]})|({args[1]})'
            elif token == 'xor':
                #return f'({args[0]})^({args[1]})'
                return f'(({args[0]})&~({args[1]}))|(~({args[0]})&({args[1]}))'
            elif token == 'not':
                return f'~({args[0]})'
            else:
                return token

    def _prefix_to_infix(self, expr, fourier=False):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in all_operators:
            args = []
            l1 = expr[1:]
            for _ in range(all_operators[t]):
                i1, l1 = self._prefix_to_infix(l1, fourier=fourier)
                args.append(i1)
            return self.write_infix(t, args, fourier=fourier), l1
        else: # leaf
            return t, expr[1:]

    def prefix_to_infix(self, expr, fourier=False):
        """
        Convert prefix expressions to a format that SymPy can parse.        
        """
        p, r = self._prefix_to_infix(expr, fourier=fourier)
        if len(r) > 0:
            raise InvalidPrefixExpression(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
        return f'({p})'


    def infix_to_sympy(self, infix, simplify_form="basic"):
        """
        Convert an infix expression to SymPy.
        """
        if simplify_form == "boolean_package":
            algebra = boolean.BooleanAlgebra()
            expr = algebra.parse(infix)
            return expr.simplify()
        
        expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        if simplify_form == "basic":
            pass            
        elif simplify_form == "shortest":
            expr1 = sp.simplify_logic(expr, form='cnf')
            expr2 = sp.simplify_logic(expr, form='dnf')
            if len(str(expr1)) < len(str(expr2)):
                expr = expr1
            else:
                expr = expr2  
        else:
            assert simplify_form in ['cnf', 'dnf']
            expr = sp.simplify_logic(expr, form=simplify_form)    
        return expr
    
    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol) or isinstance(expr, boolean.Symbol) or isinstance(expr, bool):
            return [str(expr)]
        if isinstance(expr, boolean.boolean._FALSE) or str(expr) in ['True','1']:
            return ["True"]
        if isinstance(expr, boolean.boolean._TRUE) or str(expr) in ['False','0']:
            return ["False"]
        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # SymPy operator
        for op_type, op_name in self.BOOLEAN_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # Unknown operator
        return self._sympy_to_prefix(str(type(expr)), expr)     
      
        
    SYMPY_OPERATORS = {
        # Elementary functions
        sp.And: 'and',
        sp.Or:  'or',
        sp.Xor:  'xor',
        sp.Not: 'not',
    }

    BOOLEAN_OPERATORS = {
        # Elementary functions
        boolean.AND: 'and',
        boolean.OR :  'or',
        boolean.NOT: 'not',
    }
