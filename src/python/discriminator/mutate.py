
import os
import re
import ast
import copy
import random
import subprocess
import numpy as np

from typing import *
from pathlib import Path

from ..utils.macros import Macros
from ..utils.utils import Utils

PYTHON_ARITH_OPS = [
    '+', # addition
    '-', # subtraction
    '*', # multiplication
    '/', # division
    '%', # modulus
    '**', # exponentiation
    '//' # floor division
]
PYTHON_ARITH_OPS_IN_AST = {
    '+': ast.Add,
    '-': ast.Sub,
    '*': ast.Mult,
    '/': ast.Div,
    '%': ast.Mod,
    '**': ast.Pow,
    '//': ast.FloorDiv
}

MUTATE_METHODS = [
    'mutate_number',
    'mutate_operator'
]

class Mutate:

    def __init__(
        self,
        code_str: str,
        answer: str,
        res_dir: Path,
        pl_type: str='python'
    ):
        self.orig_code_str = code_str
        self.answer = answer
        self.pl_type = pl_type
        self.res_dir = res_dir
        self.ast_tree = self.get_ast_tree()
        self.values_in_code, self.ops_in_code = self.find_value_n_ops_used_in_code()
        self.mut_max_num = max(self.values_in_code)+100
        self.mut_min_num = min(self.values_in_code)-100

    def execute_new_code(
        self, 
        code_str: str,
    ) -> Any:
        cmd = None
        if self.pl_type=='python':
            # write code temporaly
            _code_str = f"import math\n\n{code_str}\nprint(func())\n"
            temp_file_for_exec = self.res_dir / 'code_temp.py'
            Utils.write_txt(_code_str, temp_file_for_exec)
            cmd = f"python {str(temp_file_for_exec)}"
        # end if
        try:
            output = subprocess.check_output(cmd, shell=True).strip()
            return output.decode()
        except subprocess.CalledProcessError:
            return
        # end try

    def get_ast_tree(self):
        code = self.orig_code_str.split('the answer is')[0]
        try:
            ast_tree = ast.parse(code)
        except SyntaxError:
            ast_tree = None
        # end try
        # print(ast.dump(ast_tree, indent=4))
        return ast_tree

    def convert_ast_to_code(self, tree) -> str:
        return ast.unparse(tree) if tree is not None else None

    def find_value_n_ops_used_in_code(self) -> List[Any]:
        _tree = copy.deepcopy(self.ast_tree)
        node_i = 0
        values_used_in_code = list()
        ops_used_in_code = list()
        for node in ast.walk(_tree):
            node_i += 1
            if isinstance(node, ast.Constant) and \
                (type(node.value)==int or type(node.value)==float):
                values_used_in_code.append(node.value)
            elif isinstance(node, ast.BinOp):
                ops_used_in_code.append(node.op)
            # end if
        # end for
        return values_used_in_code, ops_used_in_code

    def generate_random_number(self, val: Any) -> Any:
        if len(self.values_in_code)>1:
            # randomly choose the number of values to be used for the combination
            num_val_use = random.randint(2, len(self.values_in_code))

            # randomly choose values based on the the sampled number of values
            vals = random.sample(self.values_in_code, num_val_use)

            # randomly choose the operators to be used in the values sampled
            new_val = None
            while(True):
                vals_with_ops = list()
                for op_i in range(len(vals)):
                    if op_i<len(vals)-1:
                        if op_i==0:
                            vals_with_ops.append(str(vals[op_i]))
                        # end if
                        op = random.sample(PYTHON_ARITH_OPS, 1)[0]
                        vals_with_ops.append(op)
                        vals_with_ops.append(str(vals[op_i+1]))
                    # end if
                # end for
                eq_for_new_val = ''.join(vals_with_ops)
                try:
                    new_val = eval(eq_for_new_val)
                    if (new_val>=0 and val>=0 and new_val<self.mut_max_num) or \
                        (new_val<0 and val<0 and new_val>self.mut_min_num):
                        break
                    # end if
                except (SyntaxError, OverflowError, ZeroDivisionError) as e:
                    continue
                # end try
            # end while
            return new_val
        else:
            val = self.values_in_code[0]
            if type(self.values_in_code[0])==int:
                val = random.randint(0, val+10) if val>0 else random.randint(val-10, 0)
            elif type(self.values_in_code[0])==float:
                whole = random.randint(0, int(val)+10) if val>0 else random.randint(int(val)-10, 0)
                fraction = random.random() # sample random number between 0 and 1
                val = whole*1.+fraction
            elif type(self.values_in_code[0])==bool:
                # TODO: random select bool over the num_mutation
                val = random.choice([True, False])
            # end if
            return val
        # end if
    
    def generate_random_operator(
        self, 
        node: ast.BinOp
    ) -> Any:
        orig_op = node.op
        arith_op_insts = [
            v for v in PYTHON_ARITH_OPS_IN_AST.values()
            if not isinstance(orig_op,v)
        ]
        new_op = random.sample(arith_op_insts, 1)[0]
        return new_op

    def mutate_number(self) -> str:
        _tree = copy.deepcopy(self.ast_tree)
        node_i = 0
        num_val_use = random.randint(1, len(self.values_in_code))
        vals_of_interest = random.sample(self.values_in_code, num_val_use)

        for node in ast.walk(_tree):
            node_i += 1
            if isinstance(node, ast.Constant) and \
                (type(node.value)==int or type(node.value)==float) and \
                node.value in vals_of_interest:
                new_val = self.generate_random_number(node.value)
                node.value = new_val
            # end if
        # end for
        new_code = self.convert_ast_to_code(_tree)
        return new_code

    def mutate_operator(self) -> str:
        _tree = copy.deepcopy(self.ast_tree)
        node_i = 0
        num_ops_use = random.randint(1, len(self.ops_in_code)) if len(self.ops_in_code)>1 else 1
        ops_of_interest = random.sample(self.ops_in_code, num_ops_use) if len(self.ops_in_code)>1 else self.ops_in_code
        # arith_op_insts = list(PYTHON_ARITH_OPS_IN_AST.values())

        for node in ast.walk(_tree):
            node_i += 1
            if isinstance(node, ast.BinOp):
                for ooi in ops_of_interest:
                    if type(node.op)==type(ooi):
                        new_op = self.generate_random_operator(node)
                        node.op = new_op()
                    # end if
                # end for
            # end if
        # end for
        new_code = self.convert_ast_to_code(_tree)
        return new_code

    def replace_func(self) -> str:
        pass

    def generate_negative_examples(
        self, 
        num_neg_examples: int
    ) -> List[str]:
        num_neg_example_generated = 0
        neg_codes = list()
        while(True):
            method = random.sample(MUTATE_METHODS, 1)[0]
            mut_code = None
            if method==MUTATE_METHODS[0]:
                mut_code = self.mutate_number()
            elif method==MUTATE_METHODS[1]:
                mut_code = self.mutate_operator()
            # end if
            new_out = self.execute_new_code(mut_code)
            if new_out is not None and \
                self.answer.strip()!=new_out.strip() and \
                mut_code not in neg_codes:
                num_neg_example_generated += 1
                neg_codes.append(mut_code)
            # end if
            if num_neg_example_generated==num_neg_examples:
                break
            # end if
        # end while
        return neg_codes
