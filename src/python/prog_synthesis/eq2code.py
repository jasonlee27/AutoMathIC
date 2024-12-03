
import os
import re
import subprocess
import numpy as np

from typing import *
from pathlib import Path

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class Eq2code:

    def __init__(
        self, 
        data: Dict[str,str],
        var_dict: Dict[str,str],
        dataset_name: str, 
        pl_type='python'
    ):
        self.res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        self.math_eq = data['equation'].split('=')[0]
        self.id = data['id']
        self.answer = data['answer']
        self.var_dict = var_dict
        self.eq_cksum_val = Utils.get_cksum(self.id, length=7)
        self.code_headers = Macros.code_header[pl_type]
        self.code_footers = Macros.code_footer[pl_type]
        self.init_statements, self.statements = self.generate_code()
        self.pl_type = pl_type
        self.indent = Macros.pl_indentation_dict[pl_type]
        self.code = self.combine_statements_with_headers_n_footers(
            self.code_headers,
            self.code_footers,
            self.var_dict,
            self.init_statements,
            self.statements
        )

    def find_mostinner_bracket_pair(self) -> List[str]:
        # remove the outermost brackets
        if self.math_eq[0]=='(' and self.math_eq[-1]==')':
            self.math_eq = self.math_eq[1:-1].strip()
        # end if
        return re.findall(r'\([^()]+\)', self.math_eq)

    def parse_operation(self, operation: str):
        operators = set('+-*/')
        op_out = list() # This holds the operators that are found in the string (left to right)
        num_out = list() # this holds the non-operators that are found in the string (left to right)
        buff = list()
        for c in operation:  #examine 1 character at a time
            if c in operators:
                #found an operator.  Everything we've accumulated in `buff` is 
                #a single "number". Join it together and put it in `num_out`.
                num_out.append(''.join(buff))
                buff = list()
                op_out.append(c)
            else:
                #not an operator.  Just accumulate this character in buff.
                buff.append(c)
            # end if
        num_out.append(''.join(buff))
        # 19.0 + 4.0
        return num_out, op_out
    
    def replace_numbers_with_vars(self, values, ops):
        # replace numbers in the operation with variable if related.
        var_list = list()
        for val in values:
            if val.strip().isdigit() or \
               (val.strip().replace('.','',1).isdigit() and \
                val.strip().count('.') < 2):
                for var, _val in self.var_dict.items():
                    if val.strip()==str(_val) or \
                        eval(val.strip())==_val:
                        var_list.append(var)
                        break
                    # end if
                # end for
            else:
                var_list.append(val.strip())
            # end if
        # end for
        statement = ''
        # print(values, ops, self.var_dict)
        if any(ops):
            for op_i, op in enumerate(ops):
                if op_i==0:
                    statement += f"{var_list[op_i]} {op} {var_list[op_i+1]}"
                else:
                    statement += f"{op} {var_list[op_i+1]}"
                # end if
            # end for
        else:
            statement = var_list[0]
        # end if
        return statement

    def generate_statement(
        self, 
        math_eq: str, 
        stat_i: int
    ):
        # check if the math_eq has bracket pairs. 
        # if not, we make return statement.
        bracket_search = re.search(r'\(([^()]+)\)', math_eq)
        var = None
        ops_w_vars = list()
        if bracket_search is None:
            operation = math_eq
            values, ops = self.parse_operation(operation)
            statement = self.replace_numbers_with_vars(values, ops)
            operation = f"return {statement}"
            ops_w_vars.append(operation)
        else:
            var = f"var{stat_i}"
            operation = bracket_search.group(1)
            # parse operation in string into numbers and operators
            values, ops = self.parse_operation(operation)
            statement = self.replace_numbers_with_vars(values, ops)
            op_w_vars = f"{var} = {statement}\n"
            ops_w_vars.append(op_w_vars)
        # end if
        return ops_w_vars, var
    
    def update_math_eq(
        self,
        var_name: str, 
        math_eq_with_bracket: str
    ) -> None:
        # self.math_eq = re.sub(math_eq_with_bracket, var_name, self.math_eq)
        self.math_eq = self.math_eq.replace(math_eq_with_bracket, var_name)
        return

    def combine_statements_with_headers_n_footers(
        self,
        code_headers,
        code_footers,
        var_dict,
        init_statements,
        statements
    ) -> str:
        param_str = code_headers[0]
        print_func_str = '\n\n'+''.join(init_statements)+code_footers[0]
        num_vars = len(var_dict.keys())
        for var_i, var in enumerate(var_dict.keys()):
            if var_i+1<num_vars:
                param_str += f"{var}, "
                print_func_str += f"{var}, "
            else:
                param_str += f"{var}{code_headers[1]}"
                print_func_str += f"{var}{code_footers[1]}"
            # end if
        # end for
        func_body = f"{self.indent}".join(statements)
        code_str = param_str+func_body+print_func_str
        return code_str

    def generate_code(self) -> List[str]:
        stat_i = 0
        done = False
        statements = list()
        init_statements = list()
        for var in self.var_dict.keys():
            init_statements.append(
                f"{var} = {self.var_dict[var]}\n"
            )
        # end for
        math_eq_list = self.find_mostinner_bracket_pair()
        while(any(math_eq_list)):
            for math_eq in math_eq_list:
                # var, operation = self.generate_statement(math_eq, stat_i)
                ops_w_vars, var = self.generate_statement(math_eq, stat_i)
                self.update_math_eq(var, math_eq)
                stat_i += 1
                statements.extend(ops_w_vars)
            # end for
            math_eq_list = self.find_mostinner_bracket_pair()
        # end while
        ops_w_vars, _ = self.generate_statement(self.math_eq, stat_i)
        statements.extend(ops_w_vars)
        return init_statements, statements

    def write_code(
        self, 
        pl_type: str='python'
    ) -> None:
        self.res_dir.mkdir(parents=True, exist_ok=True)
        if pl_type=='python':
            Utils.write_txt(self.code, self.res_dir / f"{self.eq_cksum_val}.py")

            # add answer in the var_dict
            self.var_dict[Macros.answer_key_for_var_dict] = self.answer
            Utils.write_json(self.var_dict, self.res_dir / f"vars-{self.eq_cksum_val}.json")
        # end if
        return

    def execute_code(self) -> None:
        cmd = None
        tgt_code_file = self.res_dir / f"{self.eq_cksum_val}.py"
        if self.pl_type=='python':
            cmd = f"python {str(tgt_code_file)}"
        # end if
        output = subprocess.check_output(cmd, shell=True).strip()
        return output.decode()

def main():
    eq = '(5*9)+(2*1)' # '7.0 - ( 3.0 + 2.0 )'
    cv_obj = Eq2code(eq)
    print(eq)
    print(cv_obj.code)
    cv_obj.write_code()
    cv_obj.execute_code()
    return

