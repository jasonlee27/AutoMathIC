
import os
import re
import ast
import copy
import subprocess
import numpy as np

# from typing import *
# from pathlib import Path

# from .consistency import ModalConsistency
# from ..mutation.mutate2nl import NUMBER_IN_ENGLISH

# from ..utils.utils import Utils
# from ..utils.macros import Macros
# from ..utils.logger import Logger


# class AlgorithmConsistency:

#     def __init__(
#         self, 
#         cksum_val: str,
#         eval_res_dir: Path,
#         include_only_modconst: bool=False,
#         mutation_of_interest: str=None
#     ):
#         # self.eval_res_dir = Macros.res_dir / 'nl2nl' / dataset_name / 'evaluate_consistency' / model_under_test_name
#         self.cksum_val = cksum_val
#         self.include_only_modconst = include_only_modconst
#         self.eval_res_dir = eval_res_dir
#         self.res_dir = eval_res_dir / 'alg_consistency'
#         if include_only_modconst:
#             self.res_dir = eval_res_dir / 'alg_consistency_only_modconst'
#         # end if
#         self.res_dir.mkdir(parents=True, exist_ok=True)
#         self.tgt_code_str, \
#         self.mut_code_strs = self.get_code_str(mutation_of_interest=mutation_of_interest)
#         self.num_muts = len(self.mut_code_strs.keys())
#         if self.num_muts>0:
#             self.tgt_tree, \
#             self.mut_trees = self.get_ast_tree()
#             self.tgt_values, \
#             self.mut_values = self.get_values_in_question()
#         # end if

#     def is_modal_consistency(
#         self, 
#         cot_resp: str, 
#         code_resp: str,
#         answer_gt: Any
#     ) -> bool:
#         answer_from_cot = ModalConsistency.get_answer_from_cot_resp(cot_resp)
#         answer_from_code = ModalConsistency.get_answer_from_code_resp(code_resp)
#         answer_from_cot = eval(answer_from_cot) if answer_from_cot is not None else answer_from_cot
#         answer_from_code = eval(answer_from_code) if answer_from_code is not None else answer_from_code
#         return answer_from_cot==answer_from_code
        
#     def convert_string_to_number(self, str_num: str) -> bool:
#         try:
#             num_complex = complex(str_num)
#             return num_complex.real
#         except ValueError:
#             return NUMBER_IN_ENGLISH.get(str_num.lower(), None)
#         # end try
#         return 

#     def get_tokens_w_values(
#         self, 
#         input_text: str,
#     ):
#         val_tokens = list()
#         val_token_inds = list()
#         tokens = Utils.tokenize(input_text)
#         for t_i, t in enumerate(tokens):
#             t_val = self.convert_string_to_number(t)
#             if t_val is not None:
#                 val_tokens.append(t)
#                 val_token_inds.append(t_i)
#             # end if
#         # end for
#         return val_tokens, val_token_inds

#     def get_values_in_question(self):
#         tgt_q = list(self.tgt_code_str.keys())[0]
#         tgt_val_tokens, tgt_val_token_inds = self.get_tokens_w_values(tgt_q)
#         tgt_value_dict = {
#             tgt_q: {
#                 'tokens': tgt_val_tokens,
#                 'indices': tgt_val_token_inds
#             }
#         }
#         mut_value_dict = dict()
#         for mut_q in self.mut_code_strs.keys():
#             mut_val_tokens, \
#             mut_val_token_inds = self.get_tokens_w_values(mut_q)
#             mut_value_dict[mut_q] = {
#                 'tokens': mut_val_tokens,
#                 'indices': mut_val_token_inds
#             }
#         # end for
#         return tgt_value_dict, mut_value_dict

#     def get_code_str(self, mutation_of_interest: str=None):    
#         eval_res = Utils.read_json(
#             self.eval_res_dir / f"eval-{self.cksum_val}.json"
#         )
#         tgt_code_str = {
#             eval_res['orig']['question']: eval_res['orig']['code_response']
#         }
#         mut_code_strs = dict()
#         if eval_res['orig']['question']==mutation_of_interest:
#             mut_code_strs[eval_res['orig']['question']] = eval_res['orig']['code_response']
#         # end if
#         for mut in eval_res['mutation']:
#             if mutation_of_interest is not None:
#                 if mutation_of_interest==mut['question']:
#                     mut_code_strs[mut['question']] = mut['code_response']
#                 # end if
#             elif self.include_only_modconst:
#                 if self.is_modal_consistency(
#                     mut['cot_response'],
#                     mut['code_response'],
#                     eval_res['orig']['answer']
#                 ):
#                     # we only accept the modal-consistent mutations
#                     mut_code_strs[mut['question']] = mut['code_response']
#                 # end if
#             else:
#                 mut_code_strs[mut['question']] = mut['code_response']
#             # end if
#         # end for
#         return tgt_code_str, mut_code_strs

#     def get_ast_tree(self):
#         tgt_q = list(self.tgt_code_str.keys())[0]
#         code = self.tgt_code_str[tgt_q].split('the answer is')[0]
#         try:
#             ast_obj = ast.parse(code)
#         except SyntaxError:
#             ast_obj = None
#         # end try
#         tgt_tree = {
#             tgt_q: ast_obj
#         }
#         mut_trees = dict()
#         for key in self.mut_code_strs.keys():
#             code = self.mut_code_strs[key].split('the answer is')[0]
#             try:
#                 ast_obj = ast.parse(code)
#             except SyntaxError:
#                 ast_obj = None
#             # end try
#             mut_trees[key] = ast_obj
#         # end for
#         return tgt_tree, mut_trees
    
#     def replace_value_in_ast(
#         self, 
#         tree, 
#         values_from: str, 
#         values_to: str
#     ):
#         if tree is None:
#             return None
#         # end if
#         _tree = copy.deepcopy(tree)
#         node_inds_replaced = list()
#         for v_f, v_t in zip(values_from , values_to):
#             if not v_f.isdigit():
#                 v_f = str(NUMBER_IN_ENGLISH.get(v_f.lower(), None))
#             # end if
#             if not v_t.isdigit():
#                 v_t = str(NUMBER_IN_ENGLISH.get(v_t.lower(), None))
#             # end if
#             node_i = 0
#             for node in ast.walk(_tree):
#                 node_i += 1
#                 if isinstance(node, ast.Constant) and \
#                     str(node.value)==v_f.strip() and \
#                     node_i not in node_inds_replaced:
#                     node.value = eval(v_t)
#                     node_inds_replaced.append(node_i)
#                 # end if
#             # end for
#         # end for
#         return _tree
    
#     def convert_ast_to_code(
#         self,
#         tree
#     ) -> str:
#         return ast.unparse(tree) if tree is not None else None

#     def get_new_codes(self, save: bool=True):
#         tgt_q = list(self.tgt_code_str.keys())[0]
#         tgt_tree = self.tgt_tree[tgt_q]
#         tgt_values = self.tgt_values[tgt_q]['tokens']
#         res_dict = dict()
#         res_dict[f"CODE_USED::{tgt_q}"] = {
#             f"VALUE_USED::{tgt_q}": None
#         }
#         for mut_q in self.mut_code_strs.keys():
#             mut_values = self.mut_values[mut_q]['tokens']
#             new_ast = self.replace_value_in_ast(
#                 tgt_tree,
#                 tgt_values,
#                 mut_values
#             )
#             new_code = self.convert_ast_to_code(new_ast)
#             res_dict[f"CODE_USED::{tgt_q}"][f"VALUE_USED::{mut_q}"] = new_code
#         # end for

#         for mut_q in self.mut_code_strs.keys():
#             mut_tree = self.mut_trees[mut_q]
#             mut_values = self.mut_values[mut_q]['tokens']
#             res_dict[f"CODE_USED::{mut_q}"] = {
#                 f"VALUE_USED::{mut_q}": None
#             }
#             new_ast = self.replace_value_in_ast(
#                 mut_tree, 
#                 mut_values,
#                 tgt_values
#             )
#             new_code = self.convert_ast_to_code(new_ast)
#             res_dict[f"CODE_USED::{mut_q}"][f"VALUE_USED::{tgt_q}"] = new_code
            
#             for _mut_q in self.mut_code_strs.keys():
#                 if mut_q!=_mut_q:
#                     _mut_values = self.mut_values[_mut_q]['tokens']
#                     new_ast = self.replace_value_in_ast(
#                         mut_tree, 
#                         mut_values,
#                         _mut_values
#                     )
#                     new_code = self.convert_ast_to_code(new_ast)
#                     res_dict[f"CODE_USED::{mut_q}"][f"VALUE_USED::{_mut_q}"] = new_code
#                 # end if
#             # end for
#         # end for
#         if save:
#             Utils.write_json(
#                 res_dict,
#                 self.res_dir / f"new-codes-{self.cksum_val}.json",
#                 pretty_format=True
#             )
#         # end if
#         return res_dict
    
#     def execute_new_code(
#         self, 
#         code_str: str,
#         pl_type: str
#     ):
#         cmd = None
#         if pl_type=='python':
#             # write code temporaly
#             _code_str = f"import math\n\n{code_str}\nprint(func())"
#             temp_file_for_exec = self.res_dir / 'code_temp.py'
#             Utils.write_txt(_code_str, temp_file_for_exec)
#             cmd = f"python {str(temp_file_for_exec)}"
#         # end if
#         try:
#             output = subprocess.check_output(cmd, shell=True).strip()
#             return output.decode()
#         except subprocess.CalledProcessError:
#             return None
#         # end try
        
#     def get_result_matrix(
#         self, 
#         new_code_dict: Dict,
#         dataset_name: str, 
#         pl_type: str
#     ):
#         res_mat_dict = dict()
#         mut_i = 0
#         tgt_mut_inds = {
#             'tgt': dict(),
#             'mut': dict()
#         }
#         for key_code_used in sorted(new_code_dict.keys()):
#             mut_i += 1
#             key_q = key_code_used.split('CODE_USED::')[-1]
#             if key_q in self.tgt_code_str.keys():
#                 tgt_mut_inds['tgt'][key_q] = 'TGT'
#             elif key_q in self.mut_code_strs.keys():
#                 tgt_mut_inds['mut'][key_q] = f"MUT_{mut_i}"
#             # end if
#         # end for

#         for key_code_used in tgt_mut_inds['tgt'].keys():
#             res_key_code_used = 'CODE_USED::'+tgt_mut_inds['tgt'][key_code_used]
#             res_mat_dict.setdefault(res_key_code_used, dict())
#             res_mat_dict[res_key_code_used].setdefault(
#                 'VALUE_USED::'+tgt_mut_inds['tgt'][key_code_used],
#                 None
#             )
#             for key_val_used in tgt_mut_inds['mut'].keys():
#                 res_key_val_used = 'VALUE_USED::'+tgt_mut_inds['mut'][key_val_used]
#                 new_code = new_code_dict['CODE_USED::'+key_code_used]['VALUE_USED::'+key_val_used]
#                 elem_val = 0.
#                 if new_code is not None:
#                     out = self.execute_new_code(new_code, pl_type)

#                     mut_code_resp = self.mut_code_strs[key_val_used]
#                     orig_code_ans = ModalConsistency.get_answer_from_code_resp(
#                         mut_code_resp
#                     )
#                     if out==orig_code_ans:
#                         elem_val = 1.
#                     # end if
#                 # end if
#                 res_mat_dict[res_key_code_used].setdefault(
#                     res_key_val_used, 
#                     elem_val
#                 )
#             # end for
#         # end for

#         for key_code_used in tgt_mut_inds['mut'].keys():
#             res_key_code_used = 'CODE_USED::'+tgt_mut_inds['mut'][key_code_used]
#             res_mat_dict.setdefault(res_key_code_used, dict())
            
#             for key_val_used in tgt_mut_inds['tgt'].keys():
#                 res_key_val_used = 'VALUE_USED::'+tgt_mut_inds['tgt'][key_val_used]
#                 new_code = new_code_dict['CODE_USED::'+key_code_used]['VALUE_USED::'+key_val_used]
#                 elem_val = 0.
#                 if new_code is not None:
#                     out = self.execute_new_code(new_code, pl_type)
#                     tgt_code_resp = self.tgt_code_str[key_val_used]
#                     orig_code_ans = ModalConsistency.get_answer_from_code_resp(
#                         tgt_code_resp
#                     )
#                     if out==orig_code_ans:
#                         elem_val = 1.
#                     # end if
#                 # end if
#                 res_mat_dict[res_key_code_used].setdefault(
#                     res_key_val_used,
#                     elem_val
#                 )
#             # end for

#             for key_val_used in tgt_mut_inds['mut'].keys():
#                 res_key_val_used = 'VALUE_USED::'+tgt_mut_inds['mut'][key_val_used]
#                 if key_code_used!=key_val_used:
#                     new_code = new_code_dict['CODE_USED::'+key_code_used]['VALUE_USED::'+key_val_used]
#                     elem_val = 0.
#                     if new_code is not None:
#                         out = self.execute_new_code(new_code, pl_type)
#                         mut_code_resp = self.mut_code_strs[key_val_used]
#                         orig_code_ans = ModalConsistency.get_answer_from_code_resp(
#                             mut_code_resp
#                         )
#                         if out==orig_code_ans:
#                             elem_val = 1.
#                         # end if
#                     # end if
#                     res_mat_dict[res_key_code_used].setdefault(
#                         res_key_val_used,
#                         elem_val
#                     )
#                 else:
#                     res_mat_dict[res_key_code_used].setdefault(
#                         res_key_val_used,
#                         None
#                     )
#                 # end if
#             # end for
#         # end for
#         os.remove(str(self.res_dir / 'code_temp.py'))
#         return res_mat_dict, tgt_mut_inds

#     @classmethod
#     def get_answer_from_mut_of_interest(
#         cls,
#         cksum_val: str,
#         mutation_of_interest: str,
#         dataset_name: str,
#         model_name: str,
#         pl_type: str='python',
#         include_only_modconst: bool=False
#     ):
#         res_dir = Macros.result_dir / 'nl2nl' / dataset_name
#         eval_dir = res_dir / 'evaluate_consistency' / model_name
#         alg_const_dir = eval_dir / 'alg_consistency'
#         if include_only_modconst:
#             alg_const_dir = eval_dir / 'alg_consistency_only_modconst'
#         # end if
#         eval_res = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
#         tgt_q = eval_res['orig']['question']
#         # if mutation_of_interest==tgt_q:
#         #     out = ModalConsistency.get_answer_from_cot_resp(
#         #         eval_res['orig']['cot_response']
#         #     )
#         #     new_code = 'cot_response'
#         # else:
#         alg_const_obj = cls(
#             cksum_val, 
#             eval_dir, 
#             include_only_modconst=include_only_modconst,
#             mutation_of_interest=mutation_of_interest
#         )
#         new_code_dict = alg_const_obj.get_new_codes(save=False)
#         new_code = new_code_dict[f"CODE_USED::{mutation_of_interest}"][f"VALUE_USED::{tgt_q}"]
#         out = None
#         if new_code is not None:
#             out = alg_const_obj.execute_new_code(new_code, pl_type)
#         # end if
#         # end if 08fd240
#         return out, new_code

#     @classmethod
#     def main(
#         cls,
#         dataset_name: str,
#         model_name: str,
#         pl_type: str='python',
#         include_only_modconst: bool=False
#     ):
#         res_dir = Macros.result_dir / 'nl2nl' / dataset_name
#         eval_dir = res_dir / 'evaluate_consistency' / model_name
#         llm_response_files = sorted([
#             f for f in os.listdir(str(eval_dir))
#             if f.endswith('.json') and f.startswith('eval-') and \
#             (not f.startswith('eval-results-w-demo'))
#         ])
#         for resp_file in llm_response_files:
#             cksum_val = re.search(r'eval\-(.*)\.json', resp_file).group(1)
#             print(cksum_val)
#             alg_const_obj = cls(
#                 cksum_val, 
#                 eval_dir, 
#                 include_only_modconst=include_only_modconst
#             )
#             if alg_const_obj.num_muts>0:
#                 new_code_dict = alg_const_obj.get_new_codes()
#                 res_mat_dict, tgt_mut_inds = alg_const_obj.get_result_matrix(
#                     new_code_dict, 
#                     dataset_name, 
#                     pl_type
#                 )
#                 Utils.write_json({
#                         'res_matrix': res_mat_dict,
#                         'question_n_indices_map': tgt_mut_inds
#                     },
#                     alg_const_obj.res_dir / f"alg-consistency-mat-{cksum_val}.json",
#                     pretty_format=True
#                 )
#             # end if
#         # end for
#         return