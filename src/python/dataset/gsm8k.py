

import os
import numpy as np
# import pandas as pd

from typing import *
from pathlib import Path
# from datasets import load_dataset
from torch.utils.data import Dataset

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class Gsm8k(Dataset):

    """
    {
        "row_idx":0,
        "row":{
            "question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"
        },
        "truncated_cells":[]
    }
    """

    split_fnames = {
        'train': 'train.json',
        'test': 'test.json'
    }

    def __init__(self,
                 split='test') -> None:
        if split not in self.split_fnames.keys():
            valid_splits = '\t'.join(self.split_fnames.keys())
            raise ValueError('Unrecognized split; valid splits are %s' % valid_splits)
        # end if
        self.split = split
        root_dir: Path = Macros.dataset_dir / 'gsm8k'
        # print(root_dir / self.split_fnames[self.split])
        rows = Utils.read_json(root_dir / self.split_fnames[self.split])
        self.row_idx = [r_i for r_i in range(len(rows))]
        self.body = [r.get('body','') for r in rows]
        self.question = [r['question'] for r in rows]
        self.explanation, self.answer = self.split_answer_n_explanation(rows)

    def split_answer_n_explanation(self, rows) -> List:
        delimeter = '\n#### '
        exp_list, ans_list = list(), list()
        for r in rows:
            r_split = r['answer'].split(delimeter)
            exp, ans = r_split[0], r_split[1]
            exp_list.append(exp)
            ans_list.append(eval(ans))
        # end for
        return exp_list, ans_list
        
    # def generate_equation(self):
    #     eq_pat = r'\<\<.+\>\>'
    #     eq_res = list()
    #     for ex in self.explanation:
    #         eqs = re.findall(eq_pat, ex)
    #         if len(eqs)==1:
    #             eq_res.append(eqs[0])
    #         else:
    #             eq_dict = dict()
    #             eq_str = ''
    #             for eq in eqs:
    #                 lhs, rhs = eq.split('=')
    #                 if eq_dict.get(rhs, None) is None:
    #                     eq_dict[rhs.strip()] = lhs
    #                 else:
    #                     op_used = None
    #                     for op in Macros.math_operators:
    #                         lhs_split = lhs.split(op)
    #                         if lhs_split!=lhs:
    #                             llhs = lhs_split[0].strip()
    #                             rlhs = lhs_split[1].strip()
    #                             op_used = op
    #                         # end if
    #                     # end for                        
    #                     llhs_search = eq_dict.get(llhs, None)
    #                     rlhs_search = eq_dict.get(rlhs, None)
    #                     if llhs_search is not None:
    #                         eq_str = f"( {llhs_search} ) {op_used} {rlhs} = {rhs}"
    #                     elif rlhs_search is not None:
    #                         eq_str = f"{llhs} {op_used} ( {rlhs_search} ) = {rhs}"
    #                     # end if
    #                     eq_dict[rhs.strip()] = eq_str
    #                 # end if
    #             # end for
    #             eq_res.append(eq_str)
    #         # end if
    #     return eq_res

    def __len__(self):
        return len(self.row_idx)
    
    def __getitem__(self, index: int):
        return {
            'id': self.row_idx[index],
            'body': self.body[index],
            'question': self.question[index],
            'explanation': self.explanation[index],
            'answer': self.answer[index],
        }
