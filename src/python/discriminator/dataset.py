
import os
import subprocess
import numpy as np
# import pandas as pd

from typing import *
from pathlib import Path
# from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split

from .mutate import Mutate
from ..utils.macros import Macros
from ..utils.utils import Utils
# from ..utils.logger import Logger


class Svamp(Dataset):

    fnames = 'SVAMP.json'

    def __init__(self, num_neg_codes) -> None:
        root_dir: Path = Macros.dataset_dir / 'svamp'
        rows: List[Dict] = Utils.read_json(root_dir / self.fnames)
        self.res_dir = Macros.result_dir / 'discriminator' / 'svamp'
        self.num_neg_codes = num_neg_codes
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.row_idx = [r['ID'] for r in rows]
        self.query = [
            f"{r['Body']} {r['Question']}"
            for r in rows
        ]
        self.answer = [r['Answer'] for r in rows]
        self.type = [r['Type'] for r in rows]
        self.equation = [r['Equation'] for r in rows]
        self.code = [
            self.convert_equation_to_code(eq, self.answer[eq_i])
            for eq_i, eq in enumerate(self.equation)
        ]
        neg_codes_dict = self.generate_neg_codes(
            self.row_idx,
            self.code,
            self.answer,
            self.num_neg_codes
        )
        self.neg_codes = [
            neg_codes_dict[_id]['mut']
            for _id in self.row_idx
        ]
        raise()

    def __len__(self):
        return len(self.row_idx)
    
    def __getitem__(self, index: int):
        return {
            'query': self.query[index],
            'answer': self.answer[index],
            'equation': self.equation[index],
            'code': self.code[index], # 'neg_codes': self.neg_codes[index]
        }

    def convert_equation_to_code(
        self,
        equation: str, 
        answer: str,
        pl_type: str='python'
    ) -> str:
        if pl_type=='python':
            code_prefix = 'def func():\n    '
            eq_lhs = equation.strip()
            if eq_lhs.startswith('(') and eq_lhs.endswith(')'):
                eq_lhs = eq_lhs[1:-1].strip()
            # end if
            code_str = f"{code_prefix}return {eq_lhs}\n"
            # answer_from_code = self.execute_code_str(
            #     code_str,
            #     pl_type=pl_type
            # )
        # end if
        return code_str

    def execute_code_str(
        self,
        code_str: str,
        pl_type: str='python'
    ):
        # write code temporaly
        temp_file_for_exec = './mut_temp.py'
        Utils.write_txt(f"{code_str}\nprint(func())", temp_file_for_exec)

        # execute the mut_temp.py
        if pl_type=='python':
            cmd = f"python {str(temp_file_for_exec)}"
        # end if
        output = subprocess.check_output(cmd, shell=True).strip()
        output = output.decode()
        os.remove(temp_file_for_exec)
        return output

    def generate_neg_codes(
        self,
        ids: List[str],
        code_strs: List[str],
        answers: List[Any],
        num_examples: int,
        pl_type: str='python'
    ) -> Dict:
        if os.path.exists(str(self.res_dir / 'negative_codes.json',)):
            return Utils.read_json(self.res_dir / 'negative_codes.json')
        # end if
        res = dict()
        for id_i, _id in enumerate(ids):
            code_str = code_strs[id_i]
            answer = answers[id_i]

            mut_obj = Mutate(
                code_str=code_str,
                answer=str(answer),
                res_dir=self.res_dir
            )
            neg_codes = mut_obj.generate_negative_examples(
                num_neg_examples=num_examples
            )
            res[_id] = {
                'orig': code_str,
                'mut': neg_codes
            }
        # end for
        Utils.write_json(
            res,
            self.res_dir / 'negative_codes.json',
            pretty_format=True
        )
        return res


class Mawps(Dataset):

    split_fnames = {
        'val': 'dev.csv'
        'train': 'train.csv'
    }
    csv_headers = [
        'Question'
        'Numbers',
        'Equation',
        'Answer',
        'group_nums',
        'Body',
        'Ques_Statement'
    ]

    num_folds_in_raw_dataset = 5

    def __init__(
        self,
        split='val'
    ) -> None:
        if split not in self.split_fnames.keys():
            valid_splits = '\t'.join(self.split_fnames.keys())
            raise ValueError('Unrecognized split; valid splits are %s' % valid_splits)
        # end if

        root_dir: Path = Macros.dataset_dir / 'mawps'
        for f_n in range(num_folds_in_raw_dataset):
            fold_dir = root_dir / 'cv_mawps' / f"fold{f_n}"
            rows: List[Dict] = Utils.read_json(root_dir / self.fnames)
        # end for


class Asdiv(Dataset):
    
    split_fnames = {
        'val': 'ASDiv.xml'
    }
    
    def __init__(self,
                 split='val') -> None:
        if split not in self.split_fnames.keys():
            valid_splits = '\t'.join(self.split_fnames.keys())
            raise ValueError('Unrecognized split; valid splits are %s' % valid_splits)
        # end if
        self.split = split
        self.root_dir: Path = Macros.dataset_dir / 'asdiv'
        raw_data = Utils.read_json(self.root_dir / 'ASDiv.json')
        avail_data = self.get_available_data(raw_data)

        self.id = [r['id'] for r in avail_data]
        self.query = [
            f"{r['body']} {r['question']}"
            for r in rows
        ]
        self.answer = [r['answer'] for r in avail_data]
        self.equation = [r['equation'] for r in avail_data]
        self.code = [
            r['code'] for r in avail_data
        ]
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index: int):
        return {
            'query': self.query[index],
            'answer': self.answer[index],
            'equation': self.equation[index],
            'code': self.code[index]
        }

    def get_available_data(
        self,
        raw_data: List[Dict]
    ):
        avail_data = list()
        for r in raw_data:
            equation_split = r['equation'].split(';')
            if len(equation_split)==1 or \
                r['solution_type']!='Comparison':
                avail_data.append(r)
            # end if
        # end for
        return avail_data
    
    def convert_equation_to_code(
        self,
        equation: str,
        pl_type: str='python'
    ) -> str:
        if pl_type=='python':
            code_prefix = 'def func():\n    '
            equation_split = equation.split(';')
            if len(equation_split)==1:
                # TODO: for now, we only take the math problems in the dataset 
                # with only one equation for solving the problem
                eq_lhs = equation.strip().split('=')[0]
                code_str = f"{code_prefix}return {eq_lhs}\n"
            # end if
            
            # answer_from_code = self.execute_code_str(
            #     code_str,
            #     pl_type=pl_type
            # )
        # end if
        return code_str

    def generate_neg_codes(
        self,
        code_str: str,
        answer: str,
        num_examples: int,
        pl_type: str='python'
    ):
        pass


def get_data_loader(
    dataset_name: str, 
    batch_size: int, 
    num_workers: int,
    num_neg_codes: int=None
):
    data_pipe = None
    if dataset_name=='asdiv':
        data_pipe = Asdiv()
    elif dataset_name=='svamp':
        data_pipe = Svamp(num_neg_codes=num_neg_codes)
    # end if
    return DataLoader(
        data_pipe,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True, 
        drop_last=True,
        shuffle=True
    )
