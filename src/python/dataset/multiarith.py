

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


class MultiArith(Dataset):

    fnames = {
        'train': 'train.json',
        'test': 'test.json'
    }

    def __init__(self, mode='test') -> None:
        root_dir: Path = Macros.dataset_dir / 'multiarith'
        train_rows: List[Dict] = Utils.read_json(
            root_dir / self.fnames['train']
        )
        test_rows: List[Dict] = Utils.read_json(
            root_dir / self.fnames['test']
        )
        train_row_idx = [r_i for r_i in range(len(train_rows))]
        test_row_idx = [len(train_row_idx)+r_i for r_i in range(len(test_rows))]
        self.row_idx = train_row_idx+test_row_idx
        self.body = [r.get('body','') for r in train_rows]+[r.get('body','') for r in test_rows]
        self.question = [r['question'] for r in train_rows]+[r['question'] for r in test_rows]
        self.answer = [r['final_ans'] for r in train_rows]+[r['final_ans'] for r in test_rows]

    def __len__(self):
        return len(self.row_idx)
    
    def __getitem__(self, index: int):
        return {
            'id': self.row_idx[index],
            'body': self.body[index],
            'question': self.question[index],
            'answer': self.answer[index]
        }