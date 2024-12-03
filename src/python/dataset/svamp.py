

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


class Svamp(Dataset):

    fnames = 'SVAMP.json'

    def __init__(self) -> None:
        root_dir: Path = Macros.dataset_dir / 'svamp'
        rows: List[Dict] = Utils.read_json(root_dir / self.fnames)
        self.row_idx = [r['ID'] for r in rows]
        self.body = [r['Body'] for r in rows]
        self.question = [r['Question'] for r in rows]
        self.answer = [r['Answer'] for r in rows]
        self.type = [r['Type'] for r in rows]
        self.equation = [r['Equation'] for r in rows]

    def __len__(self):
        return len(self.row_idx)
    
    def __getitem__(self, index: int):
        return {
            'id': self.row_idx[index],
            'body': self.body[index],
            'question': self.question[index],
            'type': self.type[index],
            'answer': self.answer[index],
            'equation': self.equation[index]
        }