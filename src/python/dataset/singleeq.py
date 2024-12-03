
import os
import csv
import numpy as np
# import pandas as pd

from typing import *
from pathlib import Path
# from datasets import load_dataset
from torch.utils.data import Dataset

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class SingleEq(Dataset):
    
    split_fnames = {
        'val': 'questions.json'
    }

    def __init__(self) -> None:
        self.root_dir: Path = Macros.dataset_dir / 'singleeq'
        if os.path.exists(str(self.root_dir / 'questions.json')):
            raw_data = Utils.read_json(self.root_dir / 'questions.json')
        else:
            raise('No addsub dataset file, questions.json')
        # end if

        self.id = [r['iIndex'] for r in raw_data]
        self.body = ['' for r in raw_data]
        self.question = [r['sQuestion'] for r in raw_data]
        self.answer = [r['lSolutions'][0] for r in raw_data]

    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index: int):
        return {
            'id': self.id[index],
            'body': self.body[index],
            'question': self.question[index],
            'answer': self.answer[index]
        }