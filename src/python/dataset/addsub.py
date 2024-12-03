

import os
import csv
import numpy as np
# import pandas as pd
import xml.etree.ElementTree as ET

from typing import *
from pathlib import Path
# from datasets import load_dataset
from torch.utils.data import Dataset

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class Addsub(Dataset):
    
    split_fnames = {
        'val': 'addsub.json'
    }

    def __init__(self) -> None:
        self.root_dir: Path = Macros.dataset_dir / 'addsub'
        if os.path.exists(str(self.root_dir / 'addsub.json')):
            raw_data = Utils.read_json(self.root_dir / 'addsub.json')
        else:
            raise('No addsub dataset file, addsub.json')
        # end if

        instances = raw_data['Instances']
        self.id = [inst_i+1 for inst_i, _ in enumerate(instances)]
        self.body = ['' for inst in instances]
        self.question = [inst['Input'] for inst in instances]
        self.answer = [inst['Output Answer'][0] for inst in instances]
        self.program = [inst['Output Program'][0] for inst in instances]

    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index: int):
        return {
            'id': self.id[index],
            'body': self.body[index],
            'question': self.question[index],
            'answer': self.answer[index],
            'equation': self.program[index]
        }