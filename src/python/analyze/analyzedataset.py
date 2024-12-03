
import re, os
import sys
import time
import random
import argparse

from typing import *
from pathlib import Path

from ..dataset.asdiv import Asdiv
from ..dataset.gsm8k import Gsm8k
from ..dataset.svamp import Svamp
from ..dataset.multiarith import MultiArith
from ..dataset.addsub import Addsub

from ..utils.macros import Macros
from ..utils.utils import Utils


class AnalyzeDataset:

    @classmethod
    def read_dataset(cls, dataset_name: str) -> Any:
        dataset_obj = None
        if dataset_name=='asdiv':
            dataset_obj = Asdiv()
        elif dataset_name=='gsm8k':
            dataset_obj = Gsm8k()
        elif dataset_name=='svamp':
            dataset_obj = Svamp()
        elif dataset_name=='multiarith':
            dataset_obj = MultiArith()
        elif dataset_name=='addsub':
            dataset_obj = Addsub()
        # end if
        return dataset_obj

    @classmethod
    def get_num_tokens(cls, sent: str) -> int:
        tokens = Utils.tokenize(sent)
        return len(tokens)
        
    @classmethod
    def get_tokens_stat(cls, dataset_name: str) -> None:
        res_dir = Macros.result_dir / 'dataset' / 'stat'
        res_dir.mkdir(parents=True, exist_ok=True)
        dataset = cls.read_dataset(dataset_name)
        num_tokens_list = list()
        for d_i in range(len(dataset)):
            d = dataset[d_i]
            body = d.get('body', '')
            question = d.get('question', '')
            inp = f"{body} {question}"
            num_tokens_list.append(
                cls.get_num_tokens(inp)
            )
        # end for
        stat = {
            'sum': sum(num_tokens_list),
            'avg': Utils.avg(num_tokens_list),
            'med': Utils.median(num_tokens_list),
            'std': Utils.stdev(num_tokens_list)
        }
        Utils.write_json(
            stat, 
            res_dir / f"{dataset_name}-input-tokens-stat.json",
            pretty_format=True
        )
        return