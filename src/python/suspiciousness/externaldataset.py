
import os
import numpy as np
# import pandas as pd

from typing import *
from pathlib import Path
# from datasets import load_dataset
from torch.utils.data import Dataset

from ..dataset.asdiv import Asdiv
from ..dataset.gsm8k import Gsm8k
from ..dataset.svamp import Svamp

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class ExternalDataset:

    @classmethod
    def read_dataset(cls, dataset_name: str) -> List[Dict]:
        dataset_obj = None
        if dataset_name=='asdiv':
            dataset_obj = Asdiv()
        elif dataset_name=='gsm8k':
            dataset_obj = Gsm8k()
        elif dataset_name=='svamp':
            dataset_obj = Svamp()
        # end if
        return dataset_obj

