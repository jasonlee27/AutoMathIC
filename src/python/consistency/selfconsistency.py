
import os
import re
import math
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import *
from pathlib import Path
from numpy.linalg import norm

from ..utils.utils import Utils
from ..utils.macros import Macros
from ..utils.logger import Logger


class SelfConsistency:

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        return Utils.get_answer_from_ground_truth(str_answer)

    @classmethod
    def get_answer_from_cot_resp(cls, cot_resp: str) -> str:
        return Utils.get_answer_from_cot_resp(cot_resp)

    @classmethod
    def get_answer_from_code_resp(cls, code_resp: str, dataset_name: str, model_name: str=None) -> str:
        return Utils.get_answer_from_code_resp(code_resp, dataset_name, model_name=model_name)

    @classmethod
    def get_answer_from_eqn_resp(cls, eqn_resp: str) -> str:
        return Utils.get_answer_from_eqn_resp(eqn_resp)

    @classmethod
    def get_answer(cls, resp: str, mod_name: str, dataset_name: str, model_name: str=None) -> str:
        if mod_name=='cot_response':
            answer = cls.get_answer_from_cot_resp(resp)
        elif mod_name=='code_response':
            answer = cls.get_answer_from_code_resp(resp, dataset_name, model_name=model_name)
        elif mod_name=='eqn_response':
            answer = cls.get_answer_from_eqn_resp(resp)
        # end if
        return answer

    @classmethod
    def compute_self_consistency(
        cls,
        resp_dict: Dict[str, List],
        dataset_name: str,
        target_modality: str,
        model_name: str
    ) -> float:
        assert target_modality in Macros.prompts_over_modals.keys()
        
        answer_dict = {
            target_modality: list()
        }
        for resp in resp_dict[target_modality]:
            ans = cls.get_answer(
                resp, 
                target_modality, 
                dataset_name,
                model_name=model_name
            )
            answer_dict[target_modality].append(ans)
            # try:
            #     answer_dict[mod_name].append(eval(ans))

            # except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
            #     answer_dict[mod_name].append(ans)
            #     pass
            # # end try
        # end for
        if not any(answer_dict[target_modality]):
            return 0., answer_dict
        # end if
        self_consistency = 1.-len(set(answer_dict[target_modality]))*1. / len(answer_dict[target_modality])
        return self_consistency, answer_dict
