
import re, os
import sys
import time
import random
import numpy as np
from scipy.stats import ttest_ind

from typing import *
from pathlib import Path

from ..utils.macros import Macros
from ..utils.utils import Utils


class AnalyzeSuspiciousness:

    @classmethod
    def get_stats(
        cls,
        correctness: List[float],
        decimal=3
    ) -> Dict[str, float]:
        stats = {
            'count': len(correctness),
            'sum': sum(correctness),
            'avg': Utils.avg(correctness, decimal=decimal),
            'median': Utils.median(correctness, decimal=decimal),
            'stdev': Utils.stdev(correctness, decimal=decimal)
        }
        return stats

    @classmethod
    def get_correctness(
        cls,
        dataset_name: str,
        ext_dataset_name: str,
        retrival_method: str = 'embedding'
    ):
        sus_dir = Macros.result_dir / 'suspiciousness'
        retrieve_data_file = sus_dir / f"retrieved-data-using-{retrival_method}-{dataset_name}-from-ext-{ext_dataset_name}.json"
        retrieval_res: List[Dict] = Utils.read_json(retrieve_data_file)
        correctness_dict = {
            'num_retrieved_data': 5,
            'query_correct': list(),
            'query_incorrect': list(),
            'stats_on_query_correct': None,
            'stats_on_query_incorrect': None
        }
        for d_i, d in enumerate(retrieval_res):
            query_correctness = d['query_correctness']
            ret_correctness_score = 0.
            for rd_i, rd in enumerate(d['retrieved_texts_from_ext_dataset'].keys()):
                ret_score = d['retrieved_texts_from_ext_dataset'][rd]['score']
                ret_answer = d['retrieved_texts_from_ext_dataset'][rd]['answer']
                ret_cot_resp = d['retrieved_texts_from_ext_dataset'][rd]['cot_resp']
                ret_correctness = d['retrieved_texts_from_ext_dataset'][rd]['correctness']
                ret_correctness_score += 1. if ret_correctness else 0.
            # end for
            ret_correctness_score = ret_correctness_score/(1.*len(d['retrieved_texts_from_ext_dataset'].keys()))
            if query_correctness:
                correctness_dict['query_correct'].append(ret_correctness_score)
            else:
                correctness_dict['query_incorrect'].append(ret_correctness_score)
            # end if
        # end for
        correctness_dict['stats_on_query_correct'] = cls.get_stats(correctness_dict['query_correct'])
        correctness_dict['stats_on_query_incorrect'] = cls.get_stats(correctness_dict['query_incorrect'])
        t_value, p_value = ttest_ind(
            correctness_dict['query_correct'], 
            correctness_dict['query_incorrect']
        )
        alpha = 0.05
        correctness_dict['significance_ttest'] = {
            't_value': round(t_value, 5), 
            'p_value': round(p_value, 5),
            'alpha': alpha,
            'significance': False
        }
        if p_value <= alpha:
            correctness_dict['significance_ttest']['significance'] = True
        # end if
        Utils.write_json(
            correctness_dict,
            sus_dir / f"analysis-retrieved-data-using-{retrival_method}-{dataset_name}-from-ext-{ext_dataset_name}.json",
            pretty_format=True
        )
        return