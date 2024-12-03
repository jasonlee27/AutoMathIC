
import os
import re
import math
import random
import numpy as np

from typing import *
from pathlib import Path
from collections import Counter
from scipy.stats import ttest_ind

from ..utils.macros import Macros
from ..utils.utils import Utils


class AnalyzeAutocot:

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        return Utils.get_answer_from_ground_truth(str_answer)

    @classmethod
    def get_answer_from_cot_resp(cls, cot_resp: str) -> str:
        return Utils.get_answer_from_cot_resp(cot_resp)

    @classmethod
    def get_answer_from_code_resp(cls, code_resp: str, dataset_name: str) -> str:
        return Utils.get_answer_from_code_resp(code_resp, dataset_name)

    @classmethod
    def get_answer_from_eqn_resp(cls, eqn_resp: str) -> str:
        return Utils.get_answer_from_eqn_resp(eqn_resp)
    
    @classmethod
    def get_answers_from_modals(
        cls, 
        eval_res: Dict,
        modal_name: str,
        dataset_name: str,
        cksum_val: str=None
    ):
        if cksum_val is not None:
            resp = eval_res[cksum_val]['final_response'][modal_name]['msg'].strip().lower()
        else:
            resp = eval_res[modal_name]['msg'].strip().lower()
        # end if
        answers_over_modals = None
        if modal_name=='cot_response':
            answers_over_modals = cls.get_answer_from_cot_resp(resp)
        elif modal_name=='code_response':
            answers_over_modals = cls.get_answer_from_code_resp(resp, dataset_name)
        elif modal_name=='eqn_response':
            answers_over_modals = cls.get_answer_from_eqn_resp(resp)
        # end if
        return answers_over_modals


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
    def bootstrap(cls,
        full_a_scores: List[float],
        full_b_scores: List[float],
        num_samples: int = 10_000,
        test_size: int  = 1000,
        is_pairwise: bool = True,
    ) -> float:
        """
        Uses bootstrap method to perform the statistical significance test for the hypothesis: a > b, and returns the p-value.
        :param full_a_scores: the full set of a values to sample from
        :param full_b_scores: the full set of b values to sample from
        :param num_samples: the number of samples to repeat
        :param test_size: the size at each sample
        :param is_pairwise: if the test is pairwise (require len(full_a_scores) == len(full_b_scores))
        :return: the p-value, which should be smaller than or equal to the significance level (usually 5%) to claim as statistical significant
        """
        if is_pairwise and len(full_a_scores) != len(full_b_scores):
            raise("Cannot perform pairwise significance test if two sets do not have the same size.")
        # end if

        if len(full_a_scores) < test_size:
            raise(f"Test size {test_size} bigger than set a's size {len(full_a_scores)}.")
        # end if
        if len(full_b_scores) < test_size:
            raise(f"Test size {test_size} bigger than set b's size {len(full_b_scores)}.")
        # end if

        a_scores = []
        b_scores = []

        for sample in range(num_samples):
            if is_pairwise:
                sampled_indices = random.choices(range(len(full_a_scores)), k=test_size)
                selected_a_scores = [full_a_scores[a] for a in sampled_indices]
                selected_b_scores = [full_b_scores[b] for b in sampled_indices]
            else:
                selected_a_scores = random.choices(full_a_scores, k=test_size)
                selected_b_scores = random.choices(full_b_scores, k=test_size)
            # end if

            a_score = sum(selected_a_scores) / float(len(selected_a_scores))
            b_score = sum(selected_b_scores) / float(len(selected_b_scores))

            a_scores.append(a_score)
            b_scores.append(b_score)
        # end for

        significance_a_over_b = 0
        for i in range(num_samples):
            if a_scores[i] > b_scores[i]:
                significance_a_over_b += 1
            # end if
        # end for
        return 1 - significance_a_over_b / float(num_samples)

    @classmethod
    def main(cls,
        llm_name: str, 
        dataset_name: str,
        pl_type: str='python',
    ) -> None:
        # to see the number of mutation used for the case of autocot passes but genetic-fg failed.
        # dataset_names = [
        #     'asdiv', 'svamp', 'multiarith', 'gsm8k'
        # ]
        SIGNIFICANCE_LEVEL = 0.05
        autocot_res_dir = Macros.result_dir / 'autocot' / 'evaluate' / dataset_name / llm_name
        genetic_fg_res_dir = Macros.result_dir / 'genetic_fg' / dataset_name
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / llm_name
        genetic_fg_res_file = genetic_fg_res_dir / 'evaluate_consistency' / llm_name / 'acc-res-genetic-fg.json'
        autocot_res_file = autocot_res_dir / 'acc-res-autocot.json'        

        genetic_fg_res: Dict = Utils.read_json(genetic_fg_res_file)
        autocot_res: Dict = Utils.read_json(autocot_res_file)

        num_muts_for_genetic_fg_fail_list = list()
        num_muts_for_genetic_fg_succ_list = list()

        for cksum_val in genetic_fg_res.keys():
            if genetic_fg_res[cksum_val]['genetic']==0.:
                num_muts_for_genetic_fg_fail_list.append(
                    genetic_fg_res[cksum_val]['num_demo_used']
                )
            elif genetic_fg_res[cksum_val]['genetic']==1.:
                num_muts_for_genetic_fg_succ_list.append(
                    genetic_fg_res[cksum_val]['num_demo_used']
                )
            # end if
        # end for

        a_list = num_muts_for_genetic_fg_succ_list
        b_list = num_muts_for_genetic_fg_fail_list

        pval = cls.bootstrap(
            a_list,
            b_list,
            num_samples=10_000,
            test_size=min(len(a_list), len(b_list)),
            is_pairwise= False
        )
        str_significant = "is" if pval <= SIGNIFICANCE_LEVEL else "is not"
        avg_num_muts_for_genetic_fg_fail = sum(num_muts_for_genetic_fg_fail_list)/len(num_muts_for_genetic_fg_fail_list)
        avg_num_muts_for_genetic_fg_succ = sum(num_muts_for_genetic_fg_succ_list)/len(num_muts_for_genetic_fg_succ_list)
        print(f"NUM_MUTS<genetic_fg:fail>::{avg_num_muts_for_genetic_fg_fail}")
        print(f"NUM_MUTS<genetic_fg:succ>::{avg_num_muts_for_genetic_fg_succ}")
        print(f"{str_significant} significant::{pval}")
        return

    # @classmethod
    # def main(cls,
    #     llm_name: str, 
    #     pl_type: str='python',
    # ) -> None:
    #     # to see the number of mutation used for the case of autocot passes but genetic-fg failed.
    #     dataset_names = [
    #         'asdiv', 'svamp', 'multiarith', 'gsm8k'
    #     ]
    #     SIGNIFICANCE_LEVEL = 0.05

    #     for dataset_name in dataset_names:

    #         autocot_res_dir = Macros.result_dir / 'autocot' / 'evaluate' / dataset_name / llm_name
    #         genetic_fg_res_dir = Macros.result_dir / 'genetic_fg' / dataset_name
    #         res_dir = Macros.result_dir / 'nl2nl' / dataset_name
    #         eval_dir = res_dir / 'evaluate_consistency' / llm_name
    #         genetic_fg_res_file = genetic_fg_res_dir / 'evaluate_consistency' / llm_name / 'acc-res-genetic-fg.json'
    #         autocot_res_file = autocot_res_dir / 'acc-res-autocot.json'        

    #         genetic_fg_res: Dict = Utils.read_json(genetic_fg_res_file)
    #         autocot_res: Dict = Utils.read_json(autocot_res_file)

    #         num_muts_for_genetic_fg_fail_list = list()
    #         num_muts_for_genetic_fg_succ_list = list()

    #         for cksum_val in genetic_fg_res.keys():
    #             if genetic_fg_res[cksum_val]['genetic']==0. and \
    #                 autocot_res[cksum_val]==1.:
    #                 num_muts_for_genetic_fg_fail_list.append(
    #                     genetic_fg_res[cksum_val]['num_demo_used']
    #                 )
    #             elif genetic_fg_res[cksum_val]['genetic']==1. and \
    #                 autocot_res[cksum_val]==0.:
    #                 num_muts_for_genetic_fg_succ_list.append(
    #                     genetic_fg_res[cksum_val]['num_demo_used']
    #                 )
    #             # end if
    #         # end for
            
    #     # end for

    #     pval = cls.bootstrap(
    #         num_muts_for_genetic_fg_succ_list,
    #         num_muts_for_genetic_fg_fail_list,
    #         num_samples=10_000,
    #         test_size=min(
    #             len(num_muts_for_genetic_fg_succ_list), 
    #             len(num_muts_for_genetic_fg_fail_list)
    #         ),
    #         is_pairwise= False
    #     )
    #     str_significant = "is" if pval <= SIGNIFICANCE_LEVEL else "is not"

    #     avg_num_muts_for_genetic_fg_fail = sum(num_muts_for_genetic_fg_fail_list)/len(num_muts_for_genetic_fg_fail_list)
    #     avg_num_muts_for_genetic_fg_succ = sum(num_muts_for_genetic_fg_succ_list)/len(num_muts_for_genetic_fg_succ_list)
    #     print(f"NUM_MUTS<genetic_fg:fail&autocot:succ>::{avg_num_muts_for_genetic_fg_fail}")
    #     print(f"NUM_MUTS<genetic_fg:succ&autocot:fail>::{avg_num_muts_for_genetic_fg_succ}")
    #     print(f"{str_significant} significant::{pval}")
    #     return
