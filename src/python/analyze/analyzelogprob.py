
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


class AnalyzeLogprob:

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
    def get_logprobs_agg_from_modals(
        cls, 
        eval_res: Dict,
        modal_name: str
    ):
        logprob_list = eval_res[modal_name]['logprob']
        return sum(logprob_list), sum(logprob_list)/len(logprob_list)

    @classmethod
    def get_correctness(
        cls, 
        query_answer: str, 
        ref_sent: Any
    ):
        _ref_sent = cls.get_answer_from_ground_truth(ref_sent)
        if type(_ref_sent)==str and _ref_sent!='<N/A>':
            try:
                ref_sent = eval(_ref_sent)
            except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                ref_sent = _ref_sent
                pass
            # end try
        # end if
        if type(query_answer)==str and query_answer!='<N/A>':
            try:
                correctness = 1. if eval(query_answer)==ref_sent else 0.
            except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                correctness = 1. if query_answer==ref_sent else 0.
                pass
            # end try
        else:
            correctness = 1. if query_answer==ref_sent else 0.
        # end if
        return correctness

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
    def main(cls,
        llm_name: str, 
        dataset_name: str,
        pl_type: str='python',
    ) -> None:

        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / llm_name
        llm_response_files = [
            (
                f, 
                re.search(
                    r'eval\-(.*)\.json', f
                ).group(1) # checksum_val
            ) for f in os.listdir(str(eval_dir))
            if f.startswith('eval-') and f.endswith('.json') and \
            (not f.startswith('eval-results-w'))
        ]
        
        for mod_name in list(Macros.prompts_over_modals.keys()):
            acc_res = dict()
            acc_stats = dict()
            correctness_list = list()
            logprob_sum_correct = list()
            logprob_sum_incorrect = list()
            logprob_avg_correct = list()
            logprob_avg_incorrect = list()
            # if os.path.exists(str(eval_dir / f"acc-res-bl-{mod_name}.json")):
            #     acc_res = Utils.read_json(eval_dir / f"acc-res-bl-{mod_name}.json")
            #     correctness_list = [
            #         acc_res[k]['orig'] for k in acc_res.keys()
            #     ]
            # # end if

            for eval_file, cksum_val in llm_response_files:
                if cksum_val not in acc_res.keys():
                    res_orig: Dict = Utils.read_json(eval_dir / eval_file)
                    
                    orig_ans = cls.get_answers_from_modals(
                        res_orig['orig'],
                        mod_name,
                        dataset_name
                    )

                    orig_logprobs_sum, orig_logprobs_avg = cls.get_logprobs_agg_from_modals(
                        res_orig['orig'],
                        mod_name
                    )
                    
                    correctness = cls.get_correctness(
                        orig_ans,
                        res_orig['orig']['answer']
                    )

                    # acc_res[cksum_val] = {
                    #     'orig': correctness,
                    #     'logprob_sum': orig_logprobs_sum,
                    #     'logprob_avg': orig_logprobs_avg
                    # }

                    correctness_list.append(correctness)
                    if correctness==1.:
                        logprob_sum_correct.append(orig_logprobs_sum)
                        logprob_avg_correct.append(orig_logprobs_avg)
                    else:
                        logprob_sum_incorrect.append(orig_logprobs_sum)
                        logprob_avg_incorrect.append(orig_logprobs_avg)
                    # end if
                    print(f"{mod_name}::{cksum_val}::{correctness}")
                    
                    # Utils.write_json(
                    #     acc_res,
                    #     eval_dir / f"analyze-acc-res-bl-{mod_name}.json",
                    #     pretty_format=True
                    # )
                # end if
            # end for

            acc_stats = {
                'acc-orig': cls.get_stats(correctness_list),
                'logprob_sum_correct': logprob_sum_correct,
                'logprob_sum_incorrect': logprob_sum_incorrect,
                'logprob_avg_correct': logprob_avg_correct,
                'logprob_avg_incorrect': logprob_avg_incorrect
            }
            Utils.write_json(
                acc_stats, 
                eval_dir / f"analyze-acc-stats-bl-{mod_name}.json",
                pretty_format=True
            )
        # end for
        return

