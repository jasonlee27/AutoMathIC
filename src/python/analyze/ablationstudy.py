
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


class AblationStudy:

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        return Utils.get_answer_from_ground_truth(str_answer)

    @classmethod
    def get_answer_from_cot_resp(cls, cot_resp: str) -> str:
        return Utils.get_answer_from_cot_resp(cot_resp)

    @classmethod
    def get_answer_from_code_resp(cls, code_resp: str, dataset_name: str, model_name: str) -> str:
        return Utils.get_answer_from_code_resp(code_resp, dataset_name, model_name=model_name)

    @classmethod
    def get_answer_from_eqn_resp(cls, eqn_resp: str) -> str:
        return Utils.get_answer_from_eqn_resp(eqn_resp)
    
    @classmethod
    def get_answer_from_resp(
        cls, 
        response_list: List[str],
        dataset_name: str,
        modal_name: str,
        model_name: str=None
    ) -> Any:
        answer_list = list()
        if response_list is None:
            return None, 0
        # end if
        if type(response_list) is not str:
            for resp in response_list:
                if modal_name=='cot_response':            
                    answer = cls.get_answer_from_cot_resp(resp)
                elif modal_name=='code_response':
                    answer = cls.get_answer_from_code_resp(resp, dataset_name, model_name=model_name)
                else:
                    answer = cls.get_answer_from_eqn_resp(resp)
                # end if
                if answer is not None:
                    answer_list.append(answer)
                # end if
            # end for
        else:
            if modal_name=='cot_response':
                answer = cls.get_answer_from_cot_resp(response_list)
            elif modal_name=='code_response':
                answer = cls.get_answer_from_code_resp(response_list, dataset_name, model_name=model_name)
            else:
                answer = cls.get_answer_from_eqn_resp(response_list)
            # end if
            if answer is not None:
                answer_list.append(answer)
            # end if
        # end if
        if any(answer_list):
            c = Counter(answer_list)
            answer_freqs = c.most_common()
            answer = answer_freqs[0][0]
            return answer, answer_freqs
        else:
            return None, []
        # end if

    @classmethod
    def compute_consistency(
        cls,
        response_dict: Dict,
        dataset_name: str,
        model_name: str,
        answer_dict: Dict=None
    ) -> Dict:
        # compute answer and its self-consistency
        modal_weights = Macros.genetic_fg_belief_weight_over_modals
        score_dict_over_unique_answers = dict()
        if answer_dict is None:
            answer_dict = dict()
            for modal_name in Macros.prompts_over_modals.keys():
                resp_list = response_dict[modal_name]
                answer_maj, answer_freqs = cls.get_answer_from_resp(
                    resp_list, 
                    dataset_name, 
                    modal_name, 
                    model_name=model_name
                )
                answer_dict[modal_name] = dict()
                for ans_freq in answer_freqs:
                    ans = ans_freq[0]
                    self_consistency_score = 0.
                    if ans is not None:
                        self_consistency_score = ans_freq[1]*1./len(resp_list)
                        answer_dict[modal_name][ans] = {
                            'self-consistency': self_consistency_score
                        }
                        if ans not in score_dict_over_unique_answers.keys():
                            score_dict_over_unique_answers[ans] = modal_weights[modal_name]*self_consistency_score
                        else:
                            score_dict_over_unique_answers[ans] += modal_weights[modal_name]*self_consistency_score
                        # end if
                    # end if
                # end for
            # end for
        else:
            for modal_name in Macros.prompts_over_modals.keys():
                for ans in answer_dict[modal_name].keys():
                    if ans not in score_dict_over_unique_answers.keys():
                        score_dict_over_unique_answers[ans] = modal_weights[modal_name]*answer_dict[modal_name][ans]['self-consistency']
                    else:
                        score_dict_over_unique_answers[ans] += modal_weights[modal_name]*answer_dict[modal_name][ans]['self-consistency']
                    # end if
                # end for
            # end for
        # end if

        # Normalizing scores
        sum_scores = sum([
            score_dict_over_unique_answers[answer]
            for answer in score_dict_over_unique_answers.keys()
        ])
        if sum_scores>0.:
            for answer in score_dict_over_unique_answers.keys():
                score_dict_over_unique_answers[answer] = score_dict_over_unique_answers[answer]/sum_scores
            # end for
        else:
            for answer in score_dict_over_unique_answers.keys():
                score_dict_over_unique_answers[answer] = 0.
            # end for
        # end if
        if any(score_dict_over_unique_answers.values()):
            max_score = max(score_dict_over_unique_answers.values())
        else:
            max_score = 0.
        # end if
        return max_score, answer_dict, score_dict_over_unique_answers

    @classmethod
    def get_answer_by_cons_scores(
        cls, 
        cons_score_dict_over_answers: Dict
    ):
        final_ans = list()
        if not any(cons_score_dict_over_answers.values()):
            return None
        else:
            max_cons_score = max(cons_score_dict_over_answers.values())
        # end if
        for ans_per_mod, score in cons_score_dict_over_answers.items():
            if score==max_cons_score:
                final_ans.append(ans_per_mod)
            # end if
        # end for
        return final_ans[0]

    @classmethod
    def load_eval_results(cls, res_file: Path):
        # res_file = self.eval_dir / f"fg-eval-{self.cksum_val}.json"
        eval_res = Utils.read_json(res_file)
        return eval_res

    @classmethod
    def get_final_answer_without_mut_optmization(
        cls,
        target_question_response_dict: Dict,
        llm_name: str,
        dataset_name: str
    ):
        cons_orig, \
        final_ans_dict, \
        cons_score_dict_over_answers = cls.compute_consistency(
            target_question_response_dict, 
            dataset_name,
            model_name=llm_name if llm_name=='gpt4' else None
        )
        ans_by_scores = cls.get_answer_by_cons_scores(cons_score_dict_over_answers)
        return ans_by_scores

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
        if not any(correctness):
            return
        # end if
        stats = {
            'count': len(correctness),
            'sum': sum(correctness),
            'avg': Utils.avg(correctness, decimal=decimal),
            'median': Utils.median(correctness, decimal=decimal),
            'stdev': Utils.stdev(correctness, decimal=decimal) if len(correctness)>1 else 0.
        }
        return stats

    @classmethod
    def main_phaseone_only(
        cls,
        dataset_name: str,
        llm_name: str
    ) -> None:
        print(f"***** AblationStudy.main_phaseone_only::DATASET_{dataset_name}::LLM_{llm_name} *****")
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        genetic_dir = Macros.result_dir / 'genetic_fg'/ dataset_name / 'evaluate_consistency' / llm_name
        ablation_study_dir = Macros.result_dir / 'genetic_fg'/ dataset_name / 'ablation_study' / llm_name
        ablation_study_dir.mkdir(parents=True, exist_ok=True)

        eval_res_files = sorted([
            f_name for f_name in os.listdir(str(eval_dir))
            if f_name.endswith('.json') and \
                f_name.startswith('fg-eval-') and \
                (not f_name.startswith('eval-results-w'))
        ])
        correctness_list = list()
        acc_res_list = list()
        for eval_res_file in eval_res_files:
            print(f"DATASET_{dataset_name}::LLM_{llm_name}::{eval_res_file}")
            cksum_val = eval_res_file.split('-')[-1].split('.json')[0].strip()
            eval_res = cls.load_eval_results(eval_dir / eval_res_file)
            final_answer = cls.get_final_answer_without_mut_optmization(
                eval_res['orig'],
                llm_name,
                dataset_name
            )
            correctness = cls.get_correctness(
                final_answer,
                eval_res['orig']['answer']
            )
            correctness_list.append(correctness)
            acc_res = {
                'cksum_val': cksum_val,
                'phase_one_only_correctness': correctness
            }
            acc_res_list.append(acc_res)
        # end for
        acc_stats = cls.get_stats(correctness_list)
        Utils.write_json(
            acc_res_list,
            ablation_study_dir / 'acc-results-phaseone-only.json',
            pretty_format=True
        )
        Utils.write_json(
            acc_stats,
            ablation_study_dir / 'acc-phaseone-only.json',
            pretty_format=True
        )
        return

    @classmethod
    def analyze(
        cls,
        dataset_name: str,
        llm_name: str
    ) -> None:
        print(f"***** AblationStudy.analyze::DATASET_{dataset_name}::LLM_{llm_name} *****")
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        genetic_dir = Macros.result_dir / 'genetic_fg'/ dataset_name / 'evaluate_consistency' / llm_name
        ablation_study_dir = Macros.result_dir / 'genetic_fg'/ dataset_name / 'ablation_study' / llm_name
        ablation_study_res_file = ablation_study_dir / 'acc-results-phaseone-only.json'

        genetic_fg_res_file = genetic_dir / 'acc-res-genetic-fg.json'
        genetic_fg_file = genetic_dir / 'final_answers.json'
        ab_analyze_res_file = ablation_study_dir / 'analyze-results-phaseone-only.json'

        ablation_study_res = Utils.read_json(ablation_study_res_file)
        genetic_fg_res = Utils.read_json(genetic_fg_res_file)
        genetic_fg_resps = Utils.read_json(genetic_fg_file)
        both_fail_list = list()
        only_ab_fail_list = list()
        only_gf_fail_list = list()
        orig_score_list = list()
        fin_score_list = list()
        num_mut_for_both_fail_list = list()
        num_mut_for_only_ab_fail_list = list()
        num_mut_for_only_gf_fail_list = list()
        for ablation_study_res_per_cksum in ablation_study_res:
            cksum_val = ablation_study_res_per_cksum['cksum_val']
            gf_res_per_cksum = genetic_fg_res[cksum_val]
            ab_correctness = ablation_study_res_per_cksum['phase_one_only_correctness']
            gf_correctness = genetic_fg_res[cksum_val]['genetic']
            num_muts_gf = genetic_fg_res[cksum_val]['num_demo_used']

            orig_score = genetic_fg_resps[cksum_val]['original_consistency_score']
            fin_score = genetic_fg_resps[cksum_val]['final_consistency_score']
            orig_score_list.append(orig_score)
            fin_score_list.append(fin_score)

            if ab_correctness==0. and gf_correctness==0.:
                both_fail_list.append(cksum_val)
                num_mut_for_both_fail_list.append(num_muts_gf)
            elif ab_correctness==1. and gf_correctness==0.:
                only_gf_fail_list.append(cksum_val)
                num_mut_for_only_gf_fail_list.append(num_muts_gf)
            elif ab_correctness==0. and gf_correctness==1.:
                only_ab_fail_list.append(cksum_val)
                num_mut_for_only_ab_fail_list.append(num_muts_gf)
            # end if
        # end for
        Utils.write_json({
                'both_fail_list': both_fail_list,
                'num_mut_for_both_fail_list': num_mut_for_both_fail_list,
                'num_mut_for_both_fail_stats': cls.get_stats(num_mut_for_both_fail_list),
                'only_gf_fail_list': only_gf_fail_list,
                'num_mut_for_only_gf_fail_list': num_mut_for_only_gf_fail_list,
                'num_mut_for_only_gf_fail_stats': cls.get_stats(num_mut_for_only_gf_fail_list),
                'only_ab_fail_list': only_ab_fail_list,
                'num_mut_for_only_ab_fail_list': num_mut_for_only_ab_fail_list,
                'num_mut_for_only_ab_fail_stats': cls.get_stats(num_mut_for_only_ab_fail_list),
                'orig_consistency_score_stats': cls.get_stats(orig_score_list),
                'final_consistency_score_stats': cls.get_stats(fin_score_list)
            },
            ab_analyze_res_file,
            pretty_format=True
        )
        return
            
            
