
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


class AnalyzeMutation:

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
    def significance_test(
        cls,
        a_list: List,
        b_list: List,
        sample_size: int,
        n: int =1000
    ):
        cnt = 0
        for n_i in range(n):
            random.seed(n_i)
            a_sample = random.sample(a_list,n)
            b_sample = random.sample(b_list,n)
            avg_a = sum(a_sample)/len(a_sample)
            avg_b = sum(b_sample)/len(b_sample)
            if avg_a>avg_b:
                cnt += 1
            # end if
        # end for
        return cnt/n

    @classmethod
    def get_sample_data(
        cls,
        dataset_name: str,
        llm_name: str,
        sample_size: int
    ) -> None:
        random.seed(Macros.RAND_SEED)

        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        # eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_res_dir = Macros.result_dir / 'genetic'/ dataset_name / 'evaluate_consistency' / llm_name
        # genetic_dir.mkdir(parents=True, exist_ok=True)
        eval_res_files = sorted([
            f_name for f_name in os.listdir(str(eval_dir))
            if f_name.endswith('.json') and \
                f_name.startswith('eval-') and \
                (not f_name.startswith('eval-results-w'))
        ])
        
        genetic_res_file = genetic_res_dir / 'final_answers.json'
        genetic_res = Utils.read_json(genetic_res_file)
        genetic_acc_res_file = genetic_res_dir / 'acc-res-genetic.json'
        genetic_acc_res = Utils.read_json(genetic_acc_res_file)
        genetic_acc_bl_cot_res_file = eval_dir / 'acc-res-bl-cot_response.json'
        genetic_acc_bl_cot_res = Utils.read_json(genetic_acc_bl_cot_res_file)

        data_using_muts = dict()
        cksum_vals_using_muts = list()
        orig_const_scores = list()
        final_const_scores = list()
        bl_cot_correctness_using_muts = list()
        bl_cot_correctness_not_using_muts = list()
        bl_maj_correctness_using_muts = list()
        bl_maj_correctness_not_using_muts = list()
        correctness_using_muts = list()
        correctness_not_using_muts = list()

        for eval_res_file in eval_res_files:
            cksum_val = eval_res_file.split('-')[-1].split('.json')[0].strip()
            if genetic_res.get(cksum_val, None) is not None:
                muts_used = genetic_res[cksum_val]['mutation']
                if len(muts_used)>0:
                    cksum_vals_using_muts.append(cksum_val)
                    bl_cot_correctness_using_muts.append(genetic_acc_bl_cot_res[cksum_val]['orig'])
                    bl_maj_correctness_using_muts.append(genetic_acc_res[cksum_val]['maj'])
                    correctness_using_muts.append(genetic_acc_res[cksum_val]['genetic'])
                    # print(cksum_val) # just printing at window for easier copy/paste into excel
                else:
                    bl_cot_correctness_not_using_muts.append(genetic_acc_bl_cot_res[cksum_val]['orig'])
                    bl_maj_correctness_not_using_muts.append(genetic_acc_res[cksum_val]['maj'])
                    correctness_not_using_muts.append(genetic_acc_res[cksum_val]['genetic'])
                # end if
            # end if
        # end for
        print(f"{dataset_name}::{len(cksum_vals_using_muts)}_OUTOF_{len(eval_res_files)}")

        bl_cot_correctness_using_muts_stat = cls.get_stats(bl_cot_correctness_using_muts)
        bl_cot_correctness_not_using_muts_stat = cls.get_stats(bl_cot_correctness_not_using_muts)
        bl_maj_correctness_using_muts_stat = cls.get_stats(bl_maj_correctness_using_muts)
        bl_maj_correctness_not_using_muts_stat = cls.get_stats(bl_maj_correctness_not_using_muts)
        correctness_using_muts_stat = cls.get_stats(correctness_using_muts)
        correctness_not_using_muts_stat = cls.get_stats(correctness_not_using_muts)

        ttest_res_btw_ours_n_bl_cot = ttest_ind(correctness_using_muts, bl_cot_correctness_using_muts)
        ttest_res_btw_ours_n_bl_maj = ttest_ind(correctness_using_muts, bl_maj_correctness_using_muts)

        print('correctness_using_muts_stat:')
        print(correctness_using_muts_stat)
        print('bl_cot_correctness_using_muts_stat:')
        print(bl_cot_correctness_using_muts_stat)
        print('bl_maj_correctness_using_muts_stat:')
        print(bl_maj_correctness_using_muts_stat)
        print('-----')
        print('correctness_not_using_muts_stat:')
        print(correctness_not_using_muts_stat)
        print('bl_cot_correctness_not_using_muts_stat:')
        print(bl_cot_correctness_not_using_muts_stat)
        print('bl_maj_correctness_not_using_muts_stat:')
        print(bl_maj_correctness_not_using_muts_stat)
        print('-----')
        print('ttest_res_btw_ours_n_bl_cot:')
        print(ttest_res_btw_ours_n_bl_cot)
        print('ttest_res_btw_ours_n_bl_maj')
        print(ttest_res_btw_ours_n_bl_maj)
        print('=====\n')
        # for cksum_val in cksum_vals_using_muts:
        #     print(f"\"{cksum_val}\"")
        # # end for
        # random.sample()

        # randomly sample
        # eval_res_files = random.sample(eval_res_files, sample_size)
        # print(dataset_name)
        return


class AnalyzeConsistency:

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
    def get_mod_consistency(cls, resp_dict: Dict, dataset_name: str):
        ans_dict = dict()
        for mod_name in Macros.prompts_over_modals.keys():
            resp = resp_dict[mod_name]
            ans = None
            if mod_name=='cot_response':
                ans = cls.get_answer_from_cot_resp(resp)
            elif mod_name=='code_response':
                ans = cls.get_answer_from_code_resp(resp, dataset_name)
            elif mod_name=='eqn_response':  
                ans = cls.get_answer_from_eqn_resp(resp)
            # end if
            ans_dict[mod_name] = ans
        # end for

        c = Counter(ans_dict.values())
        value_freqs = c.most_common()
        if len(set([v[1] for v in value_freqs]))==1:
            # in case of having no most frequent answer
            return 1, ans_dict
        else:
            return value_freqs[0][1], ans_dict # (value, freq)
        # end if

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
    def get_consistency_stats(
        cls,
        dataset_name: str,
        llm_name: str
    ) -> None:
        random.seed(Macros.RAND_SEED)

        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        # eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_res_dir = Macros.result_dir / 'genetic'/ dataset_name / 'evaluate_consistency' / llm_name
        # genetic_dir.mkdir(parents=True, exist_ok=True)
        eval_res_files = sorted([
            f_name for f_name in os.listdir(str(eval_dir))
            if f_name.endswith('.json') and \
                f_name.startswith('eval-') and \
                (not f_name.startswith('eval-results-w'))
        ])
        
        genetic_res_file = genetic_res_dir / 'final_answers.json'
        selfconst_res_file = genetic_res_dir / 'final_answers_w_self_consistency_over_cot_response.json'
        genetic_res = Utils.read_json(genetic_res_file)
        selfconst_res = Utils.read_json(selfconst_res_file)
        # data_using_muts = dict()
        # cksum_vals_using_muts = list()
        orig_mod_const_scores = list()
        final_mod_const_scores = list()
        orig_self_const_scores = list()
        final_self_const_scores = list()

        for eval_res_file in eval_res_files:
            cksum_val = eval_res_file.split('-')[-1].split('.json')[0].strip()
            if genetic_res.get(cksum_val, None) is not None:
                orig_mod_const_score = genetic_res[cksum_val]['original_consistency_score']
                final_mod_const_score = genetic_res[cksum_val]['final_consistency_score']
                orig_mod_const_scores.append(orig_mod_const_score)
                final_mod_const_scores.append(final_mod_const_score)
            # end if
            if selfconst_res is not None:
                if selfconst_res.get(cksum_val, None) is not None:
                    orig_self_const_score = selfconst_res[cksum_val]['original_consistency_score']
                    final_self_const_score = selfconst_res[cksum_val]['final_consistency_score']
                    orig_self_const_scores.append(orig_self_const_score)
                    final_self_const_scores.append(final_self_const_score)
                    # if genetic_res.get(cksum_val, None) is not None:
                    #     muts_used = genetic_res[cksum_val]['mutation']
                    #     if len(muts_used)>0:
                    #         cksum_vals_using_muts.append(cksum_val)
                    #         # print(cksum_val) # just printing at window for easier copy/paste into excel
                    #     # end if
                    # # end if
                # end if
            # end if
        # end for
        orig_mod_score_stats = cls.get_stats(orig_mod_const_scores)
        final_mod_score_stats = cls.get_stats(final_mod_const_scores)
        orig_self_score_stats = cls.get_stats(orig_self_const_scores) if any(orig_self_const_scores) else None
        final_self_score_stats = cls.get_stats(final_self_const_scores) if any(final_self_const_scores) else None

        print(f"{dataset_name}::{len(eval_res_files)}")
        print(f"orig_mod_const_score_stat:")
        print(orig_mod_score_stats)
        print(f"final_mod_const_score_stat:")
        print(final_mod_score_stats)
        print('-----')
        print(f"orig_self_const_score_stat:")
        print(orig_self_score_stats)
        print(f"final_self_const_score_stat:")
        print(final_self_score_stats)
        print('==========\n')


        if dataset_name in ['multiarith', 'asdiv']:
            genetic_res_file = genetic_res_dir / 'acc-res-genetic.json'
            selfconst_res_file = genetic_res_dir / 'acc-res-genetic-w-selfconst.json'
            genetic_res = Utils.read_json(genetic_res_file)
            selfconst_res = Utils.read_json(selfconst_res_file)

            genetic_ans_file = genetic_res_dir / 'final_answers.json'
            selfconst_ans_file = genetic_res_dir / 'final_answers_w_self_consistency_over_cot_response.json'
            genetic_ans = Utils.read_json(genetic_ans_file)
            selfconst_ans = Utils.read_json(selfconst_ans_file)

            # data_using_muts = dict()
            # cksum_vals_using_muts = list()
            num_mod_const_correct_n_no_mut = list()
            num_mod_const_correct_n_mut = list()
            orig_mod_const_score_n_mut = list()
            final_mod_const_score_n_mut = list()
            orig_mod_const_score_n_no_mut = list()
            final_mod_const_score_n_no_mut = list()

            num_self_const_correct_n_no_mut = list()
            num_self_const_correct_n_mut = list()
            orig_self_const_score_n_mut = list()
            final_self_const_score_n_mut = list()
            orig_self_const_score_n_no_mut = list()
            final_self_const_score_n_no_mut = list()

            for eval_res_file in eval_res_files:
                cksum_val = eval_res_file.split('-')[-1].split('.json')[0].strip()
                if genetic_res is not None:
                    if genetic_res.get(cksum_val, None) is not None:
                        mod_const_correctness = genetic_res[cksum_val]['genetic']
                        mod_const_num_mut = genetic_res[cksum_val]['num_demo_used']
                        orig_mod_const_score = genetic_ans[cksum_val]['original_consistency_score']
                        final_mod_const_score = genetic_ans[cksum_val]['final_consistency_score']

                        if mod_const_num_mut>0:
                            num_mod_const_correct_n_mut.append(mod_const_correctness)
                            orig_mod_const_score_n_mut.append(orig_mod_const_score)
                            final_mod_const_score_n_mut.append(final_mod_const_score)
                        else:
                            num_mod_const_correct_n_no_mut.append(mod_const_correctness)
                            orig_mod_const_score_n_no_mut.append(orig_mod_const_score)
                            final_mod_const_score_n_no_mut.append(final_mod_const_score)
                        # end if
                    # end if
                # end if
                if selfconst_res is not None:
                    if selfconst_res.get(cksum_val, None) is not None:
                        self_const_correctness = selfconst_res[cksum_val]['genetic_w_selfconst']
                        self_const_num_mut = selfconst_res[cksum_val]['num_demo_used']
                        orig_self_const_score = selfconst_ans[cksum_val]['original_consistency_score']
                        final_self_const_score = selfconst_ans[cksum_val]['final_consistency_score']

                        if self_const_num_mut>0:
                            num_self_const_correct_n_mut.append(self_const_correctness)
                            orig_self_const_score_n_mut.append(orig_self_const_score)
                            final_self_const_score_n_mut.append(final_self_const_score)
                        else:
                            num_self_const_correct_n_no_mut.append(self_const_correctness)
                            orig_self_const_score_n_no_mut.append(orig_self_const_score)
                            final_self_const_score_n_no_mut.append(final_self_const_score)
                        # end if
                    # end if
                # end if
            # end for

            mod_n_mut_correctness_stats = cls.get_stats(num_mod_const_correct_n_mut) if any(num_mod_const_correct_n_mut) else None
            mod_n_no_mut_correctness_stats = cls.get_stats(num_mod_const_correct_n_no_mut) if any(num_mod_const_correct_n_no_mut) else None
            self_n_mut_correctness_stats = cls.get_stats(num_self_const_correct_n_mut) if any(num_self_const_correct_n_mut) else None
            self_n_no_mut_correctness_stats = cls.get_stats(num_self_const_correct_n_no_mut) if any(num_self_const_correct_n_no_mut) else None
            
            mod_n_mut_orig_score_stats = cls.get_stats(orig_mod_const_score_n_mut) if any(orig_mod_const_score_n_mut) else None
            mod_n_no_mut_orig_score_stats = cls.get_stats(orig_mod_const_score_n_no_mut) if any(orig_mod_const_score_n_no_mut) else None
            self_n_mut_orig_score_stats = cls.get_stats(orig_self_const_score_n_mut) if any(orig_self_const_score_n_mut) else None
            self_n_no_mut_orig_score_stats = cls.get_stats(orig_self_const_score_n_no_mut) if any(orig_self_const_score_n_no_mut) else None

            mod_n_mut_final_score_stats = cls.get_stats(final_mod_const_score_n_mut) if any(final_mod_const_score_n_mut) else None
            # mod_n_no_mut_final_score_stats = cls.get_stats(num_mod_const_correct_n_no_mut) if any(num_mod_const_correct_n_no_mut) else None
            self_n_mut_final_score_stats = cls.get_stats(final_self_const_score_n_mut) if any(final_self_const_score_n_mut) else None
            # self_n_no_mut_final_score_stats = cls.get_stats(num_self_const_correct_n_no_mut) if any(num_self_const_correct_n_no_mut) else None


            print(f"{dataset_name}::{len(eval_res_files)}")
            print(f"mod_n_mut_correctness_stats:")
            print(mod_n_mut_correctness_stats)
            print(f"mod_n_no_mut_correctness_stats:")
            print(mod_n_no_mut_correctness_stats)
            print('-----')
            print(f"self_n_mut_correctness_stats:")
            print(self_n_mut_correctness_stats)
            print(f"self_n_no_mut_correctness_stats:")
            print(self_n_no_mut_correctness_stats)
            print('-----')
            print(f"mod_n_mut_orig_score_stats:")
            print(mod_n_mut_orig_score_stats)
            print(f"mod_n_mut_final_score_stats:")
            print(mod_n_mut_final_score_stats)
            print(f"mod_n_no_mut_orig_score_stats:")
            print(mod_n_no_mut_orig_score_stats)
            print(f"self_n_mut_orig_score_stats:")
            print(self_n_mut_orig_score_stats)
            print(f"self_n_mut_final_score_stats:")
            print(self_n_mut_final_score_stats)
            print(f"self_n_no_mut_orig_score_stats:")
            print(self_n_no_mut_orig_score_stats)
            print('==========\n')
        # end if
        return

    @classmethod
    def get_mutation_consistency(
        cls,
        dataset_name: str,
        llm_name: str
    ) -> None:
        random.seed(Macros.RAND_SEED)

        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        # eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_res_dir = Macros.result_dir / 'genetic'/ dataset_name / 'evaluate_consistency' / llm_name
        # genetic_dir.mkdir(parents=True, exist_ok=True)
        eval_res_files = sorted([
            f_name for f_name in os.listdir(str(eval_dir))
            if f_name.endswith('.json') and \
                f_name.startswith('eval-') and \
                (not f_name.startswith('eval-results-w'))
        ])
        
        genetic_res_file = genetic_res_dir / 'final_answers.json'
        selfconst_res_file = genetic_res_dir / 'final_answers_w_self_consistency_over_cot_response.json'
        genetic_res = Utils.read_json(genetic_res_file)
        selfconst_res = Utils.read_json(selfconst_res_file)
        scores_over_all_data = list()

        for eval_res_file in eval_res_files:
            eval_res = Utils.read_json(eval_dir / eval_res_file)
            cksum_val = eval_res_file.split('-')[-1].split('.json')[0].strip()
            print(cksum_val)
            scores = list()
            mutations = eval_res['mutation']
            for m_i, mut in enumerate(mutations):
                if eval_res['mutation'][m_i].get('ans_dict', None) is None:
                    score, mut_ans_dict = cls.get_mod_consistency(mut, dataset_name)
                    scores.append(score)
                    scores_over_all_data.append(score)
                    eval_res['mutation'][m_i]['ans_dict'] = mut_ans_dict
                else:
                    c = Counter(eval_res['mutation'][m_i]['ans_dict'].values())
                    value_freqs = c.most_common()
                    if len(set([v[1] for v in value_freqs]))==1:
                        # in case of having no most frequent answer
                        scores.append(1)
                        scores_over_all_data.append(1)
                    else:
                        scores.append(value_freqs[0][1])
                        scores_over_all_data.append(value_freqs[0][1])
                    # end if
                # end if
            # end for
            # const_score_dict[cksum_val] = scores
            Utils.write_json(
                eval_res,
                eval_dir / eval_res_file,
                pretty_format=True
            )
        # end for
        c = Counter(scores_over_all_data)
        value_freqs = c.most_common()
        print(f"{dataset_name}::{llm_name}::{value_freqs}")
        return 


class AnalyzeCSVgenForDataAnalysis:

    @classmethod
    def generate_csv_gsm8k(
        cls,
        llm_name: str
    ) -> None:
        import csv
        eval_dir = Macros.result_dir / 'nl2nl'/ 'gsm8k' / 'evaluate_consistency' / llm_name
        # eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_res_dir = Macros.result_dir / 'genetic'/ 'gsm8k' / 'evaluate_consistency' / llm_name
        # genetic_dir.mkdir(parents=True, exist_ok=True)
        
        genetic_res_file = genetic_res_dir / 'final_answers.json'
        genetic_res = Utils.read_json(genetic_res_file)

        csv_file = genetic_res_dir / 'gsm8k_for_google_sheets_data_analysis.csv'

        cksum_vals_of_interest_gsm8k =[
            '0041146', '00e26af', '0402595', '0584ce5', '069059b', '069d3bb', '0787191', '0bb4aec', '0f28b5d', '0fcbc61',
            '10a5ab2', '1700002', '17326d1', '182be0c', '1aa48fc', '1b0114c', '208e43f', '20aee3a', '2387337', '2421fcb',
            '25ddc0f', '26408ff', '26588e9', '27ed0fb', '2812e5c', '2823f47', '28f0b86', '2afe456', '2bb232c', '2de5d16',
            '2df4524', '3210ddb', '33e8075', '3473dec', '3505107', '37bc2f7', '3806734', '3fe94a0', '4079016', '41f1f19',
            '428fca9', '47d1e99', '4b0a59d', '500e75a', '52720e0', '550a141', '5737c6e', '5751ec3', '58238e9', '5c04925',
            '5c572ec', '5e38810', '621461a', '632cee9', '65b9eea', '66f041e', '6766aa2', '67e103b', '6c3cf77', '6e7d2da',
            '708f3cf', '76dc611', '7c590f0', '7e7757b', '7eb3c8b', '7f6ffaa', '7fe1f8a', '818f465', '81e74d6', '847cc55',
            '85d8ce5', '86b122d', '884d247', '8c7bbbb', '8cb22bd', '8e2cfdc', '8e82ab7', '8e98d81', '903ce92', '96da2f5', 
            '97af4fb', '97e8527', '98b2979', '9908279', '9a11581', '9ab0d88', 'a666587', 'a67f096', 'a8e864d', 'a8ecbab',
            'a9078e8', 'ab1a4d0', 'ab817c9', 'ac627ab', 'ad61ab1', 'afd4836', 'b137fdd', 'b2eeb73', 'b3967a0', 'b55ec28',
            'bac9162', 'bbcbff5', 'bf82296', 'c5ab0bc', 'c60d060', 'caf1a3d', 'cc1aa43', 'cd89fef', 'd1c38a0', 'd2ed45a',
            'd6baf65', 'd6c651d', 'd6ef5f7', 'd82c8d1', 'd96409b', 'da11e8c', 'dc87c13', 'df263d9', 'e004061', 'e00da03',
            'e0ec453', 'e223121', 'e325107', 'e369853', 'e44fea3', 'e46de7e', 'e995f98', 'eaae339', 'eeb69a3', 'eefc9e1',
            'ef4e3b7', '15d4e89', '8a0e114', 'd64a340', '076a0c9'
        ]

        column_headers = [
            'original_question', 'original_anwer_gt', 'llm_responses', 'llm_answer4originalquestion(cot,code,eqn)',
            'mutated_question', 'llm_mutation_responses', 'llm_mutated_anwer'
        ]

        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(column_headers)

            for cksum_val in cksum_vals_of_interest_gsm8k:

                orig_q = genetic_res[cksum_val]["final_response"]["question"]
                orig_ans_gt = genetic_res[cksum_val]["final_response"]["answer"]
                llm_responses = f"\"cot_response\": {genetic_res[cksum_val]['final_response']['cot_response']}\n" + \
                    f"\"code_response\": {genetic_res[cksum_val]['final_response']['code_response']}\n" + \
                    f"\"eqn_response\": {genetic_res[cksum_val]['final_response']['eqn_response']}"
                llm_ans = f"{genetic_res[cksum_val]['final_answer']['cot_response']},{genetic_res[cksum_val]['final_answer']['code_response']},{genetic_res[cksum_val]['final_answer']['eqn_response']}"

                if len(genetic_res[cksum_val]["mutation"])<2:
                    mut_q = genetic_res[cksum_val]["mutation"][0]['mut']['question']
                    llm_mut_responses = f"\"cot_response\": {genetic_res[cksum_val]['mutation'][0]['mut']['cot_response']}\n" + \
                        f"\"code_response\": {genetic_res[cksum_val]['mutation'][0]['mut']['code_response']}\n" + \
                        f"\"eqn_response\": {genetic_res[cksum_val]['mutation'][0]['mut']['eqn_response']}"
                    llm_mut_anwer = f"{genetic_res[cksum_val]['mutation'][0]['mut']['ans_dict']['cot_response']}"
                    writer.writerow([
                        orig_q, orig_ans_gt, llm_responses, llm_ans, mut_q, llm_mut_responses, llm_mut_anwer
                    ])
                else:
                    mut_q = ""
                    llm_mut_responses = ""
                    llm_mut_anwer = ""
                    for mut_dict in genetic_res[cksum_val]["mutation"]:
                        mut_q += f"{mut_dict['mut']['question']}<eos>\n"
                        llm_mut_responses += f"\"cot_response\": {mut_dict['mut']['cot_response']}\n" + \
                            f"\"code_response\": {mut_dict['mut']['code_response']}\n" + \
                            f"\"eqn_response\": {mut_dict['mut']['eqn_response']}<eos>\n"
                        llm_mut_anwer += f"{mut_dict['mut']['ans_dict']['cot_response']}<eos>\n"
                    # end for
                    writer.writerow([
                        orig_q, orig_ans_gt, llm_responses, llm_ans, mut_q, llm_mut_responses, llm_mut_anwer
                    ])
                # end if
            # end for
        # end with
        return



class AnalyzeMutationCorrectness:

    @classmethod
    def print_cksum_vals(
        cls,
        llm_name: str,
        dataset_name: str
    ) -> None:
        import csv
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        # eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_res_dir = Macros.result_dir / 'genetic_fg'/ dataset_name / 'evaluate_consistency' / llm_name
        # genetic_dir.mkdir(parents=True, exist_ok=True)
        
        genetic_res_file = genetic_res_dir / 'acc-res-genetic-fg.json'
        genetic_res = Utils.read_json(genetic_res_file)
        print(genetic_res_file)

        for cksum_val in genetic_res.keys():
            if genetic_res[cksum_val]['genetic']==0. and \
                genetic_res[cksum_val]['num_demo_used']>0.:
                num_muts = genetic_res[cksum_val]['num_demo_used']
                print(cksum_val)
            # end if
        # end for
        return