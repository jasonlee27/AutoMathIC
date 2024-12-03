
import os
import re
import math
import numpy as np
import wolframalpha

from typing import *
from pathlib import Path
from collections import Counter

from ..genetic.multimodals import Multimodals
from ..llmut.evaluate import EvaluateWithMultimodals
from ..utils.macros import Macros
from ..utils.utils import Utils

wolframalpha_api_key = os.environ["WOLFRAMALPHA_API_KEY"]
wa_client = wolframalpha.Client(wolframalpha_api_key)


class Accuracy:

    @classmethod
    def tokenize(cls, sent: str) -> List[str]:
        return Utils.tokenize(sent)

    @classmethod
    def convert_string_number_to_number(cls, str_num: Any) -> Any:
        # check if the number contains comma e.g.) 1,456=1456
        _str_num = str_num.replace(',', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(_str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if
        
        # check if the token is float number in string format
        _str_num = str_num.replace('.', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        # check if the token is negative number in string format
        _str_num = str_num.replace('-', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        if str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if
        return str_num

    @classmethod
    def get_num_after_decimal_points(cls, num: Any):
        if type(num)==int:
            return 0
        # end if
        rem_str = str(num).split('.')[-1]
        return len(rem_str)

    @classmethod
    def are_numbers_same(cls, query_num: Any, ref_num: Any) -> bool:
        if type(query_num)==int or \
            type(ref_num)==int:
            return query_num==ref_num
        # end if
        num1_rem_len = cls.get_num_after_decimal_points(query_num)
        num2_rem_len = cls.get_num_after_decimal_points(ref_num)
        tgt_rem_len = min(num1_rem_len, num2_rem_len)
        
        _query_num = math.floor(query_num*(10**tgt_rem_len))/(10**tgt_rem_len)
        _ref_num = math.floor(ref_num*(10**tgt_rem_len))/(10**tgt_rem_len)
        if _query_num==_ref_num:
            return True
        # end if
        return False

    @classmethod
    def get_correctness(
        cls, 
        query_sent: str, 
        ref_sent: Any
    ) -> float:
        query_tokens = cls.tokenize(query_sent)
        query_tokens = [
            cls.convert_string_number_to_number(t)
            for t in query_tokens
        ]
        correctness = 0.
        ref_val = ref_sent
        if type(ref_sent)==str:
            ref_val = eval(ref_sent)
        # end if
        for qt in query_tokens:
            if type(qt)!=str:
                if cls.are_numbers_same(qt, ref_val):
                    correctness = 1.
                    break
                # end if
            # end if
        # end for

        # if type(ref_sent)==str:
        #     ref_val = eval(ref_sent)
        #     if (int(ref_val) in query_tokens) or\
        #         (float(ref_val) in query_tokens):
        #         correctness = 1.
        #     # end if
        # else:
        #     if (int(ref_sent) in query_tokens) or\
        #         (float(ref_sent) in query_tokens):
        #         correctness = 1.
        #     # end if
        # # end if
        return correctness

    @classmethod
    def get_correctness_per_file(
        cls, 
        result_dir: Path, 
        file_name: str
    ):
        correctness_mut_list = list()
        correctness_tot_list = list()
        correctness_mut_w_neg_answer_list = list()
        correctness_mut_w_non_neg_answer_list = list()
        mut_w_neg_answer_list = list()
        pass_to_fail = 0.
        fail_to_pass = 0.

        resp = Utils.read_json(result_dir / file_name)
        # Utils.write_json(
        #     resp, 
        #     result_dir / file_name,
        #     pretty_format=True
        # )
        # print(file_name)
        correctness_orig = cls.get_correctness(
            resp['orig']['response'], 
            resp['orig']['answer']
        )
        if correctness_orig==0.:
            pass_to_fail = None
        else:
            fail_to_pass = None
        # end if
        correctness_tot_list.append(correctness_orig)

        for mut in resp['mutation']:
            correctness_mut = cls.get_correctness(
                mut['response'], 
                mut['answer']
            )
            correctness_mut_list.append(correctness_mut)
            correctness_tot_list.append(correctness_mut)
            if correctness_orig==0. and \
                correctness_mut==1.:
                fail_to_pass += 1.
            # end if
            if correctness_orig==1. and \
                correctness_mut==0.:
                pass_to_fail += 1.
            # end if

            if eval(mut['answer'])<0.:
                mut_w_neg_answer_list.append('neg_ans')
                correctness_mut_w_neg_answer_list.append(correctness_mut)
            else:
                mut_w_neg_answer_list.append('')
                correctness_mut_w_non_neg_answer_list.append(correctness_mut)
            # end if
        # end for
        return correctness_orig, \
                correctness_mut_list, \
                correctness_tot_list, \
                pass_to_fail, \
                fail_to_pass, \
                correctness_mut_w_neg_answer_list, \
                correctness_mut_w_non_neg_answer_list, \
                mut_w_neg_answer_list

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
    def classify_results(
        cls, 
        correctness_orig: float,
        correctness_mut: List[float],
    ) -> str:
        if correctness_orig==1.:
            if sum(correctness_mut)==len(correctness_mut):
                return 'orig:pass&mut:pass'
            else:
                return 'orig:pass&mut:fail'
            # end if
        else:
            if sum(correctness_mut)==len(correctness_mut):
                return 'orig:fail&mut:pass'
            else:
                return 'orig:fail&mut:fail'
            # end if
        # end if
        return

    @classmethod
    def evaluate(cls,
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python'
    ) -> None:
        res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        eval_dir = res_dir / 'evaluate' / model_name
        llm_response_files = [
            f for f in os.listdir(str(eval_dir))
            if f.endswith('.json') and f.startswith('eval-')
        ]
        acc_result = dict()
        correctness_orig_list = list()
        correctness_mut_list = list()
        correctness_tot_list = list()
        mut_w_neg_answer_list = list()
        mut_w_neg_answer_acc_list = list()
        mut_w_non_neg_answer_acc_list = list()
        model_performance_type_w_neg_answers = {
            'orig:pass&mut:pass': {
                'num-instances': 0,
                'num-orig-pass': 0,
                'acc': 0.,
                'num-mutation': list(),
                'cksum-val': list()
            },
            'orig:pass&mut:fail': {
                'num-instances': 0,
                'num-orig-pass': 0,
                'acc': 0.,
                'num-mutation': list(),
                'cksum-val': list()
            },
            'orig:fail&mut:pass': {
                'num-instances': 0,
                'num-orig-fail': 0,
                'acc': 0.,
                'num-mutation': list(),
                'cksum-val': list()
            },
            'orig:fail&mut:fail': {
                'num-instances': 0,
                'num-orig-fail': 0,
                'acc': 0.,
                'num-mutation': list(),
                'cksum-val': list()
            }
        }
        model_performance_type_w_non_neg_answers = {
            'orig:pass&mut:pass': {
                'num-instances': 0,
                'num-orig-pass': 0,
                'acc': 0.,
                'num-mutation': list(),
                'cksum-val': list()
            },
            'orig:pass&mut:fail': {
                'num-instances': 0,
                'num-orig-pass': 0,
                'acc': 0.,
                'num-mutation': list(),
                'cksum-val': list()
            },
            'orig:fail&mut:pass': {
                'num-instances': 0,
                'num-orig-fail': 0,
                'acc': 0.,
                'num-mutation': list(),
                'cksum-val': list()
            },
            'orig:fail&mut:fail': {
                'num-instances': 0,
                'num-orig-fail': 0,
                'acc': 0.,
                'num-mutation': list(),
                'cksum-val': list()
            }
        }
        p2f_list = list()
        f2p_list = list()

        for resp_file in llm_response_files:
            cksum_val = re.search(r'eval\-(.*)\.json', resp_file).group(1)
            acc_result[cksum_val] = dict()

            correctness_orig, \
            correctness_mut, \
            correctness_tot, \
            pass_to_fail, \
            fail_to_pass, \
            correctness_mut_w_neg_answer, \
            correctness_mut_w_non_neg_answer, \
            mut_w_neg_answer = cls.get_correctness_per_file(
                eval_dir, 
                resp_file
            )

            performance_result_type_w_neg_answers = cls.classify_results(
                correctness_orig,
                correctness_mut_w_neg_answer
            )

            performance_result_type_w_non_neg_answers = cls.classify_results(
                correctness_orig,
                correctness_mut_w_non_neg_answer
            )

            model_performance_type_w_neg_answers[performance_result_type_w_neg_answers]['cksum-val'].append(cksum_val)
            model_performance_type_w_neg_answers[performance_result_type_w_neg_answers]['num-instances'] += 1
            model_performance_type_w_neg_answers[performance_result_type_w_neg_answers]['num-mutation'].append(len(correctness_mut_w_neg_answer))

            model_performance_type_w_non_neg_answers[performance_result_type_w_non_neg_answers]['cksum-val'].append(cksum_val)
            model_performance_type_w_non_neg_answers[performance_result_type_w_non_neg_answers]['num-instances'] += 1
            model_performance_type_w_non_neg_answers[performance_result_type_w_non_neg_answers]['num-mutation'].append(len(correctness_mut_w_non_neg_answer))

            acc_result[cksum_val]['orig'] = correctness_orig
            acc_result[cksum_val]['mutation'] = [
                correctness_mut[m_i] if mut_w_neg_answer[m_i]=='' else f"{correctness_mut[m_i]}::{mut_w_neg_answer[m_i]}"
                for m_i, _ in enumerate(correctness_mut)
            ]
            correctness_orig_list.append(correctness_orig)
            correctness_mut_list.extend(correctness_mut)
            correctness_tot_list.extend(correctness_tot)
            mut_w_neg_answer_list.extend([
                0. if m=='' else 1.
                for m in mut_w_neg_answer
            ])
            mut_w_neg_answer_acc_list.extend([
                correctness_mut[m_i]
                for m_i, m in enumerate(mut_w_neg_answer) if m!=''
            ])
            mut_w_non_neg_answer_acc_list.extend([
                correctness_mut[m_i]
                for m_i, m in enumerate(mut_w_neg_answer) if m==''
            ])

            if pass_to_fail is not None:
                p2f_list.append(pass_to_fail)
            # end if

            if fail_to_pass is not None:
                f2p_list.append(fail_to_pass)
            # end if
        # end for

        for key in model_performance_type_w_non_neg_answers.keys():
            if key.startswith('orig:pass&'):
                num_pass = len([
                    c for c in correctness_orig_list if c==1.
                ])
                model_performance_type_w_non_neg_answers[key]['num-orig-pass'] = num_pass
                model_performance_type_w_non_neg_answers[key]['acc'] = \
                model_performance_type_w_non_neg_answers[key]['num-instances']*1./num_pass
                model_performance_type_w_non_neg_answers[key]['num-mutation'] = \
                cls.get_stats(model_performance_type_w_non_neg_answers[key]['num-mutation']) \
                if any(model_performance_type_w_non_neg_answers[key]['num-mutation']) else 0
                
            else:
                num_fail = len([
                    c for c in correctness_orig_list if c==0.
                ])
                model_performance_type_w_non_neg_answers[key]['num-orig-fail'] = num_fail
                model_performance_type_w_non_neg_answers[key]['acc'] = \
                model_performance_type_w_non_neg_answers[key]['num-instances']*1./num_fail
                model_performance_type_w_non_neg_answers[key]['num-mutation'] = \
                cls.get_stats(model_performance_type_w_non_neg_answers[key]['num-mutation'])
            # end if
        # end for
        
        for key in model_performance_type_w_neg_answers.keys():
            if key.startswith('orig:pass&'):
                num_pass = len([
                    c for c in correctness_orig_list if c==1.
                ])
                model_performance_type_w_neg_answers[key]['num-orig-pass'] = num_pass
                model_performance_type_w_neg_answers[key]['acc'] = \
                model_performance_type_w_neg_answers[key]['num-instances']*1./num_pass
                model_performance_type_w_neg_answers[key]['num-mutation'] = \
                cls.get_stats(model_performance_type_w_neg_answers[key]['num-mutation']) \
                if any(model_performance_type_w_neg_answers[key]['num-mutation']) else 0
            else:
                num_fail = len([
                    c for c in correctness_orig_list if c==0.
                ])
                model_performance_type_w_neg_answers[key]['num-orig-fail'] = num_fail
                model_performance_type_w_neg_answers[key]['acc'] = \
                model_performance_type_w_neg_answers[key]['num-instances']*1./num_fail
                model_performance_type_w_neg_answers[key]['num-mutation'] = \
                cls.get_stats(model_performance_type_w_neg_answers[key]['num-mutation'])
            # end if
        # end for

        acc_stats = {
            'acc-orig': cls.get_stats(correctness_orig_list),
            'acc-mutation': cls.get_stats(correctness_mut_list),
            'acc-total': cls.get_stats(correctness_tot_list),
            'pass-to-fail': cls.get_stats(p2f_list),
            'fail-to-pass': cls.get_stats(f2p_list),
            'num-q-w-neg-answer': cls.get_stats(mut_w_neg_answer_list),
            'acc-q-w-neg-answer': cls.get_stats(mut_w_neg_answer_acc_list),
            'acc-q-w-non-neg-answer': cls.get_stats(mut_w_non_neg_answer_acc_list),
            # 'performance-per-type-w-neg-answers': model_performance_type_w_neg_answers
            'performance-per-type-w-non-neg-answers': model_performance_type_w_non_neg_answers
        }

        Utils.write_json(
            acc_result, 
            eval_dir / f"acc-results.json",
            pretty_format=True
        )
        Utils.write_json(
            acc_stats, 
            eval_dir / f"acc-stats.json",
            pretty_format=True
        )
        return


class AccuracyForDemo:

    emb_consist_type_n_res_file_map = {
        'random': 'eval-results-w-demo-random.json',
        'modcons-random': 'eval-results-w-demo-modcons-random.json',
        'cos_sim': 'eval-results-w-demo-simwtgtnmut.json',
        'modcons-cos_sim': 'eval-results-w-demo-modcons-simwtgtnmut.json',
        'dist': 'eval-results-w-demo-distwtgtnmut.json',
        'modcons-dist': 'eval-results-w-demo-modcons-distwtgtnmut.json',
        'avg_dist_among_muts': 'eval-results-w-demo-avgdistsmongmuts.json',
        'modcons-avg_dist_among_muts': 'eval-results-w-demo-modcons-avgdistsmongmuts.json'
    }

    @classmethod
    def tokenize(cls, sent: str) -> List[str]:
        return Utils.tokenize(sent)

    @classmethod
    def convert_string_number_to_number(cls, str_num: Any) -> Any:
        # check if the number contains comma e.g.) 1,456=1456
        _str_num = str_num.replace(',', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(_str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        # check if the token is float number in string format
        _str_num = str_num.replace('.', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        # check if the token is negative number in string format
        _str_num = str_num.replace('-', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        if str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if
        return str_num

    @classmethod
    def get_num_after_decimal_points(cls, num: Any):
        if type(num)==int:
            return 0
        # end if
        rem_str = str(num).split('.')[-1]
        return len(rem_str)

    @classmethod
    def are_numbers_same(cls, query_num: Any, ref_num: Any) -> bool:
        if type(query_num)==int or \
            type(ref_num)==int:
            return query_num==ref_num
        # end if
        num1_rem_len = cls.get_num_after_decimal_points(query_num)
        num2_rem_len = cls.get_num_after_decimal_points(ref_num)
        tgt_rem_len = min(num1_rem_len, num2_rem_len)
        
        _query_num = math.floor(query_num*(10**tgt_rem_len))/(10**tgt_rem_len)
        _ref_num = math.floor(ref_num*(10**tgt_rem_len))/(10**tgt_rem_len)
        if _query_num==_ref_num:
            return True
        # end if
        return False

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        return Utils.get_answer_from_ground_truth(str_answer)

    @classmethod
    def get_answers_from_cot_resp(cls, query_sent: str) -> str:
        return Utils.get_answer_from_cot_resp(query_sent)

    @classmethod
    def get_correctness(
        cls, 
        query_sent: str, 
        ref_sent: Any
    ) -> float:
        query_answer = cls.get_answers_from_cot_resp(query_sent)
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
    def evaluate(cls,
        model_name: str, 
        dataset_name: str,
        emb_consist_type: str,
        pl_type: str='python'
    ) -> None:
        # assert emb_consist_type in [
        #     'cos_sim', 'dist', 'avg_dist_among_muts'
        # ]
        # if dataset_name=='svamp':
        #     res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        # else:
        #     res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        # # end if

        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        
        eval_rand_demo_res_file = eval_dir / cls.emb_consist_type_n_res_file_map['random']
        eval_modcons_rand_demo_res_file = eval_dir / cls.emb_consist_type_n_res_file_map['modcons-random']
        
        eval_rand_demo_res = Utils.read_json(eval_rand_demo_res_file)
        eval_modcons_rand_demo_res = Utils.read_json(eval_modcons_rand_demo_res_file)

        eval_modcons_demo_res = None
        if emb_consist_type is not None:
            eval_demo_res_file = eval_dir / cls.emb_consist_type_n_res_file_map[emb_consist_type]
            eval_modcons_demo_res_file = eval_dir / cls.emb_consist_type_n_res_file_map[f"modcons-{emb_consist_type}"]
            eval_demo_res = Utils.read_json(eval_demo_res_file)
            eval_modcons_demo_res = Utils.read_json(eval_modcons_demo_res_file)
        # end if

        acc_res = dict()
        acc_stats = dict()
        acc_res_modcons = dict()
        acc_stats_modcons = dict()
        acc_res_rand_demo = dict()
        acc_stats_rand_demo = dict()
        acc_res_modcons_rand_demo = dict()
        acc_stats_modcons_rand_demo = dict()

        correctness_orig_list = list()
        correctness_demo_list = list()
        correctness_modcons_demo_list = list()
        correctness_rand_demo_list = list()
        correctness_modcons_rand_demo_list = list()
        # num_demo_list = list()
        for cksum_val in eval_rand_demo_res.keys():
            print(cksum_val)
            res_orig: Dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
            if eval_modcons_demo_res is not None:
                res_modcons_demo: Dict = eval_modcons_demo_res[cksum_val]

                correctness_orig = cls.get_correctness(
                    res_orig['orig']['cot_response'], 
                    res_orig['orig']['answer']
                )
                correctness_modcons_demo = cls.get_correctness(
                    res_modcons_demo['response'], 
                    res_modcons_demo['answer']
                )
                
                if res_modcons_demo['num_demo_used']>0:

                    res_demo: Dict = eval_demo_res[cksum_val]
                    res_rand_demo: Dict = eval_rand_demo_res[cksum_val]
                    res_modcons_rand_demo: Dict = eval_modcons_rand_demo_res[cksum_val]

                    correctness_demo = cls.get_correctness(
                        res_demo['response'], 
                        res_demo['answer']
                    )
                    correctness_rand_demo = cls.get_correctness(
                        res_rand_demo['response'], 
                        res_rand_demo['answer']
                    )
                    correctness_modcons_rand_demo = cls.get_correctness(
                        res_modcons_rand_demo['response'], 
                        res_modcons_rand_demo['answer']
                    )

                    acc_res_modcons[cksum_val] = {
                        'orig': correctness_orig,
                        'demo': correctness_modcons_demo,
                        'num_demo_used': res_modcons_demo['num_demo_used']
                    }
                    acc_res[cksum_val] = {
                        'orig': correctness_orig,
                        'demo': correctness_demo,
                        'num_demo_used': res_demo['num_demo_used']
                    }
                    acc_res_rand_demo[cksum_val] = {
                        'orig': correctness_orig,
                        'demo': correctness_rand_demo,
                        'num_demo_used': res_rand_demo['num_demo_used']
                    }
                    acc_res_modcons_rand_demo[cksum_val] = {
                        'orig': correctness_orig,
                        'demo': correctness_modcons_rand_demo,
                        'num_demo_used': res_modcons_rand_demo['num_demo_used']
                    }

                    correctness_orig_list.append(correctness_orig)
                    correctness_modcons_demo_list.append(correctness_modcons_demo)
                    correctness_demo_list.append(correctness_demo)
                    correctness_rand_demo_list.append(correctness_rand_demo)
                    correctness_modcons_rand_demo_list.append(correctness_modcons_rand_demo)
                # end if
                # num_demo_list.append(res_demo['num_demo_used'])
            else:
                correctness_orig = cls.get_correctness(
                    res_orig['orig']['cot_response'], 
                    res_orig['orig']['answer']
                )

                res_rand_demo: Dict = eval_rand_demo_res[cksum_val]
                res_modcons_rand_demo: Dict = eval_modcons_rand_demo_res[cksum_val]

                correctness_rand_demo = cls.get_correctness(
                    res_rand_demo['response'], 
                    res_rand_demo['answer']
                )
                correctness_modcons_rand_demo = cls.get_correctness(
                    res_modcons_rand_demo['response'], 
                    res_modcons_rand_demo['answer']
                )

                acc_res_rand_demo[cksum_val] = {
                    'orig': correctness_orig,
                    'demo': correctness_rand_demo,
                    'num_demo_used': res_rand_demo['num_demo_used']
                }
                acc_res_modcons_rand_demo[cksum_val] = {
                    'orig': correctness_orig,
                    'demo': correctness_modcons_rand_demo,
                    'num_demo_used': res_modcons_rand_demo['num_demo_used']
                }

                correctness_orig_list.append(correctness_orig)
                correctness_rand_demo_list.append(correctness_rand_demo)
                correctness_modcons_rand_demo_list.append(correctness_modcons_rand_demo)
                # num_demo_list.append(res_demo['num_demo_used'])
            # end if
        # end for
        if emb_consist_type is not None:
            acc_stats = {
                'acc-orig': cls.get_stats(correctness_orig_list),
                'acc-demo': cls.get_stats(correctness_demo_list)
            }
            acc_stats_modcons = {
                'acc-orig': cls.get_stats(correctness_orig_list),
                'acc-demo': cls.get_stats(correctness_modcons_demo_list)
            }
        # end if
        acc_stats_rand_demo = {
            'acc-orig': cls.get_stats(correctness_orig_list),
            'acc-demo': cls.get_stats(correctness_rand_demo_list)
        }
        acc_stats_modcons_rand_demo = {
            'acc-orig': cls.get_stats(correctness_orig_list),
            'acc-demo': cls.get_stats(correctness_modcons_rand_demo_list)
        }
        
        if emb_consist_type is not None:
            Utils.write_json(
                acc_res,
                eval_dir / f"acc-res-{emb_consist_type}.json",
                pretty_format=True
            )

            Utils.write_json(
                acc_stats, 
                eval_dir / f"acc-stats-{emb_consist_type}.json",
                pretty_format=True
            )

            Utils.write_json(
                acc_res_modcons,
                eval_dir / f"acc-res-modcons-{emb_consist_type}.json",
                pretty_format=True
            )

            Utils.write_json(
                acc_stats_modcons, 
                eval_dir / f"acc-stats-modcons-{emb_consist_type}.json",
                pretty_format=True
            )
        # end if

        Utils.write_json(
            acc_res_rand_demo, 
            eval_dir / f"acc-res-random.json",
            pretty_format=True
        )

        Utils.write_json(
            acc_stats_rand_demo, 
            eval_dir / f"acc-stats-random.json",
            pretty_format=True
        )

        Utils.write_json(
            acc_res_modcons_rand_demo, 
            eval_dir / f"acc-res-modcons-random.json",
            pretty_format=True
        )

        Utils.write_json(
            acc_stats_modcons_rand_demo, 
            eval_dir / f"acc-stats-modcons-random.json",
            pretty_format=True
        )
        return


class AccuracyForDemoFromAlgConsistency:

    # emb_consist_type_n_res_file_map = {
    #     'random': 'eval-results-w-demo-random.json',
    #     'modcons-random': 'eval-results-w-demo-modcons-random.json',
    #     'cos_sim': 'eval-results-w-demo-simwtgtnmut.json',
    #     'modcons-cos_sim': 'eval-results-w-demo-modcons-simwtgtnmut.json',
    #     'dist': 'eval-results-w-demo-distwtgtnmut.json',
    #     'modcons-dist': 'eval-results-w-demo-modcons-distwtgtnmut.json',
    #     'avg_dist_among_muts': 'eval-results-w-demo-avgdistsmongmuts.json',
    #     'modcons-avg_dist_among_muts': 'eval-results-w-demo-modcons-avgdistsmongmuts.json'
    # }

    @classmethod
    def tokenize(cls, sent: str) -> List[str]:
        return Utils.tokenize(sent)

    @classmethod
    def convert_string_number_to_number(cls, str_num: Any) -> Any:
        # check if the number contains comma e.g.) 1,456=1456
        _str_num = str_num.replace(',', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(_str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        # check if the token is float number in string format
        _str_num = str_num.replace('.', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        # check if the token is negative number in string format
        _str_num = str_num.replace('-', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        if str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if
        return str_num

    @classmethod
    def get_num_after_decimal_points(cls, num: Any):
        if type(num)==int:
            return 0
        # end if
        rem_str = str(num).split('.')[-1]
        return len(rem_str)

    @classmethod
    def are_numbers_same(cls, query_num: Any, ref_num: Any) -> bool:
        if type(query_num)==int or \
            type(ref_num)==int:
            return query_num==ref_num
        # end if
        num1_rem_len = cls.get_num_after_decimal_points(query_num)
        num2_rem_len = cls.get_num_after_decimal_points(ref_num)
        tgt_rem_len = min(num1_rem_len, num2_rem_len)
        
        _query_num = math.floor(query_num*(10**tgt_rem_len))/(10**tgt_rem_len)
        _ref_num = math.floor(ref_num*(10**tgt_rem_len))/(10**tgt_rem_len)
        if _query_num==_ref_num:
            return True
        # end if
        return False

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        return Utils.get_answer_from_ground_truth(str_answer)

    @classmethod
    def get_correctness(
        cls, 
        query_answer: str, 
        ref_sent: Any
    ) -> float:
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
    def evaluate(cls,
        model_name: str, 
        dataset_name: str,
        pl_type: str='python',
        include_only_modconst=False
    ) -> None:
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        alg_const_dir = eval_dir / 'alg_consistency'
        if include_only_modconst:
            alg_const_dir = eval_dir / 'alg_consistency_only_modconst'
        # end if

        eval_demo_res_file = alg_const_dir / 'eval-results-w-demo-alg-consistency.json'
        print(eval_demo_res_file)
        print(os.path.exists(str(eval_demo_res_file)))
        eval_demo_res = Utils.read_json(eval_demo_res_file)
        
        acc_res = dict()
        correctness_orig_list = list()
        correctness_demo_list = list()
        
        # num_demo_list = list()
        for cksum_val in eval_demo_res.keys():
            print(cksum_val)
            res_orig: Dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
            res_modcons_demo: Dict = eval_demo_res[cksum_val]
            correctness_orig = cls.get_correctness(
                res_orig['orig']['cot_response'], 
                res_orig['orig']['answer']
            )
            correctness_demo = cls.get_correctness(
                res_modcons_demo['response'], 
                res_modcons_demo['answer']
            )
            
            res_demo: Dict = eval_demo_res[cksum_val]
            correctness_demo = cls.get_correctness(
                res_demo['response'], 
                res_demo['answer']
            )

            acc_res[cksum_val] = {
                'orig': correctness_orig,
                'demo': correctness_demo,
                'num_demo_used': res_demo['num_demo_used']
            }
            correctness_orig_list.append(correctness_orig)
            correctness_demo_list.append(correctness_demo)
        # end for
        
        acc_stats = {
            'acc-orig': cls.get_stats(correctness_orig_list),
            'acc-demo': cls.get_stats(correctness_demo_list)
        }
        
        Utils.write_json(
            acc_res,
            alg_const_dir / f"acc-res.json",
            pretty_format=True
        )

        Utils.write_json(
            acc_stats, 
            alg_const_dir / f"acc-stats.json",
            pretty_format=True
        )
        return


class AccuracyForCodeFromAlgConsistency:

    # emb_consist_type_n_res_file_map = {
    #     'random': 'eval-results-w-demo-random.json',
    #     'modcons-random': 'eval-results-w-demo-modcons-random.json',
    #     'cos_sim': 'eval-results-w-demo-simwtgtnmut.json',
    #     'modcons-cos_sim': 'eval-results-w-demo-modcons-simwtgtnmut.json',
    #     'dist': 'eval-results-w-demo-distwtgtnmut.json',
    #     'modcons-dist': 'eval-results-w-demo-modcons-distwtgtnmut.json',
    #     'avg_dist_among_muts': 'eval-results-w-demo-avgdistsmongmuts.json',
    #     'modcons-avg_dist_among_muts': 'eval-results-w-demo-modcons-avgdistsmongmuts.json'
    # }

    @classmethod
    def tokenize(cls, sent: str) -> List[str]:
        return Utils.tokenize(sent)

    @classmethod
    def convert_string_number_to_number(cls, str_num: Any) -> Any:
        # check if the number contains comma e.g.) 1,456=1456
        _str_num = str_num.replace(',', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(_str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        # check if the token is float number in string format
        _str_num = str_num.replace('.', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        # check if the token is negative number in string format
        _str_num = str_num.replace('-', '')
        if _str_num!=str_num and _str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if

        if str_num.isnumeric():
            try:
                return eval(str_num)
            except SyntaxError:
                return str_num
            # end try
        # end if
        return str_num

    @classmethod
    def get_num_after_decimal_points(cls, num: Any):
        if type(num)==int:
            return 0
        # end if
        rem_str = str(num).split('.')[-1]
        return len(rem_str)

    @classmethod
    def are_numbers_same(cls, query_num: Any, ref_num: Any) -> bool:
        if type(query_num)==int or \
            type(ref_num)==int:
            return query_num==ref_num
        # end if
        num1_rem_len = cls.get_num_after_decimal_points(query_num)
        num2_rem_len = cls.get_num_after_decimal_points(ref_num)
        tgt_rem_len = min(num1_rem_len, num2_rem_len)
        
        _query_num = math.floor(query_num*(10**tgt_rem_len))/(10**tgt_rem_len)
        _ref_num = math.floor(ref_num*(10**tgt_rem_len))/(10**tgt_rem_len)
        if _query_num==_ref_num:
            return True
        # end if
        return False

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        return Utils.get_answer_from_ground_truth(str_answer)

    @classmethod
    def get_correctness_orig(
        cls, 
        query_sent: str, 
        ref_sent: Any
    ) -> float:
        query_resp = None
        for l in query_sent.strip().split('\n')[::-1]:
            l = l.lower()
            l_search = re.search(r'the answer is ([-|\$]?\d+)', l)
            
            if l_search is not None:
                query_resp = l_search.group(1).strip().replace('$','')
                break
            # end if
        # end for
        if type(query_resp)==str:
            query_resp = eval(query_resp)
        # end if
        _ref_sent = cls.get_answer_from_ground_truth(ref_sent)
        if type(_ref_sent)==str and _ref_sent!='<N/A>':
            ref_sent=eval(_ref_sent)
        # end if
        correctness = 1. if query_resp==ref_sent else 0.
        return correctness

    @classmethod
    def get_correctness_code(
        cls, 
        query_sent: str, 
        ref_sent: Any,
        code_used_for_response: Any
    ) -> float:
        query_resp = None
        if type(code_used_for_response)==str:
            for l in query_sent.strip().split('\n')[::-1]:
                l = l.lower()
                l_search = re.search(r'the answer is ([-|\$]?\d+)', l)
                
                if l_search is not None:
                    query_resp = l_search.group(1).strip().replace('$','')
                    break
                # end if
            # end for
        else:
            ans_search = re.search(r'([-|\$]?\d+)', query_sent)
            if ans_search is not None:
                query_resp = ans_search.group(1).strip().replace('$','')
            # end if
        # end if
        if type(query_resp)==str:
            query_resp = eval(query_resp)
        # end if
        _ref_sent = cls.get_answer_from_ground_truth(ref_sent)
        if type(_ref_sent)==str and _ref_sent!='<N/A>':
            ref_sent=eval(_ref_sent)
        # end if
        correctness = 1. if query_resp==ref_sent else 0.
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
    def evaluate(cls,
        model_name: str, 
        dataset_name: str,
        pl_type: str='python',
        include_only_modconst=False
    ) -> None:
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        alg_const_dir = eval_dir / 'alg_consistency'
        if include_only_modconst:
            alg_const_dir = eval_dir / 'alg_consistency_only_modconst'
        # end if

        eval_code_res_file = alg_const_dir / 'eval-results-w-code-alg-consistency.json'
        eval_code_res = Utils.read_json(eval_code_res_file)
        
        acc_res = dict()
        correctness_orig_list = list()
        correctness_code_list = list()
        
        # num_demo_list = list()
        for cksum_val in eval_code_res.keys():
            print(cksum_val)
            res_orig: Dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
            res_modcons_code: Dict = eval_code_res[cksum_val]
            correctness_orig = cls.get_correctness_orig(
                res_orig['orig']['cot_response'], 
                res_orig['orig']['answer']
            )
            correctness_code = cls.get_correctness_code(
                res_modcons_code['final_response'],
                res_modcons_code['answer'],
                res_modcons_code['code_used_for_response']
            )

            code_used_for_response = res_modcons_code['code_used_for_response']
            if type(code_used_for_response)==list:
                num_code_used = len(code_used_for_response)
            elif code_used_for_response.startswith('cot_response::all_different_ans_'):
                num_code_used = len(eval(code_used_for_response.split('cot_response::all_different_ans_')[-1]))
            else:
                num_code_used = 0
            # end if

            acc_res[cksum_val] = {
                'orig': correctness_orig,
                'code': correctness_code,
                'num_code_used': num_code_used
            }
            correctness_orig_list.append(correctness_orig)
            correctness_code_list.append(correctness_code)
        # end for
        
        acc_stats = {
            'acc-orig': cls.get_stats(correctness_orig_list),
            'acc-code': cls.get_stats(correctness_code_list)
        }
        
        Utils.write_json(
            acc_res,
            alg_const_dir / f"acc-code-res.json",
            pretty_format=True
        )

        Utils.write_json(
            acc_stats, 
            alg_const_dir / f"acc-code-stats.json",
            pretty_format=True
        )
        return


class AccuracyForModconstNDiscriminator:

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        return Utils.get_answer_from_ground_truth(str_answer)

    @classmethod
    def get_correctness(
        cls, 
        query_sent: str, 
        ref_sent: Any
    ) -> float:
        query_resp = None
        for l in query_sent.strip().split('\n')[::-1]:
            l = l.lower()
            l_search = re.search(r'the answer is ([-|\$]?\d+)', l)
            
            if l_search is not None:
                query_resp = l_search.group(1).strip().replace('$','')
                break
            # end if
        # end for
        if type(query_resp)==str:
            query_resp = eval(query_resp)
        # end if
        _ref_sent = cls.get_answer_from_ground_truth(ref_sent)
        if type(_ref_sent)==str and _ref_sent!='<N/A>':
            ref_sent=eval(_ref_sent)
        # end if
        correctness = 1. if query_resp==ref_sent else 0.
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
    def evaluate(cls,
        model_name: str, 
        dataset_name: str,
        pl_type: str='python',
    ) -> None:
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        eval_res_file = eval_dir / 'eval-results-w-modconst-n-discriminator.json'
        eval_bl_res_file = eval_dir / 'acc-res-modcons-random.json'
        eval_res = Utils.read_json(eval_res_file)
        eval_bl_res = Utils.read_json(eval_bl_res_file)
        
        acc_res = dict()
        correctness_orig_list = list()
        correctness_modconst_disc_list = list()
        
        # num_demo_list = list()
        for cksum_val in eval_bl_res.keys():
            print(cksum_val)
            res_orig: Dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
            res_modconst_disc: Dict = eval_res[cksum_val]
            correctness_orig = cls.get_correctness(
                res_orig['orig']['cot_response'], 
                res_orig['orig']['answer']
            )
            correctness_modconst_disc = cls.get_correctness(
                res_modconst_disc['response'],
                res_modconst_disc['answer']
            )
            num_demo_used = res_modconst_disc['num_demo_used']
            discriminator_scores = res_modconst_disc['discriminator_scores']
            
            acc_res[cksum_val] = {
                'orig': correctness_orig,
                'demo': correctness_modconst_disc,
                'num_demo_used': num_demo_used,
                'discriminator_scores': discriminator_scores
            }
            correctness_orig_list.append(correctness_orig)
            correctness_modconst_disc_list.append(correctness_modconst_disc)
        # end for
        
        acc_stats = {
            'acc-orig': cls.get_stats(correctness_orig_list),
            'acc-code': cls.get_stats(correctness_modconst_disc_list)
        }
        
        Utils.write_json(
            acc_res,
            eval_dir / f"acc-res-modconst-disc.json",
            pretty_format=True
        )

        Utils.write_json(
            acc_stats, 
            eval_dir / f"acc-stats-modconst-disc.json",
            pretty_format=True
        )
        return


class AccuracyForEachModal:

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
            if eval_res[cksum_val]['final_response'][modal_name] is None:
                resp = None
            else:
                resp = eval_res[cksum_val]['final_response'][modal_name].strip().lower()
            # end if
        else:
            if eval_res[modal_name] is None:
                resp = None
            else:
                resp = eval_res[modal_name].strip().lower()
            # end if
        # end if
        answers_over_modals = None
        if resp is not None:
            if modal_name=='cot_response':
                answers_over_modals = cls.get_answer_from_cot_resp(resp)
            elif modal_name=='code_response':
                answers_over_modals = cls.get_answer_from_code_resp(resp, dataset_name)
            elif modal_name=='eqn_response':
                answers_over_modals = cls.get_answer_from_eqn_resp(resp)
            # end if
        # end if
        return answers_over_modals

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
    def evaluate(cls,
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
                    correctness = cls.get_correctness(
                        orig_ans,
                        res_orig['orig']['answer']
                    )
                    acc_res[cksum_val] = {
                        'orig': correctness
                    }
                    correctness_list.append(correctness)
                    print(f"{mod_name}::{cksum_val}::{correctness}")
                    
                    Utils.write_json(
                        acc_res,
                        eval_dir / f"acc-res-bl-{mod_name}.json",
                        pretty_format=True
                    )
                # end if
            # end for

            acc_stats = {
                'acc-orig': cls.get_stats(correctness_list)
            }
            Utils.write_json(
                acc_stats, 
                eval_dir / f"acc-stats-bl-{mod_name}.json",
                pretty_format=True
            )
        # end for
        return


class AccuracyForGeneticAlg:

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
    def get_majority_answer(
        cls, 
        final_response_over_modals: Dict,
        dataset_name: str,
        final_answer_over_modals: Dict=None,
    ):  
        if final_answer_over_modals is None:
            answers_over_modals = dict()
            for modal_name, resp in final_response_over_modals.items():
                answers_over_modals[modal_name] = None
                if resp is not None:
                    if modal_name=='cot_response':
                        answers_over_modals[modal_name] = cls.get_answer_from_cot_resp(resp)
                    elif modal_name=='code_response':
                        answers_over_modals[modal_name] = cls.get_answer_from_code_resp(resp, dataset_name)
                    elif modal_name=='eqn_response':
                        answers_over_modals[modal_name] = cls.get_answer_from_eqn_resp(resp)
                    # end if
                # end if
            # end for
        else:
            answers_over_modals = final_answer_over_modals
        # end if
        c = Counter(answers_over_modals.values())
        value_freqs = c.most_common()
        if len(set([v[1] for v in value_freqs]))==1:
            # in case of having no most frequent answer
            return answers_over_modals['cot_response']
        else:
            return value_freqs[0][0]
        # end if

    @classmethod
    def get_answers_from_modals(
        cls, 
        eval_res: Dict,
        dataset_name: str,
        cksum_val: str=None
    ):
        if cksum_val is not None:
            if 'final_answer' in eval_res[cksum_val].keys():
                final_answer_over_modals = dict()
                for mod_name in Macros.prompts_over_modals.keys():
                    final_answer = eval_res[cksum_val]['final_answer'][mod_name]
                    if final_answer is not None:
                        final_answer_over_modals[mod_name] = final_answer.strip().lower()
                    else:
                        final_answer_over_modals[mod_name] = final_answer
                    # end if
                # end for
                final_answer = cls.get_majority_answer(
                    None, 
                    dataset_name,
                    final_answer_over_modals=final_answer_over_modals
                )
            else:
                final_response_over_modals = dict()
                for mod_name in Macros.prompts_over_modals.keys():
                    final_response_over_modals[mod_name] = eval_res[cksum_val]['final_response'][mod_name].strip().lower()
                # end for
                final_answer = cls.get_majority_answer(final_response_over_modals, dataset_name)
            # end if
            return final_answer
        else:
            final_response_over_modals = dict()
            for mod_name in Macros.prompts_over_modals.keys():
                final_response_over_modals[mod_name] = None
                if eval_res[mod_name] is not None:
                    final_response_over_modals[mod_name] = eval_res[mod_name].strip().lower()
                # end if
            # end for
            final_answer = cls.get_majority_answer(final_response_over_modals, dataset_name)
            return final_answer
        # end if

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
    def compute_modal_consistency(
        cls,
        answer_dict: Dict
    ) -> float:
        modal_consistency = 1.-len(set(answer_dict.values()))*1. / len(answer_dict.keys())
        return modal_consistency, answer_dict

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
    def evaluate(cls,
        llm_name: str, 
        dataset_name: str,
        pl_type: str='python',
    ) -> None:
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / llm_name
        genetic_res_dir = Macros.result_dir / 'genetic_fg' / dataset_name / 'evaluate_consistency' / llm_name
        # eval_res_file = eval_dir / 'eval-results-w-modcons-n-discriminator.json'
        # eval_bl_res_file = eval_dir / 'acc-res-modcons-random.json'
        genetic_res_file = genetic_res_dir / 'final_answers.json'
        bl_cot_res_file = eval_dir / 'acc-res-bl-cot_response.json'
        bl_code_res_file = eval_dir / 'acc-res-bl-code_response.json'
        bl_eqn_res_file = eval_dir / 'acc-res-bl-eqn_response.json'

        # eval_res = Utils.read_json(eval_res_file)
        # eval_bl_res = Utils.read_json(eval_bl_res_file)
        genetic_res = Utils.read_json(genetic_res_file)
        res_bl_cot: Dict = Utils.read_json(bl_cot_res_file)
        res_bl_code: Dict = Utils.read_json(bl_code_res_file)
        res_bl_eqn: Dict = Utils.read_json(bl_eqn_res_file)
        
        acc_res = dict()
        correctness_bl_cot_list = list()
        correctness_bl_code_list = list()
        correctness_bl_eqn_list = list()
        correctness_maj_list = list()
        correctness_genetic_list = list()

        cksum_vals_orig_mod_const = list()
        cksum_vals_orig_mod_const_and_correct = list()
        cksum_vals_mod_const_inc_by_mut = list()
        cksum_vals_mod_const_not_inc_by_mut = list()
        cksum_vals_mod_const_inc_by_mut_and_correct = list()
        cksum_vals_mod_const_not_inc_by_mut_and_correct = list()
        cksum_vals_using_muts = list()
        cksum_vals_not_orig_mod_const_but_not_use_mut = list()
        cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct = list()
        cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct = list()
        mod_const_not_orig_mod_const_but_not_use_mut_and_correct = list()
        mod_const_not_orig_mod_const_but_not_use_mut_and_not_correct = list()
        num_mod_const_muts_not_orig_mod_const_and_use_mut_and_correct = list()
        num_mod_const_muts_not_orig_mod_const_and_use_mut_and_not_correct = list()
        num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_correct = list()
        num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_not_correct = list()

        max_cons_val = 1.-(1./len(Macros.prompts_over_modals.keys()))
        
        # num_demo_list = list()
        for cksum_val in genetic_res.keys():
            eval_res = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
            mod_const_muts = list()
            # for m_i, m in enumerate(eval_res['mutation']):
            #     answer_dict = m.get('ans_dict', None)
            #     m_cons_mut, _ = cls.compute_modal_consistency(answer_dict)
            #     if m_cons_mut==max_cons_val:
            #         mod_const_muts.append(m_i)
            #     # end if
            # # end for
            if cksum_val not in acc_res.keys():
                
                correctness_bl_cot = res_bl_cot[cksum_val]['orig']
                correctness_bl_code = res_bl_code[cksum_val]['orig']
                correctness_bl_eqn = res_bl_eqn[cksum_val]['orig']

                maj_ans = cls.get_answers_from_modals(
                    eval_res['orig'],
                    dataset_name
                )
                
                correctness_maj = cls.get_correctness(
                    maj_ans,
                    eval_res['orig']['answer']
                )
                
                gt_ans = genetic_res[cksum_val]['answer']

                acc_res[cksum_val] = {
                    'bl_cot': correctness_bl_cot,
                    'bl_code': correctness_bl_code,
                    'bl_eqn': correctness_bl_eqn,
                    'maj': correctness_maj
                }
                correctness_maj_list.append(correctness_maj)
                correctness_bl_cot_list.append(correctness_bl_cot)
                correctness_bl_code_list.append(correctness_bl_code)
                correctness_bl_eqn_list.append(correctness_bl_eqn)

                Utils.write_json(
                    acc_res,
                    genetic_res_dir / f"acc-res-genetic.json",
                    pretty_format=True
                )
            else:
                correctness_maj_list.append(acc_res[cksum_val]['maj'])
                correctness_bl_cot_list.append(acc_res[cksum_val]['bl_cot'])
                correctness_bl_code_list.append(acc_res[cksum_val]['bl_code'])
                correctness_bl_eqn_list.append(acc_res[cksum_val]['bl_eqn'])
            # end if
        # end for
        
        acc_stats = {
            'acc-bl-cot': cls.get_stats(correctness_bl_cot_list),
            'acc-bl-code': cls.get_stats(correctness_bl_code_list),
            'acc-bl-eqn': cls.get_stats(correctness_bl_eqn_list),
            'acc-maj': cls.get_stats(correctness_maj_list)
        }
        
        Utils.write_json(
            acc_res,
            genetic_res_dir / f"acc-res-genetic.json",
            pretty_format=True
        )

        Utils.write_json(
            acc_stats, 
            genetic_res_dir / f"acc-stats-genetic.json",
            pretty_format=True
        )
        return

    @classmethod
    def compare_with_baseline(cls,
        llm_name: str, 
        dataset_name: str,
        pl_type: str='python',
    ) -> None:
        bl_res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        bl_eval_dir = bl_res_dir / 'evaluate_consistency' / llm_name
        genetic_res_dir = Macros.result_dir / 'genetic' / dataset_name / 'evaluate_consistency' / llm_name

        acc_res_genetic = Utils.read_json(
            genetic_res_dir / f"acc-res-genetic.json"
        )
        acc_res_bl = Utils.read_json(
            bl_eval_dir / f"acc-res-modcons-random.json"
        )

        correctness_orig_list = list()
        correctness_genetic_list = list()
        correctness_bl_list = list()
        correctness_fix_by_bl = list()
        correctness_fix_by_genetic = list()
        correctness_fix_only_by_bl = list()
        correctness_fix_only_by_genetic = list()

        correctness_damage_by_bl = list()
        correctness_damage_by_genetic = list()
        correctness_damage_only_by_bl = list()
        correctness_damage_only_by_genetic = list()

        for cksum_val in acc_res_genetic.keys():
            bl_res = acc_res_bl.get(cksum_val, None)
            if bl_res is not None:
                correctness_orig = acc_res_genetic[cksum_val]['orig']
                correctness_genetic = acc_res_genetic[cksum_val]['demo']
                correctness_bl = bl_res['demo']

                correctness_orig_list.append(correctness_orig)
                correctness_genetic_list.append(correctness_genetic)
                correctness_bl_list.append(correctness_bl)

                if correctness_orig==0.0 and \
                    correctness_bl==1.0:
                    correctness_fix_by_bl.append(cksum_val)
                # end if

                if correctness_orig==0.0 and \
                    correctness_genetic==1.0:
                    correctness_fix_by_genetic.append(cksum_val)
                # end if

                if correctness_orig==0.0 and \
                    correctness_bl==0.0 and \
                    correctness_genetic==1.0:
                    correctness_fix_only_by_genetic.append(cksum_val)
                # end if

                if correctness_orig==0.0 and \
                    correctness_bl==1.0 and \
                    correctness_genetic==0.0:
                    correctness_fix_only_by_bl.append(cksum_val)
                # end if

                if correctness_orig==1.0 and \
                    correctness_bl==0.0:
                    correctness_damage_by_bl.append(cksum_val)
                # end if

                if correctness_orig==1.0 and \
                    correctness_genetic==0.0:
                    correctness_damage_by_genetic.append(cksum_val)
                # end if

                if correctness_orig==1.0 and \
                    correctness_bl==1.0 and \
                    correctness_genetic==0.0:
                    correctness_damage_only_by_genetic.append(cksum_val)
                # end if

                if correctness_orig==1.0 and \
                    correctness_bl==0.0 and \
                    correctness_genetic==1.0:
                    correctness_damage_only_by_bl.append(cksum_val)
                # end if
            # end if
        # end for
        acc_stats = {
            'acc-orig': cls.get_stats(correctness_orig_list),
            'acc-genetic': cls.get_stats(correctness_genetic_list),
            'acc-modcons-random': cls.get_stats(correctness_bl_list),
            'num-fix-by-bl': len(correctness_fix_by_bl),
            'num-fix-by-genetic': len(correctness_fix_by_genetic),
            'num-fix-only-by-bl': len(correctness_fix_only_by_bl),
            'num-fix-only-by-genetic': len(correctness_fix_only_by_genetic),
            'num-damage-by-bl': len(correctness_damage_by_bl),
            'num-damage-by-genetic': len(correctness_damage_by_genetic),
            'num-damage-only-by-bl': len(correctness_damage_only_by_bl),
            'num-damage-only-by-genetic': len(correctness_damage_only_by_genetic)
        }
        Utils.write_json(
            acc_stats,
            genetic_res_dir / f"acc-stats-genetic-n-modcons-random.json",
            pretty_format=True
        )
        return


class AccuracyForGeneticAlgSelfConsistency:

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
    def get_majority_answer(
        cls, 
        answer_list: List
    ):  
        c = Counter(answer_list)
        value_freqs = c.most_common()
        if len(set([v[1] for v in value_freqs]))==1:
            # in case of having no most frequent answer
            return answer_list[0]
        else:
            return value_freqs[0][0]
        # end if

    @classmethod
    def get_answers_from_modals(
        cls, 
        eval_res: Dict,
        dataset_name: str,
        target_modality: str,
        cksum_val: str=None
    ):
        final_answer = cls.get_majority_answer(eval_res[cksum_val]['fianl_answer'][target_modality])
        return final_answer

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
    def evaluate(cls,
        llm_name: str, 
        dataset_name: str,
        target_modality: str='cot_response',
        pl_type: str='python',
    ) -> None:
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / llm_name
        genetic_res_dir = Macros.result_dir / 'genetic' / dataset_name / 'evaluate_consistency' / llm_name
        genetic_fg_res_dir = Macros.result_dir / 'genetic_fg' / dataset_name / 'evaluate_consistency' / llm_name

        # eval_res_file = eval_dir / 'eval-results-w-modcons-n-discriminator.json'
        # eval_bl_res_file = eval_dir / 'acc-res-modcons-random.json'
        selfconst_res_file = genetic_res_dir / f"final_answers_w_self_consistency_over_{target_modality}.json"
        bl_genetic_res_file = genetic_fg_res_dir / 'acc-res-genetic-fg.json'

        # eval_res = Utils.read_json(eval_res_file)
        # eval_bl_res = Utils.read_json(eval_bl_res_file)
        selfconst_res: Dict = Utils.read_json(selfconst_res_file)
        bl_genetic_res: Dict = Utils.read_json(bl_genetic_res_file)

        acc_res = dict()
        correctness_selfconst_list = list()
        correctness_bl_genetic_list = list()
        
        # num_demo_list = list()
        for cksum_val in selfconst_res.keys():
            print(cksum_val)
            res_orig: Dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
            # eval_res_dict: Dict = eval_res[cksum_val]

            selfconst_ans = cls.get_answers_from_modals(
                selfconst_res,
                dataset_name,
                target_modality,
                cksum_val=cksum_val
            )

            correctness_genetic = bl_genetic_res[cksum_val]['genetic']
            correctness_selfconst = cls.get_correctness(
                selfconst_ans,
                selfconst_res[cksum_val]['answer']
            )
            num_demo_used = len(selfconst_res[cksum_val]['mutation'])
            # print(genetic_ans, gt_ans, correctness_genetic, num_demo_used)
            
            acc_res[cksum_val] = {
                'genetic': correctness_genetic,
                'genetic_w_selfconst': correctness_selfconst,
                'num_demo_used': num_demo_used
            }
            correctness_selfconst_list.append(correctness_selfconst)
            correctness_bl_genetic_list.append(correctness_genetic)
        # end for
        
        acc_stats = {
            'acc-genetic': cls.get_stats(correctness_bl_genetic_list),
            'acc-genetic_w_selfconst': cls.get_stats(correctness_selfconst_list)
        }
        
        Utils.write_json(
            acc_res,
            genetic_fg_res_dir / f"acc-res-genetic-w-selfconst.json",
            pretty_format=True
        )

        Utils.write_json(
            acc_stats, 
            genetic_fg_res_dir / f"acc-stats-genetic-w-selfconst.json",
            pretty_format=True
        )
        return


class AccuracyForAutoCot:

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        return Utils.get_answer_from_ground_truth(str_answer)

    @classmethod
    def get_answer_from_cot_resp(cls, cot_resp: str) -> str:
        return Utils.get_answer_from_cot_resp(cot_resp)

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
    def evaluate(cls,
        llm_name: str, 
        dataset_name: str,
        pl_type: str='python',
    ) -> None:
        res_dir = Macros.result_dir / 'autocot' /  'evaluate' / dataset_name / llm_name
        eval_files = sorted([
            f for f in os.listdir(str(res_dir))
            if f.endswith('.json') and f.startswith('eval-')
        ])

        acc_res = dict()
        correctness_autocot_list = list()

        for e in eval_files:
            eval_file = res_dir / e
            eval_res: Dict = Utils.read_json(eval_file)
            cksum_val = re.search(r'eval\-(.*)\.json', e).group(1)

            answer_str = eval_res['answer']
            cot_resp_str = eval_res['response']
            if type(cot_resp_str)!=str:
                cot_resp_str = eval_res['response']['msg']
            # end if

            eval_gt_answer = cls.get_answer_from_ground_truth(answer_str)
            eval_cot_resp = None
            if eval_res['response'] is not None:
                eval_cot_resp = cls.get_answer_from_cot_resp(cot_resp_str)
            # end if
            correctness = cls.get_correctness(
                eval_cot_resp,
                eval_gt_answer
            )
            correctness_autocot_list.append(correctness)

            acc_res[cksum_val] = correctness
            Utils.write_json(
                acc_res,
                res_dir / f"acc-res-autocot.json",
                pretty_format=True
            )
        # end for
        Utils.write_json(
            acc_res,
            res_dir / f"acc-res-autocot.json",
            pretty_format=True
        )

        acc_stats = cls.get_stats(correctness_autocot_list)
        Utils.write_json(
            acc_stats, 
            res_dir / f"acc-stats-autocot.json",
            pretty_format=True
        )
        return


class AccuracyForGeneticFgAlg:

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
    def get_answer_by_cons_scores(
        cls, 
        cons_score_dict_over_answers: Dict
    ):
        if len(cons_score_dict_over_answers.keys())==1:
            return list(cons_score_dict_over_answers.keys())[0]
        # end if        
        final_ans = list()
        max_cons_score = max(cons_score_dict_over_answers.values())
        for ans_per_mod, score in cons_score_dict_over_answers.items():
            if score==max_cons_score:
                final_ans.append(ans_per_mod)
            # end if
        # end for
        return final_ans[0]

    @classmethod
    def get_answers_from_modals(
        cls, 
        eval_res: Dict,
        dataset_name: str,
        cksum_val: str=None
    ):
        if cksum_val is not None:
            if 'final_answer' in eval_res[cksum_val].keys():
                final_answer_over_modals = dict()
                for mod_name in Macros.prompts_over_modals.keys():
                    final_answer = eval_res[cksum_val]['final_answer'][mod_name]
                    if final_answer is not None:
                        final_answer_over_modals[mod_name] = final_answer.strip().lower()
                    else:
                        final_answer_over_modals[mod_name] = final_answer
                    # end if
                # end for
                final_answer = cls.get_majority_answer(
                    None, 
                    dataset_name,
                    final_answer_over_modals=final_answer_over_modals
                )
            else:
                final_response_over_modals = dict()
                for mod_name in Macros.prompts_over_modals.keys():
                    final_response_over_modals[mod_name] = eval_res[cksum_val]['final_response'][mod_name].strip().lower()
                # end for
                final_answer = cls.get_majority_answer(final_response_over_modals, dataset_name)
            # end if
            return final_answer
        else:
            final_response_over_modals = dict()
            for mod_name in Macros.prompts_over_modals.keys():
                final_response_over_modals[mod_name] = eval_res[mod_name].strip().lower()
            # end for
            final_answer = cls.get_majority_answer(final_response_over_modals, dataset_name)
            return final_answer
        # end if

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
    def compute_modal_consistency(
        cls,
        answer_dict: Dict
    ) -> float:
        modal_consistency = 1.-len(set(answer_dict.values()))*1. / len(answer_dict.keys())
        return modal_consistency, answer_dict

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
    def evaluate(cls,
        llm_name: str, 
        dataset_name: str,
        pl_type: str='python',
    ) -> None:
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / llm_name
        genetic_res_dir = Macros.result_dir / 'genetic_fg' / dataset_name / 'evaluate_consistency' / llm_name
        # eval_res_file = eval_dir / 'eval-results-w-modcons-n-discriminator.json'
        # eval_bl_res_file = eval_dir / 'acc-res-modcons-random.json'
        genetic_res_file = genetic_res_dir / 'final_answers.json'

        # eval_res = Utils.read_json(eval_res_file)
        # eval_bl_res = Utils.read_json(eval_bl_res_file)
        genetic_res = Utils.read_json(genetic_res_file)
        
        acc_res = dict()
        if os.path.exists(str(genetic_res_dir / 'acc-res-genetic-fg.json')):
            acc_res = Utils.read_json(
                genetic_res_dir / 'acc-res-genetic-fg.json'
            )
        # end if
        correctness_genetic_list = list()

        cksum_vals_orig_mod_const = list()
        cksum_vals_orig_mod_const_and_correct = list()
        cksum_vals_mod_const_inc_by_mut = list()
        cksum_vals_mod_const_not_inc_by_mut = list()
        cksum_vals_mod_const_inc_by_mut_and_correct = list()
        cksum_vals_mod_const_not_inc_by_mut_and_correct = list()
        cksum_vals_using_muts = list()
        cksum_vals_not_orig_mod_const_but_not_use_mut = list()
        cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct = list()
        cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct = list()
        mod_const_not_orig_mod_const_but_not_use_mut_and_correct = list()
        mod_const_not_orig_mod_const_but_not_use_mut_and_not_correct = list()
        num_mod_const_muts_not_orig_mod_const_and_use_mut_and_correct = list()
        num_mod_const_muts_not_orig_mod_const_and_use_mut_and_not_correct = list()
        num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_correct = list()
        num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_not_correct = list()

        max_cons_val = 1.-(1./len(Macros.prompts_over_modals.keys()))
        
        # num_demo_list = list()
        for cksum_val in genetic_res.keys():
            if cksum_val not in acc_res.keys():
                eval_res = Utils.read_json(eval_dir / f"fg-eval-{cksum_val}.json")
                # from the result, how many are using the mutation, and howmany is corrected using mutations, howmany consistent
                print(cksum_val)

                orig_mod_const = genetic_res[cksum_val]['original_consistency_score']
                final_mod_const = genetic_res[cksum_val]['final_consistency_score']
                num_demo_used = len(genetic_res[cksum_val]['mutation'])

                if orig_mod_const==1.0:
                    cksum_vals_orig_mod_const.append(cksum_val)
                # end if

                if num_demo_used>0 and orig_mod_const<final_mod_const:
                    cksum_vals_mod_const_inc_by_mut.append(cksum_val)
                # end if

                if num_demo_used>0:
                    cksum_vals_using_muts.append(cksum_val)
                # end if

                if num_demo_used>0 and orig_mod_const==final_mod_const:
                    cksum_vals_mod_const_not_inc_by_mut.append(cksum_val)
                # end if

                if num_demo_used==0 and orig_mod_const<1.0:
                    cksum_vals_not_orig_mod_const_but_not_use_mut.append(cksum_val)
                # end if

                # res_orig: Dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
                # # eval_res_dict: Dict = eval_res[cksum_val]

                genetic_ans = cls.get_answer_by_cons_scores(
                    genetic_res[cksum_val]['score_dict_over_unique_answers']
                )

                gt_ans = genetic_res[cksum_val]['answer']
                correctness_genetic = cls.get_correctness(
                    genetic_ans,
                    gt_ans
                )

                if correctness_genetic==1. and \
                    orig_mod_const==1.0:
                    cksum_vals_orig_mod_const_and_correct.append(cksum_val)
                # end if

                if correctness_genetic==1. and \
                    num_demo_used>0 and \
                    orig_mod_const<final_mod_const:
                    cksum_vals_mod_const_inc_by_mut_and_correct.append(cksum_val)
                # end if

                if correctness_genetic==1. and \
                    num_demo_used>0 and \
                    orig_mod_const==final_mod_const:
                    cksum_vals_mod_const_not_inc_by_mut_and_correct.append(cksum_val)
                # end if

                if correctness_genetic==1. and \
                    num_demo_used==0 and \
                    orig_mod_const<1.0:
                    cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct.append(cksum_val)
                    mod_const_not_orig_mod_const_but_not_use_mut_and_correct.append(orig_mod_const)
                elif correctness_genetic==0. and \
                    num_demo_used==0 and \
                    orig_mod_const<1.0:
                    cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct.append(cksum_val)
                    mod_const_not_orig_mod_const_but_not_use_mut_and_not_correct.append(orig_mod_const)
                # end if

                # print(genetic_ans, gt_ans, correctness_genetic, num_demo_used)
                
                acc_res[cksum_val] = {
                    'genetic': correctness_genetic,
                    'num_demo_used': num_demo_used,
                    'cksum_vals_orig_mod_const': 1. if cksum_val in cksum_vals_orig_mod_const else 0.,
                    'cksum_vals_using_muts': 1. if cksum_val in cksum_vals_using_muts else 0.,
                    'cksum_vals_mod_const_inc_by_mut':1. if cksum_val in cksum_vals_mod_const_inc_by_mut else 0.,
                    'cksum_vals_mod_const_not_inc_by_mut': 1. if cksum_val in cksum_vals_mod_const_not_inc_by_mut else 0.,
                    'cksum_vals_not_orig_mod_const_but_not_use_mut': 1. if cksum_val in cksum_vals_not_orig_mod_const_but_not_use_mut else 0.,
                    'cksum_vals_orig_mod_const_and_correct': 1. if cksum_val in cksum_vals_orig_mod_const_and_correct else 0.,
                    'cksum_vals_mod_const_inc_by_mut_and_correct': 1. if cksum_val in cksum_vals_mod_const_inc_by_mut_and_correct else 0.,
                    'cksum_vals_mod_const_not_inc_by_mut_and_correct': 1. if cksum_val in cksum_vals_mod_const_not_inc_by_mut_and_correct else 0.,
                    'cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct': 1. if cksum_val in cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct else 0.,
                    'cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct': 1. if cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct else 0.,
                }
                correctness_genetic_list.append(correctness_genetic)

                Utils.write_json(
                    acc_res,
                    genetic_res_dir / 'acc-res-genetic-fg.json',
                    pretty_format=True
                )
            # end if
        # end for
        
        acc_stats = {
            'acc-genetic': cls.get_stats(correctness_genetic_list),
            'num-genetic-cksum_vals_orig_mod_const': len(cksum_vals_orig_mod_const),
            'num-genetic-cksum_vals_using_muts': len(cksum_vals_using_muts),
            'num-genetic-cksum_vals_mod_const_inc_by_mut': len(cksum_vals_mod_const_inc_by_mut),
            'num-genetic-cksum_vals_mod_const_not_inc_by_mut': len(cksum_vals_mod_const_not_inc_by_mut),
            'num-genetic-cksum_vals_not_orig_mod_const_but_not_use_mut': len(cksum_vals_not_orig_mod_const_but_not_use_mut),
            'num-genetic-cksum_vals_orig_mod_const_and_correct': len(cksum_vals_orig_mod_const_and_correct),
            'num-genetic-cksum_vals_mod_const_inc_by_mut_and_correct': len(cksum_vals_mod_const_inc_by_mut_and_correct),
            'num-genetic-cksum_vals_mod_const_not_inc_by_mut_and_correct': len(cksum_vals_mod_const_not_inc_by_mut_and_correct),
            'num-genetic-cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct': len(cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct),
            'num-genetic-cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct': len(cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct),
            'cksum_vals_orig_mod_const': cksum_vals_orig_mod_const,
            'cksum_vals_using_muts': cksum_vals_using_muts,
            'cksum_vals_mod_const_inc_by_mut': cksum_vals_mod_const_inc_by_mut,
            'cksum_vals_mod_const_not_inc_by_mut': cksum_vals_mod_const_not_inc_by_mut,
            'cksum_vals_mod_const_inc_by_mut_and_correct': cksum_vals_mod_const_inc_by_mut_and_correct,
            'cksum_vals_mod_const_not_inc_by_mut_and_correct': cksum_vals_mod_const_not_inc_by_mut_and_correct,
            'cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct': cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct,
            'cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct': cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct,
            'mod_const_not_orig_mod_const_but_not_use_mut_and_correct': mod_const_not_orig_mod_const_but_not_use_mut_and_correct,
            'mod_const_not_orig_mod_const_but_not_use_mut_and_not_correct': mod_const_not_orig_mod_const_but_not_use_mut_and_not_correct,
            'num_mod_const_muts_not_orig_mod_const_and_use_mut_and_correct': num_mod_const_muts_not_orig_mod_const_and_use_mut_and_correct,
            'num_mod_const_muts_not_orig_mod_const_and_use_mut_and_not_correct': num_mod_const_muts_not_orig_mod_const_and_use_mut_and_not_correct,
            'num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_correct': num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_correct,
            'num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_not_correct': num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_not_correct
        }
        
        Utils.write_json(
            acc_res,
            genetic_res_dir / 'acc-res-genetic-fg.json',
            pretty_format=True
        )

        Utils.write_json(
            acc_stats, 
            genetic_res_dir / 'acc-stats-genetic-fg.json',
            pretty_format=True
        )
        return


class AccuracyForGeneticFgAlgWithValidationCls:

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
    def get_answer_by_cons_scores(
        cls, 
        cons_score_dict_over_answers: Dict
    ):
        if len(cons_score_dict_over_answers.keys())==1:
            return list(cons_score_dict_over_answers.keys())[0]
        # end if        
        final_ans = list()
        max_cons_score = max(cons_score_dict_over_answers.values())
        for ans_per_mod, score in cons_score_dict_over_answers.items():
            if score==max_cons_score:
                final_ans.append(ans_per_mod)
            # end if
        # end for
        return final_ans[0]

    @classmethod
    def get_answers_from_modals(
        cls, 
        eval_res: Dict,
        dataset_name: str,
        cksum_val: str=None
    ):
        if cksum_val is not None:
            if 'final_answer' in eval_res[cksum_val].keys():
                final_answer_over_modals = dict()
                for mod_name in Macros.prompts_over_modals.keys():
                    final_answer = eval_res[cksum_val]['final_answer'][mod_name]
                    if final_answer is not None:
                        final_answer_over_modals[mod_name] = final_answer.strip().lower()
                    else:
                        final_answer_over_modals[mod_name] = final_answer
                    # end if
                # end for
                final_answer = cls.get_majority_answer(
                    None, 
                    dataset_name,
                    final_answer_over_modals=final_answer_over_modals
                )
            else:
                final_response_over_modals = dict()
                for mod_name in Macros.prompts_over_modals.keys():
                    final_response_over_modals[mod_name] = eval_res[cksum_val]['final_response'][mod_name].strip().lower()
                # end for
                final_answer = cls.get_majority_answer(final_response_over_modals, dataset_name)
            # end if
            return final_answer
        else:
            final_response_over_modals = dict()
            for mod_name in Macros.prompts_over_modals.keys():
                final_response_over_modals[mod_name] = eval_res[mod_name].strip().lower()
            # end for
            final_answer = cls.get_majority_answer(final_response_over_modals, dataset_name)
            return final_answer
        # end if

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
    def compute_modal_consistency(
        cls,
        answer_dict: Dict
    ) -> float:
        modal_consistency = 1.-len(set(answer_dict.values()))*1. / len(answer_dict.keys())
        return modal_consistency, answer_dict

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
    def evaluate(cls,
        llm_name: str, 
        dataset_name: str,
        pl_type: str='python',
    ) -> None:
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        # mut_dir = res_dir / 'mutation_with_validation_cls' / llm_name
        eval_dir = res_dir / 'evaluate_consistency' / llm_name
        genetic_res_dir = Macros.result_dir / 'genetic_fg_with_validation_cls'/ dataset_name / 'evaluate_consistency' / llm_name

        # eval_res_file = eval_dir / 'eval-results-w-modcons-n-discriminator.json'
        # eval_bl_res_file = eval_dir / 'acc-res-modcons-random.json'
        genetic_res_file = genetic_res_dir / 'final_answers.json'

        # eval_res = Utils.read_json(eval_res_file)
        # eval_bl_res = Utils.read_json(eval_bl_res_file)
        genetic_res = Utils.read_json(genetic_res_file)
        
        acc_res = dict()
        if os.path.exists(str(genetic_res_dir / 'acc-res-genetic-fg-w-val-cls.json')):
            acc_res = Utils.read_json(
                genetic_res_dir / 'acc-res-genetic-fg-w-val-cls.json'
            )
        # end if
        correctness_genetic_list = list()
        cksum_vals_orig_mod_const = list()
        cksum_vals_orig_mod_const_and_correct = list()
        cksum_vals_mod_const_inc_by_mut = list()
        cksum_vals_mod_const_not_inc_by_mut = list()
        cksum_vals_mod_const_inc_by_mut_and_correct = list()
        cksum_vals_mod_const_not_inc_by_mut_and_correct = list()
        cksum_vals_using_muts = list()
        cksum_vals_not_orig_mod_const_but_not_use_mut = list()
        cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct = list()
        cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct = list()
        mod_const_not_orig_mod_const_but_not_use_mut_and_correct = list()
        mod_const_not_orig_mod_const_but_not_use_mut_and_not_correct = list()
        num_mod_const_muts_not_orig_mod_const_and_use_mut_and_correct = list()
        num_mod_const_muts_not_orig_mod_const_and_use_mut_and_not_correct = list()
        num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_correct = list()
        num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_not_correct = list()
        
        # num_demo_list = list()
        for cksum_val in genetic_res.keys():
            if cksum_val not in acc_res.keys():
                eval_res = Utils.read_json(eval_dir / f"fg-w-val-cls-eval-{cksum_val}.json")
                # from the result, how many are using the mutation, and howmany is corrected using mutations, howmany consistent
                print(cksum_val)

                orig_mod_const = genetic_res[cksum_val]['original_consistency_score']
                final_mod_const = genetic_res[cksum_val]['final_consistency_score']
                num_demo_used = len(genetic_res[cksum_val]['mutation'])

                if orig_mod_const==1.0:
                    cksum_vals_orig_mod_const.append(cksum_val)
                # end if

                if num_demo_used>0 and orig_mod_const<final_mod_const:
                    cksum_vals_mod_const_inc_by_mut.append(cksum_val)
                # end if

                if num_demo_used>0:
                    cksum_vals_using_muts.append(cksum_val)
                # end if

                if num_demo_used>0 and orig_mod_const==final_mod_const:
                    cksum_vals_mod_const_not_inc_by_mut.append(cksum_val)
                # end if

                if num_demo_used==0 and orig_mod_const<1.0:
                    cksum_vals_not_orig_mod_const_but_not_use_mut.append(cksum_val)
                # end if

                # res_orig: Dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
                # # eval_res_dict: Dict = eval_res[cksum_val]

                genetic_ans = cls.get_answer_by_cons_scores(
                    genetic_res[cksum_val]['score_dict_over_unique_answers']
                )

                gt_ans = genetic_res[cksum_val]['answer']
                correctness_genetic = cls.get_correctness(
                    genetic_ans,
                    gt_ans
                )

                if correctness_genetic==1. and \
                    orig_mod_const==1.0:
                    cksum_vals_orig_mod_const_and_correct.append(cksum_val)
                # end if

                if correctness_genetic==1. and \
                    num_demo_used>0 and \
                    orig_mod_const<final_mod_const:
                    cksum_vals_mod_const_inc_by_mut_and_correct.append(cksum_val)
                # end if

                if correctness_genetic==1. and \
                    num_demo_used>0 and \
                    orig_mod_const==final_mod_const:
                    cksum_vals_mod_const_not_inc_by_mut_and_correct.append(cksum_val)
                # end if

                if correctness_genetic==1. and \
                    num_demo_used==0 and \
                    orig_mod_const<1.0:
                    cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct.append(cksum_val)
                    mod_const_not_orig_mod_const_but_not_use_mut_and_correct.append(orig_mod_const)
                elif correctness_genetic==0. and \
                    num_demo_used==0 and \
                    orig_mod_const<1.0:
                    cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct.append(cksum_val)
                    mod_const_not_orig_mod_const_but_not_use_mut_and_not_correct.append(orig_mod_const)
                # end if

                # print(genetic_ans, gt_ans, correctness_genetic, num_demo_used)
                
                acc_res[cksum_val] = {
                    'genetic': correctness_genetic,
                    'num_demo_used': num_demo_used,
                    'cksum_vals_orig_mod_const': 1. if cksum_val in cksum_vals_orig_mod_const else 0.,
                    'cksum_vals_using_muts': 1. if cksum_val in cksum_vals_using_muts else 0.,
                    'cksum_vals_mod_const_inc_by_mut':1. if cksum_val in cksum_vals_mod_const_inc_by_mut else 0.,
                    'cksum_vals_mod_const_not_inc_by_mut': 1. if cksum_val in cksum_vals_mod_const_not_inc_by_mut else 0.,
                    'cksum_vals_not_orig_mod_const_but_not_use_mut': 1. if cksum_val in cksum_vals_not_orig_mod_const_but_not_use_mut else 0.,
                    'cksum_vals_orig_mod_const_and_correct': 1. if cksum_val in cksum_vals_orig_mod_const_and_correct else 0.,
                    'cksum_vals_mod_const_inc_by_mut_and_correct': 1. if cksum_val in cksum_vals_mod_const_inc_by_mut_and_correct else 0.,
                    'cksum_vals_mod_const_not_inc_by_mut_and_correct': 1. if cksum_val in cksum_vals_mod_const_not_inc_by_mut_and_correct else 0.,
                    'cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct': 1. if cksum_val in cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct else 0.,
                    'cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct': 1. if cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct else 0.,
                }
                correctness_genetic_list.append(correctness_genetic)

                Utils.write_json(
                    acc_res,
                    genetic_res_dir / 'acc-res-genetic-fg-w-val-cls.json',
                    pretty_format=True
                )
            # end if
        # end for
        
        acc_stats = {
            'acc-genetic': cls.get_stats(correctness_genetic_list),
            'num-genetic-cksum_vals_orig_mod_const': len(cksum_vals_orig_mod_const),
            'num-genetic-cksum_vals_using_muts': len(cksum_vals_using_muts),
            'num-genetic-cksum_vals_mod_const_inc_by_mut': len(cksum_vals_mod_const_inc_by_mut),
            'num-genetic-cksum_vals_mod_const_not_inc_by_mut': len(cksum_vals_mod_const_not_inc_by_mut),
            'num-genetic-cksum_vals_not_orig_mod_const_but_not_use_mut': len(cksum_vals_not_orig_mod_const_but_not_use_mut),
            'num-genetic-cksum_vals_orig_mod_const_and_correct': len(cksum_vals_orig_mod_const_and_correct),
            'num-genetic-cksum_vals_mod_const_inc_by_mut_and_correct': len(cksum_vals_mod_const_inc_by_mut_and_correct),
            'num-genetic-cksum_vals_mod_const_not_inc_by_mut_and_correct': len(cksum_vals_mod_const_not_inc_by_mut_and_correct),
            'num-genetic-cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct': len(cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct),
            'num-genetic-cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct': len(cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct),
            'cksum_vals_orig_mod_const': cksum_vals_orig_mod_const,
            'cksum_vals_using_muts': cksum_vals_using_muts,
            'cksum_vals_mod_const_inc_by_mut': cksum_vals_mod_const_inc_by_mut,
            'cksum_vals_mod_const_not_inc_by_mut': cksum_vals_mod_const_not_inc_by_mut,
            'cksum_vals_mod_const_inc_by_mut_and_correct': cksum_vals_mod_const_inc_by_mut_and_correct,
            'cksum_vals_mod_const_not_inc_by_mut_and_correct': cksum_vals_mod_const_not_inc_by_mut_and_correct,
            'cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct': cksum_vals_not_orig_mod_const_but_not_use_mut_and_correct,
            'cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct': cksum_vals_not_orig_mod_const_but_not_use_mut_and_not_correct,
            'mod_const_not_orig_mod_const_but_not_use_mut_and_correct': mod_const_not_orig_mod_const_but_not_use_mut_and_correct,
            'mod_const_not_orig_mod_const_but_not_use_mut_and_not_correct': mod_const_not_orig_mod_const_but_not_use_mut_and_not_correct,
            'num_mod_const_muts_not_orig_mod_const_and_use_mut_and_correct': num_mod_const_muts_not_orig_mod_const_and_use_mut_and_correct,
            'num_mod_const_muts_not_orig_mod_const_and_use_mut_and_not_correct': num_mod_const_muts_not_orig_mod_const_and_use_mut_and_not_correct,
            'num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_correct': num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_correct,
            'num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_not_correct': num_mod_const_muts_not_orig_mod_const_but_not_use_mut_and_not_correct
        }
        
        Utils.write_json(
            acc_res,
            genetic_res_dir / 'acc-res-genetic-fg-w-val-cls.json',
            pretty_format=True
        )

        Utils.write_json(
            acc_stats, 
            genetic_res_dir / 'acc-stats-genetic-fg-w-val-cls.json',
            pretty_format=True
        )
        return
