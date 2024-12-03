
import re, os
import sys
import time
import random
import argparse
import numpy as np

from typing import *
from pathlib import Path
from numpy.linalg import norm

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..metric.metrics import AccuracyForDemo

from sentence_transformers import SentenceTransformer


class AnalyzeResponse:

    @classmethod
    def read_eval_files(cls, eval_dir: Path) -> List[str]:
        eval_files = [
            eval_dir / f for f in os.listdir(str(eval_dir))
            if f.endswith('.json') and f.startswith('eval-')
        ]
        return eval_files

    @classmethod
    def read_eval_results(cls, eval_file: Path) -> Dict:
        return Utils.read_json(eval_file)

    @classmethod
    def trim_response(
        cls, 
        question: str, 
        response: List[str]
    ) -> List[str]:
        question_w_prompt = f"{question} {Macros.llama_prompt}"
        resp = list()
        for r in response:
            r = r.replace(question_w_prompt, '')
            resp.append(r.strip())
        # end for
        return resp

    @classmethod
    def get_trimmed_response(
        cls, 
        eval_result: Dict
    ) -> Dict:
        resps = dict()
        # original
        res_orig = eval_result['orig']
        resp_orig = cls.trim_response(
            res_orig['question'],
            res_orig['response'],
        )
        resps['orig'] = {
            'question': res_orig['question'],
            'response': resp_orig,
            'answer': res_orig['answer']
        }

        # mutations
        resps['mutation'] = list()
        res_muts = eval_result['mutation']
        for res_mut in res_muts:
            resp_mut = cls.trim_response(
                res_mut['question'],
                res_mut['response'],
            )
            resps['mutation'].append({
                'question': res_mut['question'],
                'response': resp_mut,
                'answer': res_mut['answer']
            })
        # end for
        return resps

    @classmethod
    def write_trimmed_response(
        cls, 
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python'
    ) -> None:
        res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        eval_dir = res_dir / 'evaluate' / model_name
        eval_files = cls.read_eval_files(eval_dir)

        results = dict()
        for eval_file in eval_files:
            cksum_val = re.search(
                r'eval\-(.*)\.json', str(eval_file)
            ).group(1)
            eval_res = cls.read_eval_results(eval_file)
            results[cksum_val] = cls.get_trimmed_response(eval_res)
        # end for

        # write results
        Utils.write_json(
            results, 
            eval_dir / 'eval-agg-results.json',
            pretty_format=True
        )
        return

    # @classmethod
    # def get_accuracy(
    #     cls, 
    #     response: str, 
    #     answer: str
    # ) -> float:
    #     pass


class AnalyzeGPTResponse:

    @classmethod
    def read_eval_results(cls, eval_file: Path) -> Dict:
        return Utils.read_json(eval_file)

    @classmethod
    def print_cot_responses(cls, eval_file: Path) -> None:
        eval_res = cls.read_eval_results(eval_file)
        orig_cot_resp = eval_res['orig']['cot_response']
        orig_code_resp = eval_res['orig']['code_response']

        print(f"FILE: {eval_file}")
        print(f"ORIG_RESP_COT:\n{orig_cot_resp.strip()}\n-----")
        print(f"ORIG_RESP_CODE:\n{orig_code_resp.strip()}\n==========")
        for m_i, mut in enumerate(eval_res['mutation']):
            mut_cot_resp = mut['cot_response']
            mut_code_resp = mut['code_response']
            print(f"{m_i}::MUT_RESP_COT:\n{mut_cot_resp.strip()}\n-----")
            print(f"{m_i}::MUT_RESP_CODE:\n{mut_code_resp.strip()}\n==========")
        # end for
        print('####################')
        return

    @classmethod
    def analyze_demonstration_effect(
        cls, 
        model_name: str, 
        dataset_name: str,
        pl_type: str
    ) -> None:
        res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        if dataset_name!='svamp':
            res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        # end if
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        llm_response_file = eval_dir / 'acc-res.json'
        resp_res = Utils.read_json(llm_response_file)
        f2p_list = list()
        p2f_list = list()
        for cksum_val in resp_res.keys():
            if resp_res[cksum_val]['num_demo_used']>0:
                orig_res = resp_res[cksum_val]['orig']
                demo_res = resp_res[cksum_val]['demo']
                if orig_res==0. and demo_res==1.: # fail to pass
                    f2p_list.append(cksum_val)
                elif orig_res==1. and demo_res==0.: # pass to fail
                    p2f_list.append(cksum_val)
                # end if
            # end if
        # end for
        print("analyze_demonstration_effect")
        print(f"===== DATASET:{dataset_name} =====")
        print(f"ORIG_FAIL->DEMO_PASS::\nNUM:{len(f2p_list)}\nLIST:{f2p_list}\n")
        print(f"ORIG_PASS->DEMO_FAIL::\nNUM:{len(p2f_list)}\nLIST:{p2f_list}")
        print(f"==========")
        return

    @classmethod
    def analyze_correctness_of_consistent_examples(
        cls, 
        model_name: str, 
        dataset_name: str,
        pl_type: str
    ):
        # res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        # if dataset_name!='svamp':
        #     res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        # # end if
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        cons_res_file = eval_dir / 'consistency-results.json'
        llm_response_file = eval_dir / 'acc-res.json'

        cons_res = Utils.read_json(cons_res_file)
        resp_res = Utils.read_json(llm_response_file)
        orig_cons_corr_res = list()
        mut_cons_corr_res = list()
        orig_cons_incorr_res = list()
        mut_cons_incorr_res = list()
        num_orig_data = 0
        num_mut_data = 0
        for cksum_val in cons_res.keys():
            if resp_res[cksum_val]['num_demo_used']>0:
                orig_res = cons_res[cksum_val]['orig']
                orig_corr = cons_res[cksum_val]['orig']['correctness']['cot'][-1]
                orig_cons = cons_res[cksum_val]['orig']['consistency']
                num_orig_data += 1
                if orig_corr and orig_cons:
                    orig_cons_corr_res.append(cksum_val)
                elif (not orig_corr) and orig_cons:
                    orig_cons_incorr_res.append(cksum_val)
                # end if

                mut_res = cons_res[cksum_val]['mutation']
                for m_i, m in enumerate(mut_res):
                    num_mut_data += 1
                    mut_corr = m['correctness']['cot'][-1]
                    mut_cons = m['consistency']
                    if mut_corr and mut_cons:
                        mut_cons_corr_res.append(f"{cksum_val}-{m_i}")
                    elif (not mut_corr) and mut_cons:
                        mut_cons_incorr_res.append(f"{cksum_val}-{m_i}")
                    # end if
                # end for
            # end if
        # end for
        print("analyze_correctness_of_consistent_examples")
        print(f"===== DATASET:{dataset_name} =====")
        print(f"ORIG::CONSISTENT_N_CORRECT::NUM:{len(orig_cons_corr_res)} out of {num_orig_data}")
        print(f"ORIG::CONSISTENT_N_INCORRECT::NUM:{len(orig_cons_incorr_res)} out of {num_orig_data}")        
        print(f"MUT::CONSISTENT_N_CORRECT::NUM:{len(mut_cons_corr_res)} out of {num_mut_data}")
        print(f"MUT::CONSISTENT_N_INCORRECT::NUM:{len(mut_cons_incorr_res)} out of {num_mut_data}")
        print(f"==========")
        return

    @classmethod
    def context_sim_between_tgt_n_mut(
        cls,
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python'
    ) -> None:
        def get_vec_dist( 
            vec_a: np.array, 
            vec_b: np.array
        ):
            return round(norm(vec_a-vec_b), 5)
        # end def
        res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        if dataset_name!='svamp':
            res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        # end if
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        acc_res_file = eval_dir / 'acc-res.json'
        acc_res = Utils.read_json(acc_res_file)

        emb_model_name = 'all-MiniLM-L6-v2'
        emb_model = SentenceTransformer(emb_model_name)
        res_dict = dict()
        avg_sim_over_orig_correctness = dict()
        sim_over_orig_correct_tgt_n_mut = list()
        sim_over_orig_correct_mut_n_mut = list()
        sim_over_orig_incorrect_tgt_n_mut = list()
        sim_over_orig_incorrect_mut_n_mut = list()
        for cksum_val in acc_res.keys():
            eval_res_file = eval_dir / f"eval-{cksum_val}.json"
            eval_res = Utils.read_json(eval_res_file)

            orig_correctness = acc_res[cksum_val]['orig']
            demo_correctness = acc_res[cksum_val]['demo']

            tgt_resp = eval_res['orig']['cot_response']
            mut_resps = [
                m['cot_response'] for m in eval_res['mutation']
            ]

            cot_resps = [tgt_resp]+mut_resps
            embeddings = emb_model.encode(cot_resps) # embeddings[s_i,:]
            tgt_embeddings = embeddings[0,:]
            mut_embeddings = embeddings[1:,:]
            sim_list_from_tgt = list()
            sim_list_over_muts = list()
            for m_i in range(len(mut_resps)):
                m_emb = mut_embeddings[m_i,:]
                sim_list_from_tgt.append(
                    get_vec_dist(tgt_embeddings, m_emb)
                )
                d_list = list()
                for _m_i in range(len(mut_resps)):
                    if _m_i!=m_i:
                        _m_emb = mut_embeddings[_m_i,:]
                        d_list.append(
                            get_vec_dist(m_emb, _m_emb)
                        )
                    # end if
                # end for
                sim_list_over_muts.append(
                    Utils.avg(d_list, decimal=5),
                )
            # end for
            res_dict[cksum_val] = {
                'orig_correctness': orig_correctness,
                'demo_correctness': demo_correctness,
                'count': len(sim_list_from_tgt),
                'avg_between_tgt_n_mut': Utils.avg(sim_list_from_tgt, decimal=3),
                'avg_between_mut_n_mut': Utils.avg(sim_list_over_muts, decimal=3),
                'raw_between_tgt_n_mut': sim_list_from_tgt,
                'raw_between_mut_n_mut': sim_list_over_muts
            }
            if orig_correctness:
                sim_over_orig_correct_tgt_n_mut.append(res_dict[cksum_val]['avg_between_tgt_n_mut'])
                sim_over_orig_correct_mut_n_mut.append(res_dict[cksum_val]['avg_between_mut_n_mut'])
            else:
                sim_over_orig_incorrect_tgt_n_mut.append(res_dict[cksum_val]['avg_between_tgt_n_mut'])
                sim_over_orig_incorrect_mut_n_mut.append(res_dict[cksum_val]['avg_between_mut_n_mut'])
            # end if
        # end for
        avg_sim_over_orig_correctness['correct'] = {
            'count': len(sim_over_orig_correct_tgt_n_mut),
            'avg_tgt_n_mut': Utils.avg(sim_over_orig_correct_tgt_n_mut, decimal=3),
            'median_tgt_n_mut': Utils.median(sim_over_orig_correct_tgt_n_mut, decimal=3),
            'stdev_tgt_n_mut': Utils.stdev(sim_over_orig_correct_tgt_n_mut, decimal=3),
            'min_tgt_n_mut': min(sim_over_orig_correct_tgt_n_mut),
            'max_tgt_n_mut': max(sim_over_orig_correct_tgt_n_mut),
            'avg_mut_n_mut': Utils.avg(sim_over_orig_correct_mut_n_mut, decimal=3),
            'median_mut_n_mut': Utils.median(sim_over_orig_correct_mut_n_mut, decimal=3),
            'stdev_mut_n_mut': Utils.stdev(sim_over_orig_correct_mut_n_mut, decimal=3),
            'min_mut_n_mut': min(sim_over_orig_correct_mut_n_mut),
            'max_mut_n_mut': max(sim_over_orig_correct_mut_n_mut)
        }
        avg_sim_over_orig_correctness['incorrect'] = {
            'count': len(sim_over_orig_incorrect_tgt_n_mut),
            'avg_tgt_n_mut': Utils.avg(sim_over_orig_incorrect_tgt_n_mut, decimal=3),
            'median_tgt_n_mut': Utils.median(sim_over_orig_incorrect_tgt_n_mut, decimal=3),
            'stdev_tgt_n_mut': Utils.stdev(sim_over_orig_incorrect_tgt_n_mut, decimal=3),
            'min_tgt_n_mut': min(sim_over_orig_incorrect_tgt_n_mut),
            'max_tgt_n_mut': max(sim_over_orig_incorrect_tgt_n_mut),
            'avg_mut_n_mut': Utils.avg(sim_over_orig_incorrect_mut_n_mut, decimal=3),
            'median_mut_n_mut': Utils.median(sim_over_orig_incorrect_mut_n_mut, decimal=3),
            'stdev_mut_n_mut': Utils.stdev(sim_over_orig_incorrect_mut_n_mut, decimal=3),
            'min_mut_n_mut': min(sim_over_orig_incorrect_mut_n_mut),
            'max_mut_n_mut': max(sim_over_orig_incorrect_mut_n_mut)
        }
        Utils.write_json(
            res_dict,
            eval_dir / f"cot-response-sim-between-tgt-n-mut-over-cksumvals.json",
            pretty_format=True
        )
        Utils.write_json(
            avg_sim_over_orig_correctness, 
            eval_dir / f"cot-response-avg-sim-over-orig-correctness.json",
            pretty_format=True
        )
        return

    @classmethod
    def analyze_inclusion_among_selection_methods(
        cls,
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python'
    ) -> None:
        # This analyze how the performance improvement comes from.
        # For example, original pass cases are all included in
        # the pass cases with demo?
        emb_consist_type_n_res_file_map = {
            'random': 'acc-res-random.json',
            'modcons-random': 'acc-res-modcons-random.json',
            'cos_sim': 'acc-res-cos_sim.json',
            'modcons-cos_sim': 'acc-res-modcons-cos_sim.json',
            'dist': 'acc-res-dist.json',
            'modcons-dist': 'acc-res-modcons-dist.json',
            'avg_dist_among_muts': 'acc-res-avg_dist_among_muts.json',
            'modcons-avg_dist_among_muts': 'acc-res-modcons-avg_dist_among_muts.json'
        }
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        
        res_dict = dict()
        for tgt_emb_consist_type in emb_consist_type_n_res_file_map.keys():
            other_emb_consist_types = [
                t for t in emb_consist_type_n_res_file_map.keys()
                if t!=tgt_emb_consist_type
            ]
            tgt_emb_consist_file = emb_consist_type_n_res_file_map[tgt_emb_consist_type]
            tgt_emb_res = Utils.read_json(eval_dir / tgt_emb_consist_file)

            res_dict[f"tgt::{tgt_emb_consist_type}"] = dict()
            for _emb_consist_type in other_emb_consist_types:

                _emb_consist_file = emb_consist_type_n_res_file_map[_emb_consist_type]
                _emb_res = Utils.read_json(eval_dir / _emb_consist_file)
                res_dict[f"tgt::{tgt_emb_consist_type}"][f"query::{_emb_consist_type}"] = dict()
                same_improv_list = list()
                tgt_only_improv_list = list()
                query_only_improv_list = list()
                tgt_only_collapse_list = list()
                query_only_collapse_list = list()
                for tgt_cksum_val in tgt_emb_res.keys():
                    if tgt_emb_res[tgt_cksum_val]['orig']==0. and \
                        tgt_emb_res[tgt_cksum_val]['demo']==1. and \
                        _emb_res[tgt_cksum_val]['orig']==0. and \
                        _emb_res[tgt_cksum_val]['demo']==1.:
                        same_improv_list.append(tgt_cksum_val)
                    elif tgt_emb_res[tgt_cksum_val]['orig']==0. and \
                        tgt_emb_res[tgt_cksum_val]['demo']==1. and \
                        _emb_res[tgt_cksum_val]['orig']==0. and \
                        _emb_res[tgt_cksum_val]['demo']==0.:
                        tgt_only_improv_list.append(tgt_cksum_val)
                    elif tgt_emb_res[tgt_cksum_val]['orig']==0. and \
                        tgt_emb_res[tgt_cksum_val]['demo']==0. and \
                        _emb_res[tgt_cksum_val]['orig']==0. and \
                        _emb_res[tgt_cksum_val]['demo']==1.:
                        query_only_improv_list.append(tgt_cksum_val)
                    elif tgt_emb_res[tgt_cksum_val]['orig']==1. and \
                        tgt_emb_res[tgt_cksum_val]['demo']==0. and \
                        _emb_res[tgt_cksum_val]['orig']==1. and \
                        _emb_res[tgt_cksum_val]['demo']==1.:
                        tgt_only_collapse_list.append(tgt_cksum_val)
                    elif tgt_emb_res[tgt_cksum_val]['orig']==1. and \
                        tgt_emb_res[tgt_cksum_val]['demo']==1. and \
                        _emb_res[tgt_cksum_val]['orig']==1. and \
                        _emb_res[tgt_cksum_val]['demo']==0.:
                        query_only_collapse_list.append(tgt_cksum_val)
                    # end if
                # end for
                res_dict[f"tgt::{tgt_emb_consist_type}"][f"query::{_emb_consist_type}"] = {
                    'all_improv': same_improv_list,
                    'tgt_only_improv': tgt_only_improv_list,
                    'query_only_improv': query_only_improv_list,
                    'tgt_only_collapse': tgt_only_collapse_list,
                    'query_only_collapse': query_only_collapse_list,
                    '#all_improv': len(same_improv_list),
                    '#tgt_only_improv': len(tgt_only_improv_list),
                    '#query_only_improv': len(query_only_improv_list),
                    '#tgt_only_collapse': len(tgt_only_collapse_list),
                    '#query_only_collapse': len(query_only_collapse_list),
                }
            # end for
        # end for
        Utils.write_json(
            res_dict,
            eval_dir / 'improvement-inclusion-over-types.json',
            pretty_format=True
        )
        return

    @classmethod
    def analyze_modconst_n_correctness(
        cls,
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python'
    ):
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name / 'evaluate_consistency' / model_name
        modcons_res_file = res_dir / 'modal-consistency-results.json'
        modcons_res = Utils.read_json(modcons_res_file)

        orig_modcons_n_correct_examples = list()
        orig_non_modcons_n_correct_examples = list()
        orig_all_correct_examples = list()

        orig_modcons_n_incorrect_examples = list()
        orig_non_modcons_n_incorrect_examples = list()
        orig_all_incorrect_examples = list()


        for cksum_val in modcons_res.keys():
            orig_modcons = modcons_res[cksum_val]['orig']['consistency']
            orig_cot_correctness =  modcons_res[cksum_val]['orig']['correctness']['cot'][1]
            orig_code_correctness =  modcons_res[cksum_val]['orig']['correctness']['code'][1]
            if orig_cot_correctness:
                orig_all_correct_examples.append(cksum_val)
                if orig_modcons:
                    orig_modcons_n_correct_examples.append(cksum_val)
                elif not orig_modcons:
                    orig_non_modcons_n_correct_examples.append(cksum_val)
                # end if
            else:
                orig_all_incorrect_examples.append(cksum_val)
                if orig_modcons:
                    orig_modcons_n_incorrect_examples.append(cksum_val)
                elif not orig_modcons:
                    orig_non_modcons_n_incorrect_examples.append(cksum_val)
                # end if
            # end if
        # end for
        print(f"===== DATASET: {dataset_name} =====")
        print(f"#CORRECT_EXAMPLES: {len(orig_all_correct_examples)}")
        print(f"#CORRECT_N_MODCONS_EXAMPLES: {len(orig_modcons_n_correct_examples)}")
        print(f"#CORRECT_N_NOT_MODCONS_EXAMPLES: {len(orig_non_modcons_n_correct_examples)}")
        print(f"\n#INCORRECT_EXAMPLES: {len(orig_all_incorrect_examples)}")
        print(f"#INCORRECT_N_MODCONS_EXAMPLES: {len(orig_modcons_n_incorrect_examples)}")
        print(f"#INCORRECT_N_NOT_MODCONS_EXAMPLES: {len(orig_non_modcons_n_incorrect_examples)}\n")
        return

    @classmethod
    def main(
        cls,
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python'
    ) -> None:
        # res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        # eval_dir = res_dir / 'evaluate_consistency' / model_name
        # llm_response_files = [
        #     f for f in os.listdir(str(eval_dir))
        #     if f.endswith('.json') and f.startswith('eval-') and (not f.startswith('eval-results-w-demo'))
        # ]
        # for resp_file in llm_response_files[:5]:
        #     cls.print_cot_responses(eval_dir / resp_file)
        # # end for

        # cls.analyze_demonstration_effect(
        #     model_name,
        #     dataset_name,
        #     pl_type=pl_type
        # )
        # cls.analyze_correctness_of_consistent_examples(
        #     model_name, 
        #     dataset_name,
        #     pl_type=pl_type
        # )
        # cls.context_sim_between_tgt_n_mut(
        #     model_name, 
        #     dataset_name,
        #     pl_type=pl_type
        # )
        # cls.analyze_inclusion_among_selection_methods(
        #     model_name,
        #     dataset_name,
        #     pl_type=pl_type
        # )
        cls.analyze_modconst_n_correctness(
            model_name,
            dataset_name,
            pl_type=pl_type
        )
        return

