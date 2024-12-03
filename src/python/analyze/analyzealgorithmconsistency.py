
import re, os
import sys
import csv
import time
import random
import numpy as np

from typing import *
from pathlib import Path

from ..utils.macros import Macros
from ..utils.utils import Utils


class AnalyzeAlgorithmConsistency:

    @classmethod
    def get_matrix_stats(cls, alg_const_dict: Dict):
        # stat feature
        # 1.num_codes_used
        # 2.#0s in matrix
        # 3.#1s in matrix
        # 4.#codes that contain 1. at least once
        # 5.#codes that contain 1. dominantly
        num_zeros = 0
        num_ones = 0
        num_codes_used = len(alg_const_dict.keys())
        num_codes_with_one_at_least_once = 0
        num_codes_with_one_mt_half = 0
        num_ones_over_codes = {
            key: 0
            for key in alg_const_dict.keys()
        }
        for code_used_key in alg_const_dict.keys():
            for val_used_key in alg_const_dict[code_used_key].keys():
                if alg_const_dict[code_used_key][val_used_key]==0.:
                    num_zeros += 1
                elif alg_const_dict[code_used_key][val_used_key]==1.:
                    num_ones += 1
                    num_ones_over_codes[code_used_key] += 1
                # end if
            # end for
        # end for
        for code_used_key in num_ones_over_codes.keys():
            if num_ones_over_codes[code_used_key]>0:
                num_codes_with_one_at_least_once += 1
                if num_ones_over_codes[code_used_key]>num_codes_used//2:
                    num_codes_with_one_mt_half += 1
                # end if
            # end if
        # end if
        return {
            'num_codes_used': num_codes_used,
            'num_zeros': num_zeros,
            'num_ones': num_ones,
            'num_codes_with_one_at_least_once': num_codes_with_one_at_least_once,
            'num_codes_with_one_mt_half': num_codes_with_one_mt_half
        }

    @classmethod
    def generate_matrix_csv(
        cls, 
        alg_const_file: Path, 
        cksum_val: str
    ):
        alg_const_res: Dict = Utils.read_json(alg_const_file)
        res_mat = alg_const_res['res_matrix']
        csv_file = alg_const_file.parent / f"alg-consistency-mat-{cksum_val}.csv"
        mat_stat_file = alg_const_file.parent / f"alg-consistency-mat-{cksum_val}-stats.json"
        with open(csv_file, 'w') as csvfile:
            dict_writer = csv.writer(csvfile)            
            mut_headers = [
                n for n in res_mat['CODE_USED::TGT'].keys()
                if n!='VALUE_USED::TGT'
            ]
            headers = ['N/A', 'VALUE_USED::TGT']+mut_headers
            dict_writer.writerow(headers)
            for key in headers[1:]:
                key_row = 'CODE_USED::'+key.split('VALUE_USED::')[-1]
                d_row = [key_row] + [
                    res_mat[key_row][h] for h in headers[1:]
                ]
                dict_writer.writerow(d_row)
            # end for
        # end with
        alg_const_res_stat = cls.get_matrix_stats(res_mat)
        Utils.write_json(
            alg_const_res_stat,
            mat_stat_file,
            pretty_format=True
        )
        return

    @classmethod
    def generate_matrix_csv_over_all_alg_consts(
        cls,
        dataset_name: str,
        model_name: str
    ):
        res_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency'/ model_name / 'alg_consistency'
        alg_const_files = [
            f for f in os.listdir(res_dir)
            if f.startswith('alg-consistency-mat') and \
                f.endswith('.json') and \
                not f.endswith('-stats.json')
        ]
        for alg_const_file in alg_const_files:
            cksum_val = re.search(
                r'alg\-consistency\-mat\-(.*)\.json',
                alg_const_file
            ).group(1)
            print(cksum_val)
            cls.generate_matrix_csv(
                res_dir / alg_const_file,
                cksum_val
            )
        # end for
        return

    @classmethod
    def get_list_of_res_given_type(
        cls, 
        res_data: Dict,
        bl_res_data: Dict,
        _type: str
    ):
        left_res_cksum_vals = set()
        right_res_cksum_vals = set()

        for cksum_val in res_data.keys():
            if _type=='orig:pass->demo:pass':
                if res_data[cksum_val]['orig']==1. and \
                    res_data[cksum_val]['demo']==1.:
                    left_res_cksum_vals.add(cksum_val)
                # end if
            elif _type=='orig:pass->demo:fail':
                if res_data[cksum_val]['orig']==1. and \
                    res_data[cksum_val]['demo']==0.:
                    left_res_cksum_vals.add(cksum_val)
                # end if
            elif _type=='orig:fail->demo:pass':
                if res_data[cksum_val]['orig']==0. and \
                    res_data[cksum_val]['demo']==1.:
                    left_res_cksum_vals.add(cksum_val)
                # end if
            else:
                if res_data[cksum_val]['orig']==0. and \
                    res_data[cksum_val]['demo']==0.:
                    left_res_cksum_vals.add(cksum_val)
                # end if
            # end if
        # end for

        for cksum_val in bl_res_data.keys():
            if _type=='orig:pass->demo:pass':
                if bl_res_data[cksum_val]['orig']==1. and \
                    bl_res_data[cksum_val]['demo']==1.:
                    right_res_cksum_vals.add(cksum_val)
                # end if
            elif _type=='orig:pass->demo:fail':
                if bl_res_data[cksum_val]['orig']==1. and \
                    bl_res_data[cksum_val]['demo']==0.:
                    right_res_cksum_vals.add(cksum_val)
                # end if
            elif _type=='orig:fail->demo:pass':
                if bl_res_data[cksum_val]['orig']==0. and \
                    bl_res_data[cksum_val]['demo']==1.:
                    right_res_cksum_vals.add(cksum_val)
                # end if
            else:
                if bl_res_data[cksum_val]['orig']==0. and \
                    bl_res_data[cksum_val]['demo']==0.:
                    right_res_cksum_vals.add(cksum_val)
                # end if
            # end if
        # end for

        both_res_cksum_vals = sorted(list(left_res_cksum_vals.intersection(right_res_cksum_vals)))
        left_only_res_cksum_vals = sorted(list(left_res_cksum_vals.difference(right_res_cksum_vals)))
        right_only_res_cksum_vals = sorted(list(right_res_cksum_vals.difference(left_res_cksum_vals)))

        return both_res_cksum_vals,\
            left_only_res_cksum_vals,\
            right_only_res_cksum_vals

    @classmethod
    def get_overlap_with_baselines(
        cls,
        dataset_name: str,
        model_name: str,
        include_only_modconst: bool=False
    ) -> None:
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

        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency'/ model_name
        alg_const_dir = eval_dir / 'alg_consistency'
        if include_only_modconst:
            alg_const_dir = eval_dir / 'alg_consistency_only_modconst'
        # end if
        alg_const_acc_res_file = alg_const_dir / 'acc-res.json'
        alg_const_acc_res = Utils.read_json(alg_const_acc_res_file)

        res_dict = {
            'orig:pass->demo:pass': dict(),
            'orig:pass->demo:fail': dict(),
            'orig:fail->demo:pass': dict(),
            'orig:fail->demo:fail': dict()
        }

        for key in emb_consist_type_n_res_file_map.keys():
            bl_acc_res = Utils.read_json(
                eval_dir / emb_consist_type_n_res_file_map[key]
            )

            for _type in res_dict.keys():
                both_res_cksum_vals,\
                left_only_res_cksum_vals,\
                right_only_res_cksum_vals = cls.get_list_of_res_given_type(
                    alg_const_acc_res,
                    bl_acc_res,
                    _type
                )

                res_dict[_type].setdefault(key, {
                    'both': len(both_res_cksum_vals),
                    'alg-const-only': len(left_only_res_cksum_vals),
                    f"{key}-only": len(right_only_res_cksum_vals)
                })
            # end for
        # end for
        Utils.write_json(
            res_dict,
            alg_const_dir / f"res-comparison-of-alg-const-w-bls.json",
            pretty_format=True
        )
        return

    @classmethod
    def get_overlap_btw_with_n_without_modconst(
        cls,
        dataset_name: str,
        model_name: str,
    ) -> None:
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency'/ model_name
        alg_const_wo_modconst_dir = eval_dir / 'alg_consistency'
        alg_const_w_modconst_dir = eval_dir / 'alg_consistency_only_modconst'

        alg_const_wo_modconst_acc_res_file = alg_const_wo_modconst_dir / 'acc-res.json'
        alg_const_w_modconst_acc_res_file = alg_const_w_modconst_dir / 'acc-res.json'

        alg_const_wo_modconst_acc_res = Utils.read_json(alg_const_wo_modconst_acc_res_file)
        alg_const_w_modconst_acc_res = Utils.read_json(alg_const_w_modconst_acc_res_file)

        res_dict = {
            'orig:pass->demo:pass': dict(),
            'orig:pass->demo:fail': dict(),
            'orig:fail->demo:pass': dict(),
            'orig:fail->demo:fail': dict()
        }

        for _type in res_dict.keys():
            both_res_cksum_vals,\
            left_only_res_cksum_vals,\
            right_only_res_cksum_vals = cls.get_list_of_res_given_type(
                alg_const_wo_modconst_acc_res,
                alg_const_w_modconst_acc_res,
                _type
            )
            res_dict[_type] = {
                'both': len(both_res_cksum_vals),
                'alg-const-wo-modconst': len(left_only_res_cksum_vals),
                'alg-const-w-modconst': len(right_only_res_cksum_vals)
            }
        # end for
        Utils.write_json(
            res_dict,
            eval_dir / f"res-comparison-of-alg-const-w-n-wo-modconst.json",
            pretty_format=True
        )
        return

# class AnalyzeAlgorithmConsistencyUsingCode:


