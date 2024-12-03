
import os
import openai
import numpy as np

from typing import *
from pathlib import Path

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger

from ..llmut.openai import OpenAiModel


class Multimodals:

    prompts_over_modals = Macros.prompts_over_modals

    @classmethod
    def get_openai_model_response(
        cls,
        model_name: str,
        input_text: str,
        prompt: str='',
        prompt_append: bool=False
    ) -> str:
        return OpenAiModel.predict(
            input_text,
            prompt,
            prompt_append=prompt_append,
            model_name=model_name
        )
    
    @classmethod
    def get_responses_over_modals(
        cls, 
        eval_res: Dict,
        modal_names: List[str],
        llm_name: str
    ) -> Dict:
        q_orig = eval_res['orig']['question']
        for m_name in modal_names:
            if m_name.endswith('_response') and \
                m_name=='eqn_response':
                # m_name not in eval_res['orig'].keys():
                eval_res['orig'][m_name] = cls.get_openai_model_response(
                    llm_name,
                    q_orig,
                    prompt=cls.prompts_over_modals[m_name][0],
                    prompt_append=cls.prompts_over_modals[m_name][1]
                )
                # print(eval_res['orig'][m_name])
            # end if
        # end for

        mut_resps = list()
        for mut in eval_res['mutation']:
            q_mut = mut['question']
            for m_name in modal_names:
                if m_name.endswith('_response') and \
                    m_name=='eqn_response':
                    # m_name not in mut.keys():
                    mut[m_name] = cls.get_openai_model_response(
                        llm_name,
                        q_mut,
                        prompt=cls.prompts_over_modals[m_name][0],
                        prompt_append=cls.prompts_over_modals[m_name][1]
                    )
                    # print(mut[m_name])
                # end if
            # end for
            mut_resps.append(mut)
        # end for
        eval_res['mutation'] = mut_resps
        return eval_res

    @classmethod
    def main(
        cls,
        dataset_name: str,
        llm_name: str
    ):
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        eval_res_files = sorted([
            f_name for f_name in os.listdir(str(eval_dir))
            if f_name.endswith('.json') and f_name.startswith('eval-') and \
            (not f_name.startswith('eval-results-w'))
        ])
        modal_names = list(cls.prompts_over_modals.keys())
        model_name = Macros.gpt3d5_engine_name if llm_name=='gpt3.5' else Macros.gpt4_engine_name
        f_index = 0
        for eval_res_file in eval_res_files:
            cksum_val = eval_res_file.split('-')[-1].split('.json')[0].strip()
            print(f"{f_index} OUT OF {len(eval_res_files)}::{cksum_val}")
            _eval_res = Utils.read_json(eval_dir / eval_res_file)
            eval_res = cls.get_responses_over_modals(
                _eval_res,
                modal_names,
                model_name
            )
            Utils.write_json(
                eval_res, 
                eval_dir / eval_res_file,
                pretty_format=True
            )
            f_index += 1
        # end for
        return
