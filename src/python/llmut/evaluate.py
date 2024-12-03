

import os
import re
import torch
import random
import subprocess
import numpy as np
import torch.backends.cudnn as cudnn

from typing import *
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# from .llama import Llama
from .llama_model import LlamaModel
from .alpaca import Alpaca
from .openai import OpenAiModel
from .palm import PaLMModel

# from ..consistency.algconsistency import AlgorithmConsistency
from ..discriminator.models.transformer_disc import load_pretrained_model, get_correctness_score
from ..utils.macros import Macros
from ..utils.utils import Utils

random.seed(Macros.RAND_SEED)


class Evaluate:

    model_name = None

    @classmethod
    def load_model(
        cls, 
        model_name:str = None
    ):
        model = None
        tokenizer = None
        if model_name is not None:
            cls.model_name = model_name
        # end if
        if cls.model_name=='llama':
            model = LlamaModel.load_model(
                ckpt_dir = Macros.llama_model_dir,
                tokenizer_path = Macros.llama_dir
            )
        elif cls.model_name=='alpaca':
            model, tokenizer = Alpaca.load_model()
        elif cls.model_name=='gpt3.5':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt3d5_engine_name
            )
        elif cls.model_name=='gpt4':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4_engine_name
            )
        elif cls.model_name=='gpt4omini':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4omini_engine_name
            )
        # end if
        return model, tokenizer

    @classmethod
    def get_response(
        cls, 
        model: AutoModel,
        tokenizer: AutoTokenizer,
        input_text: str,
        prompt: str,
        prompt_append: bool=False
    ) -> List[str]:
        response = None
        if cls.model_name=='llama':
            response = LlamaModel.predict(
                model, 
                input_text, 
                prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='alpaca':
            response = Alpaca.predict(model, tokenizer, input_text)
        elif cls.model_name=='gpt3.5':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4omini':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        # end if
        return response

    @classmethod
    def eval(
        cls, 
        model: AutoModel,
        tokenizer: AutoTokenizer,
        input_texts: List[str],
        include_exp: bool
    ) -> List[Dict]:
        prompt = Macros.openai_prompt
        if include_exp:
            prompt = Macros.openai_prompt_w_exp
        # end if
        result = [
            cls.get_response(
                model, 
                tokenizer, 
                inp, 
                prompt
            ) for inp in input_texts
        ]
        return result

    @classmethod
    def main(
        cls, 
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python',
        include_exp: bool=False
    ) -> None:
        cls.model_name = model_name
        res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        mut_dir = res_dir / 'mutation'
        eval_dir = res_dir / 'evaluate' / model_name
        if include_exp:
            eval_dir = res_dir / 'evaluate' / f"{model_name}_w_exp"
        # end if
        eval_dir.mkdir(parents=True, exist_ok=True)
        mut_files = [
            f for f in os.listdir(str(mut_dir))
            if f.endswith('.json') and f.startswith('mut-nl')
        ]
        model, tokenizer = cls.load_model()
        # mut_files_of_interest = mut_files[len(mut_files)//10:]
        mut_files_of_interest = sorted(mut_files)
        print(f"MODEL: {model_name}")
        print(f"EVAL_DIR: {eval_dir}")
        prompt = Macros.openai_prompt_w_exp if include_exp else Macros.openai_prompt
        print(f"PROMPT: {prompt}")
        for mut_file in mut_files_of_interest:
            print(mut_file)
            cksum_val = re.search(
                r'mut\-nl\-(.*)\.json', mut_file
            ).group(1)
            '''
            q_dict = {
                'orig': {
                    'question': orig_nl_q,
                    'answer': orig_nl_ans
                },
                'mutations': mut_nls
            }
            '''
            q_dict = Utils.read_json(mut_dir / mut_file)

            # original question
            orig_resp = cls.eval(
                model,
                tokenizer,
                [q_dict['orig']['question']],
                include_exp
            )

            # mutated questions
            mut_qs = {
                'question': [
                    m_q['question'] for m_q in q_dict['mutations']
                ],
                'answer': [
                    m_q['answer'] for m_q in q_dict['mutations']
                ]
            }
            mut_resps = cls.eval(
                model,
                tokenizer,
                mut_qs['question'],
                include_exp
            )

            eval_result = {
                'orig': {
                    'question': q_dict['orig']['question'],
                    'response': orig_resp[0],
                    'answer': q_dict['orig']['answer']
                },
                'mutation': [
                    {
                        'question': m_q['question'],
                        'response': mut_resps[m_q_i],
                        'answer': m_q['answer']
                    } 
                    for m_q_i, m_q in enumerate(q_dict['mutations'])
                ]
            }
            Utils.write_json(
                eval_result, 
                eval_dir / f"eval-{cksum_val}.json",
                pretty_format=True
            )
        # end for
        return
        

class EvaluateWithMultimodals:

    model_name = None

    @classmethod
    def load_model(cls):
        model = None
        tokenizer = None
        if cls.model_name=='llama':
            model = LlamaModel.load_model(
                ckpt_dir = Macros.llama_model_dir,
                tokenizer_path = Macros.llama_model_root_dir
            )
        elif cls.model_name=='alpaca':
            model, tokenizer = Alpaca.load_model()
        elif cls.model_name=='gpt3.5':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt3d5_engine_name
            )
        elif cls.model_name=='gpt4':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4_engine_name
            )
        elif cls.model_name=='gpt4omini':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4omini_engine_name
            )
        elif cls.model_name=='palm':
            PaLMModel.set_model_name(
                engine_name=Macros.palm_engine_name
            )
        # end if
        return model, tokenizer

    @classmethod
    def get_response(
        cls, 
        model: AutoModel,
        tokenizer: AutoTokenizer,
        input_text: str,
        prompt: str,
        prompt_append: bool
    ) -> List[str]:
        response = None
        response_logprobs = None
        if cls.model_name=='llama':
            response = LlamaModel.predict(
                model, 
                input_text,
                prompt,
                prompt_append=prompt_append
            )
        elif cls.model_name=='alpaca':
            response = Alpaca.predict(model, tokenizer, input_text)
        elif cls.model_name=='gpt3.5':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4omini':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='palm':
            response = PaLMModel.predict(
                input_text,
                prompt=prompt,
                prompt_append=prompt_append
            )
        # end if
        return response

    # @classmethod
    # def execute_code(cls, code_str: str, pl_type='python'):
    #     temp_pl_file = None
    #     if pl_type=='python':
    #         # execute the llm_temp.py
    #         code_str = f"import math\nimport datetime\n{code_str}\nprint(func())\n"
    #         temp_pl_file = Macros.result_dir / 'llm_temp.py'
    #         Utils.write_txt(code_str, temp_pl_file)
    #         cmd = f"python {str(temp_pl_file)}"
    #     # end if
    #     try:
    #         output = subprocess.check_output(cmd, shell=True).strip()
    #         os.remove(str(temp_pl_file))
    #         return output.decode()
    #     except subprocess.CalledProcessError:
    #         print(f"ERROR_CODE:\n{code_str}")
    #         os.remove(str(temp_pl_file))
    #         return None
    #     # end try

    @classmethod
    def eval(
        cls, 
        model: AutoModel,
        tokenizer: AutoTokenizer,
        input_texts: List[str]
    ) -> List[Dict]:
        modal_names = list(Macros.prompts_over_modals.keys())
        resp_dict = dict()
        for m_name in modal_names:
            prompt, is_append =  Macros.prompts_over_modals[m_name]
            resps = [
                cls.get_response(
                    model, 
                    tokenizer, 
                    inp, 
                    prompt,
                    prompt_append=is_append,
                ) for inp in input_texts
            ]
            resp_dict[m_name] = resps
        # end for
        return resp_dict

    @classmethod
    def main(
        cls, 
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python'
    ) -> None:
        cls.model_name = model_name
        # res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        mut_dir = res_dir / 'mutation'
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        mut_files = [
            f for f in os.listdir(str(mut_dir))
            if f.endswith('.json') and f.startswith('mut-nl')
        ]
        # print(len(mut_files), len(os.listdir(str(mut_dir))))
        # raise()
        model, tokenizer = cls.load_model()

        # mut_files_of_interest = sorted(mut_files)[:len(mut_files)//10+1]
        mut_files_of_interest = sorted(mut_files)
        print(f"MODEL: {model_name}")
        print(f"EVAL_DIR: {eval_dir}")
        print(f"#DATA_UNDER_TEST: {len(mut_files_of_interest)}")
        for f_i, mut_file in enumerate(mut_files_of_interest):
            cksum_val = re.search(
                r'mut\-nl\-(.*)\.json', mut_file
            ).group(1)
            print(f"EvaluateWithMultimodals::{f_i}_OUT_OF_{len(mut_files_of_interest)}::{cksum_val}")
            # print(f_i, mut_file)
            eval_res_file = eval_dir / f"eval-{cksum_val}.json"
            eval_result = dict()
            if os.path.exists(str(eval_res_file)) and \
                os.path.exists(str(mut_dir / mut_file)):
                eval_result = Utils.read_json(eval_res_file)
                q_dict = Utils.read_json(mut_dir / mut_file)

                if eval_result.get('orig', None) is None:
                    # original question
                    orig_resp_dict = cls.eval(
                        model,
                        tokenizer,
                        [q_dict['orig']['question']]
                    )
                    for c_i, c in enumerate(orig_resp_dict['code_response']):
                        c_res = Utils.execute_code(c['msg'], dataset_name, pl_type=pl_type)
                        orig_resp_dict['code_response'][c_i]['msg'] = f"{c['msg']}\nthe answer is {c_res}"
                    # end for

                    eval_result['orig'] = {
                        'question': q_dict['orig']['question'],
                        'answer': q_dict['orig']['answer']
                    }
                    for m_name in Macros.prompts_over_modals.keys():
                        eval_result['orig'][m_name] = orig_resp_dict[m_name][0]
                    # end for
                # end if

                # mutated questions
                prev_mut_qs = [
                    m['question']
                    for m in eval_result['mutation']
                ]
                mut_qs = {
                    'question': [
                        m_q['question'] 
                        for m_q_i, m_q in enumerate(q_dict['mutations'])
                        if m_q['question'] not in prev_mut_qs
                    ],
                    'answer': [
                        m_q['answer'] 
                        for m_q_i, m_q in enumerate(q_dict['mutations'])
                        if m_q['question'] not in prev_mut_qs
                    ]
                }
                mut_resp_dict = cls.eval(
                    model,
                    tokenizer,
                    mut_qs['question']
                )
                
                for c_i, c in enumerate(mut_resp_dict['code_response']):
                    c_res = Utils.execute_code(c['msg'], dataset_name, pl_type=pl_type)
                    mut_resp_dict['code_response'][c_i]['msg'] = f"{c['msg']}\nthe answer is {c_res}"
                # end for
                
                for m_q_i, m_q in enumerate(mut_qs['question']):
                    if m_q not in prev_mut_qs:
                        eval_result['mutation'].append({
                            'question': m_q,
                            'answer': mut_qs['answer'][m_q_i]
                        })
                        for m_name in Macros.prompts_over_modals.keys():
                            eval_result['mutation'][-1][m_name] = mut_resp_dict[m_name][m_q_i]
                        # end for
                    # end if
                # end for
                Utils.write_json(
                    eval_result, 
                    eval_res_file,
                    pretty_format=True
                )
            elif (not os.path.exists(str(eval_res_file))) and \
                (os.path.exists(str(mut_dir / mut_file))):
                '''
                q_dict = {
                    'orig': {
                        'question': orig_nl_q,
                        'answer': orig_nl_ans
                    },
                    'mutations': mut_nls
                }
                '''
                q_dict = Utils.read_json(mut_dir / mut_file)

                # original question
                orig_resp_dict = cls.eval(
                    model,
                    tokenizer,
                    [q_dict['orig']['question']]
                )
                for c_i, c in enumerate(orig_resp_dict['code_response']):
                    c_res = Utils.execute_code(c['msg'], dataset_name, pl_type=pl_type)
                    orig_resp_dict['code_response'][c_i]['msg'] = f"{c['msg']}\nthe answer is {c_res}"
                # end for

                # mutated questions
                mut_qs = {
                    'question': [
                        m_q['question'] for m_q in q_dict['mutations']
                    ],
                    'answer': [
                        m_q['answer'] for m_q in q_dict['mutations']
                    ]
                }
                mut_resp_dict = cls.eval(
                    model,
                    tokenizer,
                    mut_qs['question']
                )
                for c_i, c in enumerate(mut_resp_dict['code_response']):
                    c_res = Utils.execute_code(c['msg'], dataset_name, pl_type=pl_type)
                    mut_resp_dict['code_response'][c_i]['msg'] = f"{c['msg']}\nthe answer is {c_res}"
                # end for
                eval_result = {
                    'orig': {
                        'question': q_dict['orig']['question'],
                        'answer': q_dict['orig']['answer']
                    },
                    'mutation': list()
                }
                for m_name in Macros.prompts_over_modals.keys():
                    eval_result['orig'][m_name] = orig_resp_dict[m_name][0]
                # end for
            
                for m_q_i, m_q in enumerate(q_dict['mutations']):
                    eval_result['mutation'].append({
                        'question': m_q['question'],
                        'answer': m_q['answer']
                    })
                    for m_name in Macros.prompts_over_modals.keys():
                        eval_result['mutation'][-1][m_name] = mut_resp_dict[m_name][m_q_i]
                    # end for
                # end for
                Utils.write_json(
                    eval_result, 
                    eval_res_file,
                    pretty_format=True
                )
            elif (os.path.exists(str(eval_res_file))) and \
                (not os.path.exists(str(mut_dir / mut_file))):
                os.remove(str(eval_res_file))
            # end if
        # end for
        return
        

class EvaluateWithDemo:

    model_name = None

    # emb_consist_type_n_res_file_map = {
    #     'random': 'eval-results-w-demo-random.json',
    #     'cos_sim': 'eval-results-w-demo-modcons-simwtgtnmut.json',
    #     'dist': 'eval-results-w-demo-modcons-distwtgtnmut.json',
    #     'avg_dist_among_muts': 'eval-results-w-demo-modcons-avgdistsmongmuts.json'
    # }

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
    def load_model(cls):
        model = None
        tokenizer = None
        if cls.model_name=='llama':
            model = LlamaModel.load_model(
                ckpt_dir = Macros.llama_model_dir,
                tokenizer_path = Macros.llama_model_root_dir
            )
        elif cls.model_name=='alpaca':
            model, tokenizer = Alpaca.load_model()
        elif cls.model_name=='gpt3.5':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt3d5_engine_name
            )
        elif cls.model_name=='gpt4':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4_engine_name
            )
        elif cls.model_name=='gpt4omini':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4omini_engine_name
            )
        # end if
        return model, tokenizer


    @classmethod
    def get_emb_consistency(
        cls, 
        mut_cot_resp: str, 
        emb_consistency_res: Dict,
        metric='cos_sim'
    ) -> List:
        if metric=='random':
            return None
        # end if
        return emb_consistency_res[metric][mut_cot_resp]

    @classmethod
    def select_mutations_for_demo(
        cls, 
        mut_res: List[Dict],
        emb_consist_type: str,
        top_k: int=5
    ) -> List[Dict]:
        selected_mut = list()
        for m_res in mut_res:
            q = m_res['question']
            cot_resp = m_res['cot_response']

            # measure modal consistency
            mod_cons = m_res['mod_cons']

            # measure context consistency
            emb_cons = m_res['emb_cons']

            if emb_consist_type.startswith('modcons-'):
                if mod_cons:
                    # first, we only select consistent mutations over modals
                    selected_mut.append({
                        'question': q,
                        'cot_response': cot_resp,
                        'emb_cons': emb_cons
                    })
                # end if
            else:
                selected_mut.append({
                    'question': q,
                    'cot_response': cot_resp,
                    'emb_cons': emb_cons
                })
            # end if
        # end for
        # second, we rank mutations by the emb_consistency
        if emb_consist_type.endswith('random'):
            # random shuffle the mutations
            random.shuffle(selected_mut)
        else:
            selected_mut = sorted(
                selected_mut, 
                key=lambda x: x['emb_cons'], 
                reverse=True
            )
        # end if
        return selected_mut[:top_k]

    @classmethod
    def get_demo_examples(
        cls,
        data_dict: Dict, 
        mod_consistency_res: Dict,
        emb_consistency_res: Dict,
        emb_consist_type: str,
        top_k: int=5,
    ) -> List[str]:
        mut_data = data_dict['mutation']
        mut_consist_res = mod_consistency_res['mutation']
        mut_w_cons = list()
        for m_i, m in enumerate(mut_consist_res):
            mut_q = mut_data[m_i]['question']
            mut_cot_resp = mut_data[m_i]['cot_response']
            mut_mod_cons = m['consistency'] if emb_consist_type!='random' else None
            mut_emb_cons = None
            if emb_consistency_res is not None:
                mut_emb_cons = cls.get_emb_consistency(
                    mut_cot_resp, 
                    emb_consistency_res,
                    metric=emb_consist_type.split('modcons-')[-1]
                )
            # end if
            mut_w_cons.append({
                'question': mut_q,
                'cot_response': mut_cot_resp,
                'mod_cons': mut_mod_cons,
                'emb_cons': mut_emb_cons
            })
        # end for
        demos = cls.select_mutations_for_demo(
            mut_w_cons,
            emb_consist_type,
            top_k=top_k
        )
        return demos

    @classmethod
    def get_response(
        cls, 
        model: AutoModel,
        tokenizer: AutoTokenizer,
        input_text: str,
        prompt: str,
        prompt_append: bool
    ) -> List[str]:
        response = None
        if cls.model_name=='llama':
            response = LlamaModel.predict(
                model, 
                input_text,
                prompt,
                prompt_append=prompt_append
            )
        elif cls.model_name=='alpaca':
            response = Alpaca.predict(model, tokenizer, input_text)
        elif cls.model_name=='gpt3.5':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4omini':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        # end if
        return response

    @classmethod
    def main(
        cls,
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python',
        emb_consist_type: str='dist',
        top_k=5
    ) -> None:
        cls.model_name = model_name
        # if dataset_name=='svamp':
        #     res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        # else:
        #     res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        # # end if
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        mut_dir = res_dir / 'mutation'
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        llm_response_files = [
            f for f in os.listdir(str(eval_dir))
            if f.endswith('.json') and f.startswith('eval-')
        ]
        mod_consist_res_file = eval_dir / 'modal-consistency-results.json'
        emb_consist_res_file = eval_dir / 'emb-consistency-results.json'

        mod_consist_res = Utils.read_json(mod_consist_res_file)
        emb_consist_res = Utils.read_json(emb_consist_res_file)
        model, tokenizer = cls.load_model()
        demo_res = dict()
        eval_res_file = cls.emb_consist_type_n_res_file_map.get(emb_consist_type, None)
        if eval_res_file is not None:
            eval_res_file = eval_dir / eval_res_file
            # if os.path.exists(str(eval_res_file)):
            #     # os.remove(eval_res_file)
            #     demo_res = Utils.read_json(eval_res_file)
            # # end if
        # end if

        i = 0
        num_data = len(mod_consist_res.keys())
        for cksum_val in sorted(mod_consist_res.keys()):
            if cksum_val not in demo_res.keys():
                print(f"{i} out of {num_data}", cksum_val)
                # construct prompt with demo
                demo_prompt = ''
                data_dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
                orig_mod_consist = mod_consist_res[cksum_val]['orig']
            
                orig_q = data_dict['orig']['question']
                orig_a = data_dict['orig']['answer']
                orig_r = orig_mod_consist['correctness']['cot'][0]

                _emb_consist_res = None
                if emb_consist_res is not None:
                    _emb_consist_res = emb_consist_res[cksum_val]
                # end if

                demos: List[Dict] = cls.get_demo_examples(
                    data_dict,
                    mod_consist_res[cksum_val],
                    _emb_consist_res,
                    emb_consist_type=emb_consist_type,
                    top_k=top_k
                )
                for d_i in range(len(demos)):
                    demo_q = demos[d_i]['question']
                    demo_cot_r = demos[d_i]['cot_response']
                    demo_prompt += f"Q: {demo_q} {Macros.openai_cot_prompt}\nA: {demo_cot_r.strip()}\n"
                # end for

                # get responses from llm
                inp = f"{demo_prompt}Q: {orig_q} {Macros.openai_cot_prompt}\nA: "
                orig_resp_with_demo = cls.get_response( 
                    model,
                    tokenizer,
                    inp,
                    prompt=f"{Macros.openai_cot_prompt}\nA:",
                    prompt_append=True
                )
                demo_res[cksum_val] = {
                    'input_text': inp,
                    'response': orig_resp_with_demo,
                    'answer': orig_a,
                    'num_demo_used': len(demos)
                }
                # Utils.write_json(
                #     demo_res,
                #     eval_res_file,
                #     pretty_format=True
                # )
            # end if
            i += 1
        # end for
        Utils.write_json(
            demo_res,
            eval_res_file,
            pretty_format=True
        )
        return


class EvaluateWithAlgConsistencyDemo:

    model_name = None

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
    def load_model(cls):
        model = None
        tokenizer = None
        if cls.model_name=='llama':
            model = LlamaModel.load_model(
                ckpt_dir = Macros.llama_model_dir,
                tokenizer_path = Macros.llama_model_root_dir
            )
        elif cls.model_name=='alpaca':
            model, tokenizer = Alpaca.load_model()
        elif cls.model_name=='gpt3.5':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt3d5_engine_name
            )
        elif cls.model_name=='gpt4':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4_engine_name
            )
        elif cls.model_name=='gpt4omini':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4omini_engine_name
            )
        # end if
        return model, tokenizer

    @classmethod
    def get_alg_consistency(
        cls, 
        cksum_val: str,
        alg_const_dir: Path
    ) -> Dict:
        return Utils.read_json(
            alg_const_dir / f"alg-consistency-mat-{cksum_val}.json"
        )

    @classmethod
    def select_mutations_for_demo(
        cls, 
        data_dict: Dict,
        alg_consistency: Dict,
        top_k: int=5
    ) -> List[Dict]:
        res_matrix = alg_consistency['res_matrix']
        question_n_indices_map = alg_consistency['question_n_indices_map']
        score_dict = dict()

        for code_used_key, val_used_dict in res_matrix.items():
            # simply get the argmax of sum of values for each code used
            if code_used_key!='CODE_USED::TGT':
                score_dict[code_used_key] = sum([
                    v for v in val_used_dict.values()
                    if v is not None
                ])
            # end if
        # end for

        sorted_code_used_keys = sorted(
            list(score_dict.keys()),
            reverse=True,
            key=lambda x: score_dict[x]
        )

        if len(set([
            score_dict[v] 
            for v in sorted_code_used_keys
        ]))==1:
            random.shuffle(sorted_code_used_keys)
        # end if

        # second, we rank mutations by the emb_consistency
        selected_mut = list()
        for k in sorted_code_used_keys:
            _k = k.split('CODE_USED::')[-1].strip()
            # for s, s_k in question_n_indices_map['tgt'].items():
            #     if _k==s_k: 
            #         selected_mut.append({
            #             'question': data_dict['orig']['question'],
            #             'cot_response': data_dict['orig']['cot_response'],
            #             'code_response': data_dict['orig']['code_response']
            #         })
            #     # end if
            # # end for

            for ms, ms_k in question_n_indices_map['mut'].items():
                if _k==ms_k:
                    for d in data_dict['mutation']:
                        if d['question']==ms:
                            selected_mut.append({
                                'question': d['question'],
                                'cot_response': d['cot_response'],
                                'code_response': d['code_response']
                            })
                        # end if
                    # end for
                # end if
            # end for
        # end for
        return selected_mut[:top_k]

    @classmethod
    def get_demo_examples(
        cls,
        data_dict: Dict,
        alg_consistency_res: Dict,
        top_k: int=5
    ) -> List[str]:
        return cls.select_mutations_for_demo(
            data_dict,
            alg_consistency_res,
            top_k=top_k
        )

    @classmethod
    def get_response(
        cls, 
        model: AutoModel,
        tokenizer: AutoTokenizer,
        input_text: str,
        prompt: str,
        prompt_append: bool
    ) -> List[str]:
        response = None
        if cls.model_name=='llama':
            response = LlamaModel.predict(
                model, 
                input_text,
                prompt,
                prompt_append=prompt_append
            )
        elif cls.model_name=='alpaca':
            response = Alpaca.predict(model, tokenizer, input_text)
        elif cls.model_name=='gpt3.5':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4omini':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        # end if
        return response

    @classmethod
    def main(
        cls,
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python',
        top_k=5,
        include_only_modconst: bool=False
    ) -> None:
        cls.model_name = model_name
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        mut_dir = res_dir / 'mutation'
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        alg_const_dir = eval_dir / 'alg_consistency'
        if include_only_modconst:
            alg_const_dir = eval_dir / 'alg_consistency_only_modconst'
        # end if
        eval_res_file = alg_const_dir / 'eval-results-w-demo-alg-consistency.json'

        modcons_rand_res_file = eval_dir / 'eval-results-w-demo-modcons-random.json'
        modcons_rand_res = Utils.read_json(modcons_rand_res_file)

        llm_response_files = [
            f for f in os.listdir(str(eval_dir))
            if f.startswith('eval-') and f.endswith('.json') and \
            (not f.startswith('eval-results-w-demo'))
        ]

        model, tokenizer = cls.load_model()
        demo_res = dict()
        if eval_res_file is not None:
            if os.path.exists(str(eval_res_file)):
                # os.remove(eval_res_file)
                demo_res = Utils.read_json(eval_res_file)
            # end if
        # end if

        i = 0
        num_data = len(modcons_rand_res.keys())
        # num_data = len(llm_response_files)
        # for llm_response_file in sorted(llm_response_files):
        #     cksum_val = re.search(
        #         r'eval\-(.*)\.json', 
        #         llm_response_file
        #     ).group(1)

        for cksum_val in sorted(modcons_rand_res.keys()):
            if cksum_val not in demo_res.keys():
                print(f"{i} out of {num_data}", cksum_val)
                
                # construct prompt with demo
                demo_prompt = ''
                data_dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
                alg_const_dict = cls.get_alg_consistency(
                    cksum_val,
                    alg_const_dir
                )
                if alg_const_dict is None:
                    print('not exists!!')
                else:
                    orig_q = data_dict['orig']['question']
                    orig_a = data_dict['orig']['answer']

                    demos: List[Dict] = cls.get_demo_examples(
                        data_dict,
                        alg_const_dict,
                        top_k
                    )

                    for d_i in range(len(demos)):
                        demo_q = demos[d_i]['question']
                        demo_cot_r = demos[d_i]['cot_response']
                        demo_prompt += f"Q: {demo_q} {Macros.openai_cot_prompt}\nA: {demo_cot_r.strip()}\n"
                    # end for

                    # get responses from llm
                    inp = f"{demo_prompt}Q: {orig_q} {Macros.openai_cot_prompt}\nA: "
                    orig_resp_with_demo = cls.get_response( 
                        model,
                        tokenizer,
                        inp,
                        prompt=f"{Macros.openai_cot_prompt}\nA:",
                        prompt_append=True
                    )
                    demo_res[cksum_val] = {
                        'input_text': inp,
                        'response': orig_resp_with_demo,
                        'answer': orig_a,
                        'num_demo_used': len(demos)
                    }
                    Utils.write_json(
                        demo_res,
                        eval_res_file,
                        pretty_format=True
                    )
                # end if
            # end if
            i += 1
        # end for
        Utils.write_json(
            demo_res,
            eval_res_file,
            pretty_format=True
        )
        return


# class EvaluateWithAlgConsistCode:

#     model_name = None

#     @classmethod
#     def load_model(cls):
#         model = None
#         tokenizer = None
#         if cls.model_name=='llama':
#             model = LlamaModel.load_model(
#                 ckpt_dir = Macros.llama_model_dir,
#                 tokenizer_path = Macros.llama_model_root_dir
#             )
#         elif cls.model_name=='alpaca':
#             model, tokenizer = Alpaca.load_model()
#         elif cls.model_name=='gpt3.5':
#             OpenAiModel.set_model_name(
#                 engine_name=Macros.gpt3d5_engine_name
#             )
#         elif cls.model_name=='gpt4':
#             OpenAiModel.set_model_name(
#                 engine_name=Macros.gpt4_engine_name
#             )
#         # end if
#         return model, tokenizer

#     @classmethod
#     def get_alg_consistency(
#         cls, 
#         cksum_val: str,
#         alg_const_dir: Path
#     ) -> Dict:
#         return Utils.read_json(
#             alg_const_dir / f"alg-consistency-mat-{cksum_val}.json"
#         )
    
#     @classmethod
#     def select_mutations_for_demo(
#         cls, 
#         data_dict: Dict,
#         alg_consistency: Dict,
#         top_k: int=5
#     ) -> List[Dict]:
#         res_matrix = alg_consistency['res_matrix']
#         question_n_indices_map = alg_consistency['question_n_indices_map']
#         score_dict = dict()

#         for code_used_key, val_used_dict in res_matrix.items():
#             # simply get the argmax of sum of values for each code used
#             score_dict[code_used_key] = sum([
#                 v for v in val_used_dict.values()
#                 if v is not None
#             ])
#         # end for
#         sorted_code_used_keys = sorted(
#             list(score_dict.keys()),
#             reverse=True,
#             key=lambda x: score_dict[x]
#         )

#         if len(set([
#             score_dict[v] 
#             for v in sorted_code_used_keys
#         ]))==1:
#             sorted_code_used_keys = ['CODE_USED::TGT']
#         # end if

#         # second, we rank mutations by the emb_consistency
#         selected_mut = list()
#         for k in sorted_code_used_keys[:top_k]:
#             _k = k.split('CODE_USED::')[-1].strip()
#             for ts, ts_k in question_n_indices_map['tgt'].items():
#                 if _k==ts_k:
#                     r = {
#                         'index': k,
#                         'question': data_dict['orig']['question'],
#                         'cot_response': data_dict['orig']['cot_response'],
#                         'code_response': data_dict['orig']['code_response']
#                     }
#                     if r not in selected_mut:
#                         selected_mut.append(r)
#                     # end if
#                 # end if
#             # end for
#             for ms, ms_k in question_n_indices_map['mut'].items():
#                 if _k==ms_k:
#                     for d in data_dict['mutation']:
#                         if d['question']==ms:
#                             r = {
#                                 'index': k,
#                                 'question': d['question'],
#                                 'cot_response': d['cot_response'],
#                                 'code_response': d['code_response']
#                             }
#                             if r not in selected_mut:
#                                 selected_mut.append(r)
#                             # end if
#                         # end if
#                     # end for
#                 # end if
#             # end for
#         # end for
#         return selected_mut

#     @classmethod
#     def main(
#         cls,
#         model_name: str, 
#         dataset_name: str, 
#         pl_type: str='python',
#         top_k=5,
#         include_only_modconst: bool=False
#     ) -> None:
#         cls.model_name = model_name
#         res_dir = Macros.result_dir / 'nl2nl' / dataset_name
#         mut_dir = res_dir / 'mutation'
#         eval_dir = res_dir / 'evaluate_consistency' / model_name
#         alg_const_dir = eval_dir / 'alg_consistency'
#         if include_only_modconst:
#             alg_const_dir = eval_dir / 'alg_consistency_only_modconst'
#         # end if
#         eval_res_file = alg_const_dir / 'eval-results-w-code-alg-consistency.json'

#         modcons_rand_res_file = eval_dir / 'acc-res-modcons-random.json'
#         modcons_rand_res = Utils.read_json(modcons_rand_res_file)

#         llm_response_files = [
#             f for f in os.listdir(str(eval_dir))
#             if f.startswith('eval-') and f.endswith('.json') and \
#             (not f.startswith('eval-results-w-demo'))
#         ]

#         model, tokenizer = cls.load_model()
#         eval_res = dict()
#         # if eval_res_file is not None:
#         #     if os.path.exists(str(eval_res_file)):
#         #         # os.remove(eval_res_file)
#         #         eval_res = Utils.read_json(eval_res_file)
#         #     # end if
#         # # end if

#         i = 0
#         num_data = len(modcons_rand_res.keys())
#         for cksum_val in sorted(modcons_rand_res.keys()):
#             # if cksum_val not in eval_res.keys():
#             print(f"{i} out of {num_data}", cksum_val)
#             data_dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
#             orig_q = data_dict['orig']['question']
#             orig_a = data_dict['orig']['answer']
#             alg_const_dict = cls.get_alg_consistency(
#                 cksum_val,
#                 alg_const_dir
#             )
#             selected_mut = cls.select_mutations_for_demo(
#                 data_dict,
#                 alg_const_dict,
#                 top_k=top_k
#             )
#             ans_list = list()
#             code_used_list = list()
#             for smut in selected_mut:
#                 smut_ind = smut['index']
#                 ans, code_used = AlgorithmConsistency.get_answer_from_mut_of_interest(
#                     cksum_val,
#                     smut['question'],
#                     dataset_name,
#                     model_name,
#                     pl_type=pl_type,
#                     include_only_modconst=include_only_modconst
#                 )
#                 if (ans is not None) and (ans!='cot_response'):
#                     ans_search = re.search(r'([-|\$]?\d+)', ans)
#                     if ans_search is not None:
#                         ans = ans_search.group(1).strip().replace('$', '')
#                     # end if
#                     ans_list.append(eval(ans))
#                     code_used_list.append(code_used)
#                 # end if
#             # end for
#             if len(set(ans_list))==len(ans_list) and \
#                 len(ans_list)>1:
#                 # if every answer from selected_mut is different from each other
#                 # then we just take original CoT response
#                 code_used = f"cot_response::all_different_ans_{str(ans_list)}"
#                 ans = data_dict['orig']['cot_response']
#             elif not any(ans_list):
#                 # if every answer is None,
#                 # then we just take original CoT response
#                 code_used = 'cot_response::all_none'
#                 ans = data_dict['orig']['cot_response']
#             else:
#                 ans = max(set(ans_list), key = ans_list.count)
#                 code_used = [
#                     code_used_list[c_i] 
#                     for c_i, c in enumerate(code_used_list)
#                     if ans_list[c_i]==ans
#                 ]
#             # end if
#             eval_res[cksum_val] = {
#                 'input_text': orig_q,
#                 'code_used_for_response': code_used,
#                 'final_response': str(ans),
#                 'orig_cot_response': data_dict['orig']['cot_response'],
#                 'orig_code_response': data_dict['orig']['code_response'],
#                 'answer': orig_a
#             }
#             Utils.write_json(
#                 eval_res,
#                 eval_res_file,
#                 pretty_format=True
#             )
#             i += 1
#         # end for
#         Utils.write_json(
#             eval_res,
#             eval_res_file,
#             pretty_format=True
#         )
#         return


class EvaluateWithModconstNDiscriminator:

    model_name = None

    @classmethod
    def load_model(cls):
        model = None
        tokenizer = None
        if cls.model_name=='llama':
            model = LlamaModel.load_model(
                ckpt_dir = Macros.llama_model_dir,
                tokenizer_path = Macros.llama_model_root_dir
            )
        elif cls.model_name=='alpaca':
            model, tokenizer = Alpaca.load_model()
        elif cls.model_name=='gpt3.5':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt3d5_engine_name
            )
        elif cls.model_name=='gpt4':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4_engine_name
            )
        elif cls.model_name=='gpt4omini':
            OpenAiModel.set_model_name(
                engine_name=Macros.gpt4omini_engine_name
            )
        # end if
        return model, tokenizer

    @classmethod
    def load_discriminator(
        cls,
        dataset_name: str,
        query_model_name: str,
        code_model_name: str,
        out_dim: int
    ):
        discriminator_model_dir = Macros.result_dir / 'discriminator'
        if dataset_name=='svamp':
            discriminator_train_dataset_name = 'asdiv'
        elif dataset_name=='asdiv':
            discriminator_train_dataset_name = 'svamp'
        # end if

        pretrained_query_model_path = discriminator_model_dir / f"query_model_trained_from_{discriminator_train_dataset_name}" / 'checkpoint_100.pth.tar'
        pretrained_code_model_path = discriminator_model_dir / f"code_model_trained_from_{discriminator_train_dataset_name}" / 'checkpoint_100.pth.tar'

        tokenizer_query, \
        tokenizer_code,\
        model_query, \
        model_code = load_pretrained_model(
            query_model_name,
            code_model_name,
            out_dim,
            pretrained_query_model_path,
            pretrained_code_model_path
        )
        return tokenizer_query, \
            tokenizer_code,\
            model_query, \
            model_code

    @classmethod
    def select_mutations_for_demo(
        cls, 
        mut_res: List[Dict],
        discriminator_query_tokenizer: AutoTokenizer,
        discriminator_query_model: AutoModel,
        discriminator_code_tokenizer: AutoTokenizer,
        discriminator_code_model: AutoModel,
        discriminator_device: Any,
        discriminaotr_score_threshold: float=0.5,
        top_k: int=5
    ) -> List[Dict]:
        selected_mut = list()
        for m_res in mut_res:
            q = m_res['question']
            a = m_res['answer']
            cot_resp = m_res['cot_response']
            code_resp = m_res['code_response']
            
            # measure modal consistency
            mod_cons = m_res['mod_cons']
            if mod_cons:
                # first, we only select consistent mutations over modals
                code_resp = code_resp.split('the answer is')[0].strip()
                selected_mut.append({
                    'question': q,
                    'cot_response': cot_resp,
                    'code_response': code_resp,
                    'answer': a
                })
            # end if
        # end for

        # second, we rank mutations by scores computed from discriminator
        selected_mut_w_score = list()
        low_scored_examples = list()
        for m_i, m in enumerate(selected_mut):

            score = get_correctness_score(
                discriminator_query_tokenizer, 
                discriminator_code_tokenizer,
                discriminator_query_model, 
                discriminator_code_model,
                m['question'],
                m['code_response'],
                discriminator_device
            )

            selected_mut_w_score.append({
                'question': m['question'],
                'cot_response': m['cot_response'],
                'code_response': m['code_response'],
                'answer': m['answer'],
                'discriminator_score': score
            })
            if score<discriminaotr_score_threshold:
                low_scored_examples.append(selected_mut_w_score[-1])
                print(selected_mut_w_score[-1])
                print()
            # end if
        # end for
        selected_mut_w_score = [
            m for m in selected_mut_w_score
            if m['discriminator_score'] >= discriminaotr_score_threshold
        ]
        selected_mut_w_score = sorted(
            selected_mut_w_score, 
            key=lambda x: x['discriminator_score'],
            reverse=True
        )
        return selected_mut_w_score[:top_k], low_scored_examples

    @classmethod
    def get_demo_examples(
        cls,
        data_dict: Dict, 
        mod_consistency_res: Dict,
        discriminator_query_tokenizer: AutoTokenizer,
        discriminator_query_model: AutoModel,
        discriminator_code_tokenizer: AutoTokenizer,
        discriminator_code_model: AutoModel,
        discriminaotr_device: Any,
        discriminaotr_score_threshold: float=0.5,
        top_k: int=5,
    ) -> List[str]:
        mut_data = data_dict['mutation']
        mut_consist_res = mod_consistency_res['mutation']
        mut_w_cons = list()
        for m_i, m in enumerate(mut_consist_res):
            mut_q = mut_data[m_i]['question']
            mut_a = mut_data[m_i]['answer']
            mut_cot_resp = mut_data[m_i]['cot_response']
            mut_code_resp = mut_data[m_i]['code_response']
            mut_mod_cons = m['consistency']
            mut_w_cons.append({
                'question': mut_q,
                'cot_response': mut_cot_resp,
                'code_response': mut_code_resp,
                'mod_cons': mut_mod_cons,
                'answer': mut_a,
            })
        # end for
        demos, low_scored_muts = cls.select_mutations_for_demo(
            mut_w_cons,
            discriminator_query_tokenizer,
            discriminator_query_model,
            discriminator_code_tokenizer,
            discriminator_code_model,
            discriminaotr_device,
            discriminaotr_score_threshold,
            top_k=top_k
        )
        return demos, low_scored_muts

    @classmethod
    def get_response(
        cls, 
        model: AutoModel,
        tokenizer: AutoTokenizer,
        input_text: str,
        prompt: str,
        prompt_append: bool
    ) -> List[str]:
        response = None
        if cls.model_name=='llama':
            response = LlamaModel.predict(
                model, 
                input_text,
                prompt,
                prompt_append=prompt_append
            )
        elif cls.model_name=='alpaca':
            response = Alpaca.predict(model, tokenizer, input_text)
        elif cls.model_name=='gpt3.5':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif cls.model_name=='gpt4omini':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        # end if
        return response

    @classmethod
    def main(
        cls,
        model_name: str, 
        dataset_name: str, 
        query_model_name: str,
        code_model_name: str,
        discriminator_out_dim: int,
        discriminaotr_score_threshold: float=0.5,
        pl_type: str='python',
        top_k=5
    ) -> None:
        cls.model_name = model_name
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        mut_dir = res_dir / 'mutation'
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        llm_response_files = [
            f for f in os.listdir(str(eval_dir))
            if f.endswith('.json') and f.startswith('eval-')
        ]
        mod_consist_res_file = eval_dir / 'modal-consistency-results.json'
        mod_consist_res = Utils.read_json(mod_consist_res_file)
        model, tokenizer = cls.load_model()

        discriminator_query_tokenizer, \
        discriminator_code_tokenizer,\
        discriminator_query_model, \
        discriminator_code_model = cls.load_discriminator(
            dataset_name,
            query_model_name,
            code_model_name,
            discriminator_out_dim
        )

        if torch.cuda.is_available():
            discriminator_device = torch.device('cuda')
            cudnn.deterministic = True
            cudnn.benchmark = True
        else:
            discriminator_device = torch.device('cpu')
        # end if

        # discriminator_query_tokenizer.to(discriminator_device)
        # discriminator_code_tokenizer.to(discriminator_device)
        discriminator_query_model.to(discriminator_device)
        discriminator_code_model.to(discriminator_device)
        
        eval_res_file = eval_dir / 'eval-results-w-modconst-n-discriminator.json'
        demo_res = dict()

        i = 0
        num_data = len(mod_consist_res.keys())
        low_scored_muts_dict = dict()
        for cksum_val in sorted(mod_consist_res.keys()):
            if cksum_val not in demo_res.keys():
                print(f"{i} out of {num_data}", cksum_val)
                # construct prompt with demo
                demo_prompt = ''
                data_dict = Utils.read_json(eval_dir / f"eval-{cksum_val}.json")
                orig_mod_consist = mod_consist_res[cksum_val]['orig']
            
                orig_q = data_dict['orig']['question']
                orig_a = data_dict['orig']['answer']
                orig_r = orig_mod_consist['correctness']['cot'][0]

                demos, low_scored_muts = cls.get_demo_examples(
                    data_dict,
                    mod_consist_res[cksum_val],
                    discriminator_query_tokenizer,
                    discriminator_query_model,
                    discriminator_code_tokenizer,
                    discriminator_code_model,
                    discriminator_device,
                    discriminaotr_score_threshold=discriminaotr_score_threshold,
                    top_k=top_k
                )
                low_scored_muts_dict[cksum_val] = low_scored_muts
                demo_discriminator_scores = list()
                for d_i in range(len(demos)):
                    demo_q = demos[d_i]['question']
                    demo_cot_r = demos[d_i]['cot_response']
                    demo_discriminator_scores.append(demos[d_i]['discriminator_score'])
                    demo_prompt += f"Q: {demo_q} {Macros.openai_cot_prompt}\nA: {demo_cot_r.strip()}\n"
                # end for

                # get responses from llm
                inp = f"{demo_prompt}Q: {orig_q} {Macros.openai_cot_prompt}\nA: "
                orig_resp_with_demo = cls.get_response( 
                    model,
                    tokenizer,
                    inp,
                    prompt=f"{Macros.openai_cot_prompt}\nA:",
                    prompt_append=True
                )
                
                demo_res[cksum_val] = {
                    'input_text': inp,
                    'response': orig_resp_with_demo,
                    'answer': orig_a,
                    'num_demo_used': len(demos),
                    'discriminator_scores': demo_discriminator_scores
                }
                # Utils.write_json(
                #     demo_res,
                #     eval_res_file,
                #     pretty_format=True
                # )
            # end if
            i += 1
        # end for
        Utils.write_json(
            demo_res,
            eval_res_file,
            pretty_format=True
        )
        Utils.write_json(
            low_scored_muts_dict,
            eval_dir / 'eval-results-w-modconst-n-discriminator-low-scored-mutations.json',
            pretty_format=True
        )
        return