
import os
import re
import openai
import numpy as np

from typing import *
from pathlib import Path
from openai import OpenAI
from collections import Counter

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger
# from ..llmut.openai import OpenAiModel
from ..llmut.palm import PaLMModel
from ..consistency.selfconsistency import SelfConsistency

from .multimodals import Multimodals

client = OpenAI()


class Genetic:

    def __init__(
        self, 
        eval_file_name: str,
        eval_dir: Path
    ):
        self.eval_file_name = eval_file_name
        self.eval_dir = eval_dir
        self.eval_res = self.load_eval_results()
        # self.target_question: str = eval_res['orig']['question']
        # self.orig_cot_response: str = eval_res['orig']['cot_response']
        # self.mutations: List[Dict] = eval_res['mutation']

    def load_eval_results(self):
        '''
        keys:
        1 'orig', 2. 'mutation'
        each original and mutations has question, multiple modal responses (3 for now)
        '''
        mut_file = self.eval_dir / self.eval_file_name
        return Utils.read_json(mut_file)

    def get_target_prompt(self, target_response_type: str):
        prompt, is_prompt_append = Macros.prompts_over_modals[target_response_type]
        return prompt, is_prompt_append

    def get_answer_from_cot_resp(self, cot_resp: str) -> str:
        return Utils.get_answer_from_cot_resp(cot_resp)

    def get_answer_from_code_resp(self, code_resp: str, dataset_name: str, model_name:str=None) -> str:
        return Utils.get_answer_from_code_resp(code_resp, dataset_name, model_name=model_name)

    def get_answer_from_eqn_resp(self, eqn_resp: str) -> str:
        return Utils.get_answer_from_eqn_resp(eqn_resp)

    def get_answer_from_resp(
        self, 
        response: str,
        dataset_name: str,
        modal_name: str,
        model_name: str=None
    ) -> Any:
        if modal_name=='cot_response':
            answer = self.get_answer_from_cot_resp(response)
        elif modal_name=='code_response':
            answer = self.get_answer_from_code_resp(response, dataset_name, model_name=model_name)
        else:
            answer = self.get_answer_from_eqn_resp(response)
        # end if
        return answer

    def get_majority_answer(
        self, 
        answers_over_modals: Dict
    ):  
        c = Counter(answers_over_modals.values())
        value_freqs = c.most_common()
        if len(set([v[1] for v in value_freqs]))==1:
            # in case of having no most frequent answer
            return answers_over_modals['cot_response']
        else:
            return value_freqs[0][0]
        # end if

    def get_openai_model_response(
        self,
        model_name: str,
        input_text: str,
        demo_str: str=None,
        temp: float=Macros.resp_temp,
        top_k: int=None
    ) -> str:
        if demo_str is not None:
            input_text = f"{demo_str}{input_text}"
        # end if
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    'role': 'user', 
                    'content': input_text
                }],
                top_p=1,
                temperature=temp,
                max_tokens=Macros.llm_resp_max_len
            )
            if top_k is None:
                logprobs = response.choices[0].logprobs.content
                return response.choices[0].message.content
            else:
                return [
                    r.message.content
                    for r in response.choices[:top_k]
                ]
            # end if
        except openai.BadRequestError as e:
            print(f"BadRequestError: {e}")
            if e.code=='context_length_exceeded':
                print(input_text)
            # end if
            pass
        # end try
        return
        
    def get_response_with_mutation(
        self, 
        llm_name: str,
        target_question: str,
        mutation: Dict,
        demo_str_over_modals: Dict
    ) -> Dict:
        # get responses from llm
        response_dict = dict()
        for mod_name in Macros.prompts_over_modals.keys():
            prompt, is_prompt_append = Macros.prompts_over_modals[mod_name]

            m_q = mutation['question']
            m_resp = mutation[mod_name]

            demo_prompt = demo_str_over_modals[mod_name]
            input_text = f"Q: {prompt} {target_question}\nA: "
            if is_prompt_append:
                input_text = f"Q: {target_question} {prompt}\nA: "
            # end if

            model_name = Macros.gpt3d5_engine_name if llm_name=='gpt3.5' else Macros.gpt4_engine_name
            response = self.get_openai_model_response(
                model_name,
                input_text,
                demo_str=demo_prompt
            )
            response_dict[mod_name] = response
        # end for
        return response_dict

    def compute_modal_consistency(
        self, 
        response_dict: Dict,
        dataset_name: str,
        model_name: str,
        answer_dict: Dict=None
    ) -> float:
        # modal_consistency = number of unique answers over the modals/number of modals(prompts)
        if answer_dict is not None:
            modal_consistency = 1.-len(set(answer_dict.values()))*1. / len(answer_dict.keys())
            return modal_consistency, answer_dict
        else:
            answer_dict = dict()
            num_correct_modal = 0
            for modal_name in Macros.prompts_over_modals.keys():
                resp = response_dict[modal_name]
                answer_dict[modal_name] = self.get_answer_from_resp(
                    resp, 
                    dataset_name, 
                    modal_name, 
                    model_name=model_name
                )
            # end for
            modal_consistency = 1.-len(set(answer_dict.values()))*1. / len(answer_dict.keys())
            return modal_consistency, answer_dict
        # end if

    def select_mutations_by_modal_consistency(
        self,
        target_question_response: Dict,
        mutations: List[Dict],
        prev_ans_dict: Dict,
        prev_consistency: float,
        prev_selected_mutations: List[Dict],
        llm_name: str,
        demo_str_over_modals: Dict,
        dataset_name: str,
        target_response_type: str='cot_response',
        max_mod_const_val: float = None
    ) -> List[Dict]:
        '''
        for each mutation:
            1. append the each mutation to the target question.
            2. get the responses of the appended question over the modals.
            3. compute the modal consistency from the responses.
            4. find the mutation that increase the consistency the most
            5. return the mutation selected from step 4.
        '''
        # cons_orig, _ = self.compute_modal_consistency(target_question_response, dataset_name)
        prev_selected_mutated_questions = [
            m['mut']['question'] for m in prev_selected_mutations
        ]
        prev_ans_by_maj = self.get_majority_answer(prev_ans_dict)
        positive_mutations = list()
        for m in mutations:
            m_q = m['question']
            answer_dict = m.get('ans_dict', None)
            if answer_dict is not None:
                m_cons_mut, _ = self.compute_modal_consistency(
                    m, 
                    dataset_name, 
                    model_name=llm_name if llm_name=='gpt4' else None,
                    answer_dict=answer_dict
                )
            else:
                m_cons_mut, answer_dict = self.compute_modal_consistency(
                    m, 
                    dataset_name, 
                    model_name=llm_name if llm_name=='gpt4' else None
                )
            # end if
            if (m_q not in prev_selected_mutated_questions) and \
                (m_cons_mut==max_mod_const_val):
                m_response_dict = self.get_response_with_mutation(
                    llm_name,
                    target_question_response['question'],
                    m,
                    demo_str_over_modals
                )
                m_cons, ans_dict = self.compute_modal_consistency(
                    m_response_dict, 
                    dataset_name, 
                    model_name=llm_name if llm_name=='gpt4' else None
                )
                ans_by_maj = self.get_majority_answer(ans_dict)
                if prev_consistency<m_cons:
                    # 1. if consistency gets better, then we take the mutation
                    selected_mut_dict = {
                        'mut': m,
                        'mut_response': m_response_dict,
                        'mut_mod_consistency': m_cons_mut,
                        'mod_consistency_after_append_mut': m_cons,
                        'ans_dict': ans_dict
                    }
                    if selected_mut_dict not in prev_selected_mutations:
                        positive_mutations.append(selected_mut_dict)
                    # end if
                elif prev_consistency==m_cons and \
                    prev_ans_by_maj!=ans_by_maj:
                    # 2. if consistency is same as the previous one, 
                    # but the answer is different from previous one, 
                    # then, we take the mutation because of empirical observation 
                    # that the demonstration is likely to correct the answer for the math problem
                    selected_mut_dict = {
                        'mut': m,
                        'mut_response': m_response_dict,
                        'mut_mod_consistency': m_cons_mut,
                        'mod_consistency_after_append_mut': m_cons,
                        'ans_dict': ans_dict
                    }
                    if selected_mut_dict not in prev_selected_mutations:
                        positive_mutations.append(selected_mut_dict)
                    # end if
                # end if
            # end if
        # end for

        if any(positive_mutations):
            # sort by consistency values
            positive_mutations = sorted(
                positive_mutations, 
                key=lambda x: x['mod_consistency_after_append_mut'],
                reverse=True
            )
        # end if
        return positive_mutations

    def get_final_answer(
        self,
        target_question_response_dict: Dict,
        mutations: List[Dict],
        llm_name: str,
        dataset_name: str,
        target_response_type: str='cot_response',
        max_num_muts = 5
    ):
        selected_mutations = list()
        prompt, is_prompt_append = self.get_target_prompt(target_response_type)
        steps_for_finding_muts = 0
        max_steps_for_finding_muts = 7
        demo_str_over_modals = {
            modal_name: ''
            for modal_name in Macros.prompts_over_modals.keys()
        }
        cons_orig, final_ans_dict = self.compute_modal_consistency(
            target_question_response_dict, 
            dataset_name,
            model_name=llm_name if llm_name=='gpt4' else None
        )
        cons_final = cons_orig
        max_cons_val = 1.-(1./len(Macros.prompts_over_modals.keys()))
        while steps_for_finding_muts<max_steps_for_finding_muts and \
            cons_final<max_cons_val:

            print(steps_for_finding_muts, cons_orig, cons_final, final_ans_dict, len(selected_mutations))

            if steps_for_finding_muts==0:
                cons_final = cons_orig
            # else:
            #     cons_final, ans_dict = self.compute_modal_consistency(target_question_response_dict, dataset_name)
            # end if
            
            positive_muts = self.select_mutations_by_modal_consistency(
                target_question_response_dict,
                mutations,
                final_ans_dict,
                cons_final,
                selected_mutations,
                llm_name,
                demo_str_over_modals,
                dataset_name,
                target_response_type=target_response_type,
                max_mod_const_val=max_cons_val
            )
            if any(positive_muts) and len(selected_mutations)<max_num_muts:
                # 'mut': m,
                # 'mut_response': m_response_dict,
                # 'mut_mod_consistency': m_cons_mut,
                # 'mod_consistency_after_append_mut': m_cons,
                # 'ans_dict': ans_dict
                selected_mut = positive_muts[0]
                selected_mutations.append(selected_mut)
                selected_m_q = selected_mut['mut']['question']
                selected_m_resp = selected_mut['mut'][target_response_type]
                cons_final = selected_mut['mod_consistency_after_append_mut']
                final_ans_dict = selected_mut['ans_dict']

                # update prompts over modalities with selected mutation
                for key in selected_mut['mut_response'].keys():
                    if target_response_type==key:
                        # update question
                        _demo_str = f"Q: {prompt} {selected_m_q}\nA: {selected_m_resp.strip()}\n"
                        if is_prompt_append:
                            _demo_str = f"Q: {selected_m_q} {prompt}\nA: {selected_m_resp.strip()}\n"
                        # end if
                        demo_str_over_modals[key] += _demo_str
                    else:
                        _prompt, _is_prompt_append = Macros.prompts_over_modals[key]
                        _selected_m_resp = selected_mut['mut'][key]
                        _demo_str = f"Q: {_prompt} {selected_m_q}\nA: {_selected_m_resp.strip()}\n"
                        if _is_prompt_append:
                            _demo_str = f"Q: {selected_m_q} {_prompt}\nA: {_selected_m_resp.strip()}\n"
                        # end if
                        demo_str_over_modals[key] += _demo_str
                    # end if
                    target_question_response_dict[key] = selected_mut['mut_response'][key]
                # end for
            else:
                break
            # end if
            steps_for_finding_muts += 1
        # end while

        return target_question_response_dict, selected_mutations, cons_orig, cons_final, final_ans_dict

    @classmethod
    def main(
        cls,
        dataset_name: str,
        llm_name: str
    ) -> None:
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_dir = Macros.result_dir / 'genetic'/ dataset_name / 'evaluate_consistency' / llm_name
        genetic_dir.mkdir(parents=True, exist_ok=True)

        eval_res_files = sorted([
            f_name for f_name in os.listdir(str(eval_dir))
            if f_name.endswith('.json') and \
                f_name.startswith('eval-') and \
                (not f_name.startswith('eval-results-w'))
        ])
        f_index = 0
        result = dict()
        if os.path.exists(str(genetic_dir / 'final_answers.json')):
            result = Utils.read_json(genetic_dir / 'final_answers.json')
        # end if
        for eval_res_file in eval_res_files:
            gt_answer = Utils.read_json(eval_dir / eval_res_file)['orig']['answer']
            cksum_val = eval_res_file.split('-')[-1].split('.json')[0].strip()
            if cksum_val not in result.keys():
                print(f"{f_index} OUT OF {len(eval_res_files)}::{cksum_val}")
                obj = cls(eval_res_file, eval_dir)

                final_response_dict, \
                selected_mutations, \
                orig_cons_score, \
                final_cons_score, \
                final_answer_dict = obj.get_final_answer(
                    obj.eval_res['orig'],
                    obj.eval_res['mutation'],
                    llm_name,
                    dataset_name
                )
                result[cksum_val] = {
                    'final_response': final_response_dict,
                    'fianl_answer': final_answer_dict,
                    'answer': gt_answer,
                    'original_consistency_score': orig_cons_score,
                    'final_consistency_score': final_cons_score,
                    'mutation': selected_mutations
                }
                Utils.write_json(
                    result,
                    genetic_dir / 'final_answers.json',
                    pretty_format=True
                )
            # end if
            f_index += 1
        # end for
        Utils.write_json(
            result,
            genetic_dir / 'final_answers.json',
            pretty_format=True
        )
        return

    @classmethod
    def main_w_selfconsistency(
        cls,
        dataset_name: str,
        llm_name: str,
        target_modality: str
    ) -> None:
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_dir = Macros.result_dir / 'genetic'/ dataset_name / 'evaluate_consistency' / llm_name
        genetic_dir.mkdir(parents=True, exist_ok=True)

        eval_res_files = sorted([
            f_name for f_name in os.listdir(str(eval_dir))
            if f_name.endswith('.json') and \
                f_name.startswith('eval-') and \
                (not f_name.startswith('eval-results-w'))
        ])
        f_index = 0
        result = dict()
        if os.path.exists(str(genetic_dir / f"final_answers_w_self_consistency_over_{target_modality}.json")):
            result = Utils.read_json(genetic_dir / f"final_answers_w_self_consistency_over_{target_modality}.json")
        # end if
        for eval_res_file in eval_res_files:
            gt_answer = Utils.read_json(eval_dir / eval_res_file)['orig']['answer']
            cksum_val = eval_res_file.split('-')[-1].split('.json')[0].strip()
            if cksum_val not in result.keys():
                print(f"{f_index} OUT OF {len(eval_res_files)}::{cksum_val}")
                obj = cls(eval_res_file, eval_dir)

                final_response_dict, \
                selected_mutations, \
                orig_cons_score, \
                final_cons_score, \
                final_answer_dict = obj.get_final_answer(
                    obj.eval_res['orig'],
                    obj.eval_res['mutation'],
                    llm_name,
                    dataset_name,
                    target_modality=target_modality
                )
                result[cksum_val] = {
                    'final_response': final_response_dict,
                    'fianl_answer': final_answer_dict,
                    'answer': gt_answer,
                    'original_consistency_score': orig_cons_score,
                    'final_consistency_score': final_cons_score,
                    'mutation': selected_mutations
                }
                Utils.write_json(
                    result,
                    genetic_dir / f"final_answers_w_self_consistency_over_{target_modality}.json",
                    pretty_format=True
                )
            # end if
            f_index += 1
        # end for
        Utils.write_json(
            result,
            genetic_dir / f"final_answers_w_self_consistency_over_{target_modality}.json",
            pretty_format=True
        )
        return



class GeneticUsingSelfConsistency:

    def __init__(
        self, 
        eval_file_name: str,
        eval_dir: Path
    ):
        self.eval_file_name = eval_file_name
        self.eval_dir = eval_dir
        self.eval_res = self.load_eval_results()
        # self.target_question: str = eval_res['orig']['question']
        # self.orig_cot_response: str = eval_res['orig']['cot_response']
        # self.mutations: List[Dict] = eval_res['mutation']

    def load_eval_results(self):
        '''
        keys:
        1 'orig', 2. 'mutation'
        each original and mutations has question, multiple modal responses (3 for now)
        '''
        mut_file = self.eval_dir / self.eval_file_name
        return Utils.read_json(mut_file)

    def get_target_prompt(self, target_response_type: str):
        prompt, is_prompt_append = Macros.prompts_over_modals[target_response_type]
        return prompt, is_prompt_append

    def get_answer_from_cot_resp(self, cot_resp: str) -> str:
        return Utils.get_answer_from_cot_resp(cot_resp)

    def get_answer_from_code_resp(self, code_resp: str, dataset_name: str, model_name:str=None) -> str:
        return Utils.get_answer_from_code_resp(code_resp, dataset_name, model_name=model_name)

    def get_answer_from_eqn_resp(self, eqn_resp: str) -> str:
        return Utils.get_answer_from_eqn_resp(eqn_resp)

    def get_answer_from_resp(
        self, 
        response: str,
        dataset_name: str,
        modal_name: str,
        model_name: str=None
    ) -> Any:
        if modal_name=='cot_response':
            answer = self.get_answer_from_cot_resp(response)
        elif modal_name=='code_response':
            answer = self.get_answer_from_code_resp(response, dataset_name, model_name=model_name)
        else:
            answer = self.get_answer_from_eqn_resp(response)
        # end if
        return answer

    def get_majority_answer(
        self, 
        answers: List
    ):
        if not any(answers):
            return None
        # end if
        _answers = [a for a in answers if a is not None]
        if not any(_answers):
            return None
        # end if
        c = Counter(_answers)
        value_freqs = c.most_common()
        if len(set([v[1] for v in value_freqs]))==1:
            # in case of having no most frequent answer
            return answers[0]
        else:
            return value_freqs[0][0]
        # end if

    def get_openai_model_response(
        self,
        model_name: str,
        input_text: str,
        demo_str: str=None,
        temp: float=Macros.resp_temp_for_self_consistency,
        top_k: int=None
    ) -> str:
        if demo_str is not None:
            input_text = f"{demo_str}{input_text}"
        # end if
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    'role': 'user', 
                    'content': input_text
                }],
                top_p=1,
                n=top_k,
                temperature=temp,
                max_tokens=Macros.llm_resp_max_len
            )
            if top_k is None:
                return response.choices[0].message.content
            else:
                return [
                    r.message.content
                    for r in response.choices
                ]
            # end if
        except openai.BadRequestError as e:
            print(f"BadRequestError: {e}")
            if e.code=='context_length_exceeded':
                print(input_text)
            # end if
            pass
        # end try
        return

    def get_llm_response(
        self,
        llm_name: str,
        model_name: str,
        input_text: str,
        demo_str: str=None,
        temp: float=Macros.resp_temp_for_self_consistency,
        top_k: int=None
    ) -> str:
        responses = None
        if llm_name=='gpt3.5' or \
            llm_name=='gpt4' or \
            llm_name=='gpt4omini':
            responses = self.get_openai_model_response(
                model_name=model_name,
                input_text=input_text,
                demo_str=demo_str,
                temp=temp,
                top_k=top_k
            )
        elif llm_name=='palm':
            responses = PaLMModel.get_palm_model_response(
                model_name=model_name,
                input_text=input_text,
                demo_str=demo_str,
                temp=temp,
                top_k=top_k
            )
        # end if
        return responses

    def get_response_with_mutation(
        self, 
        llm_name: str,
        target_question: str,
        mutation: Dict,
        demo_str_over_modals: Dict,
        target_modality: str,
        temp: float=Macros.resp_temp,
        top_k: int=3
    ) -> Dict:
        # get responses from llm
        response_dict = dict()
        prompt, is_prompt_append = Macros.prompts_over_modals[target_modality]

        m_q = mutation['question']
        m_resp = mutation[target_modality]

        demo_prompt = demo_str_over_modals[target_modality]
        input_text = f"Q: {prompt} {target_question}\nA: "
        if is_prompt_append:
            input_text = f"Q: {target_question} {prompt}\nA: "
        # end if

        if llm_name=='gpt3.5':
            model_name = Macros.gpt3d5_engine_name 
        elif llm_name=='gpt4':
            model_name = Macros.gpt4_engine_name
        elif llm_name=='gpt4omini':
            model_name = Macros.gpt4omini_engine_name
        else:
            model_name = PaLMModel.engine_name
        # end if

        response = self.get_llm_response(
            llm_name,
            model_name,
            input_text,
            demo_str=demo_prompt,
            temp=temp,
            top_k=top_k
        )
        response_dict[target_modality] = response
        return response_dict

    def compute_self_consistency(
        self, 
        response_dict: Dict,
        dataset_name: str,
        target_modality: str,
        model_name: str
    ) -> float:
        return SelfConsistency.compute_self_consistency(
            response_dict, 
            dataset_name,
            target_modality,
            model_name=model_name
        )

    def select_mutations_by_self_consistency(
        self,
        target_question_response: Dict,
        mutations: List[Dict],
        prev_ans_dict: Dict,
        prev_consistency: float,
        prev_selected_mutations: List[Dict],
        llm_name: str,
        demo_str_over_modals: Dict,
        dataset_name: str,
        target_modality: str='cot_response',
        temp: float=Macros.resp_temp,
        top_k: int=Macros.self_consistency_top_k,
        max_self_const_val: float = None
    ) -> List[Dict]:
        '''
        for each mutation:
            1. append the each mutation to the target question.
            2. get the responses of the appended question over the modals.
            3. compute the modal consistency from the responses.
            4. find the mutation that increase the consistency the most
            5. return the mutation selected from step 4.
        '''
        # cons_orig, _ = self.compute_modal_consistency(target_question_response, dataset_name)
        prev_selected_mutated_questions = [
            m['mut']['question'] for m in prev_selected_mutations
        ]
        prev_ans_by_maj = self.get_majority_answer(prev_ans_dict[target_modality])
        positive_mutations = list()
        for m in mutations:
            m_q = m['question']

            if m_q not in prev_selected_mutated_questions:
                m_response_dict = self.get_response_with_mutation(
                    llm_name,
                    target_question_response['question'],
                    m,
                    demo_str_over_modals,
                    target_modality,
                    temp=temp,
                    top_k=top_k
                )
                m_cons, ans_dict = self.compute_self_consistency(
                    m_response_dict, 
                    dataset_name, 
                    target_modality,
                    model_name=llm_name if llm_name=='gpt4' else None
                )
                ans_by_maj = self.get_majority_answer(ans_dict[target_modality])
                if prev_consistency<m_cons:
                    # 1. if consistency gets better, then we take the mutation
                    selected_mut_dict = {
                        'mut': m,
                        'mut_response': m_response_dict,
                        'mut_consistency': m_cons,
                        'ans_dict': ans_dict
                    }
                    if selected_mut_dict not in prev_selected_mutations:
                        positive_mutations.append(selected_mut_dict)
                    # end if
                elif prev_consistency==m_cons and \
                    prev_ans_by_maj!=ans_by_maj:
                    # 2. if consistency is same as the previous one, 
                    # but the answer is different from previous one, 
                    # then, we take the mutation because of empirical observation 
                    # that the demonstration is likely to correct the answer for the math problem
                    selected_mut_dict = {
                        'mut': m,
                        'mut_response': m_response_dict,
                        'mut_consistency': m_cons,
                        'ans_dict': ans_dict
                    }
                    if selected_mut_dict not in prev_selected_mutations:
                        positive_mutations.append(selected_mut_dict)
                    # end if
                # end if
            # end if
        # end for

        if any(positive_mutations):
            # sort by consistency values
            positive_mutations = sorted(
                positive_mutations, 
                key=lambda x: x['mut_consistency'],
                reverse=True
            )
        # end if
        return positive_mutations

    def get_final_answer(
        self,
        target_question_response_dict: Dict[str, List],
        mutations: List[Dict],
        llm_name: str,
        dataset_name: str,
        target_modality: str='cot_response',
        max_num_muts: int = 5,
        temp: float = Macros.resp_temp,
        top_k: int = 3
    ):
        selected_mutations = list()
        prompt, is_prompt_append = self.get_target_prompt(target_modality)

        steps_for_finding_muts = 0
        max_steps_for_finding_muts = 3

        demo_str_over_modals = {
            target_modality: ''
        }
        cons_orig, final_ans_dict = self.compute_self_consistency(
            target_question_response_dict, 
            dataset_name,
            target_modality,
            model_name=llm_name if llm_name=='gpt4' else None
        )
        cons_final = cons_orig
        max_cons_val = 1.-(1./top_k)
        while steps_for_finding_muts<max_steps_for_finding_muts and \
            cons_final<max_cons_val:

            print(steps_for_finding_muts, cons_orig, cons_final, final_ans_dict, len(selected_mutations))
            if steps_for_finding_muts==0:
                cons_final = cons_orig
            # else:
            #     cons_final, ans_dict = self.compute_modal_consistency(target_question_response_dict, dataset_name)
            # end if
            
            positive_muts = self.select_mutations_by_self_consistency(
                target_question_response_dict,
                mutations,
                final_ans_dict,
                cons_final,
                selected_mutations,
                llm_name,
                demo_str_over_modals,
                dataset_name,
                target_modality=target_modality,
                temp=temp,
                top_k=top_k
            )
            if any(positive_muts) and len(selected_mutations)<max_num_muts:
                selected_mut = positive_muts[0]
                selected_m_q = selected_mut['mut']['question']
                selected_m_resp = selected_mut['mut'][target_modality]
                cons_final = selected_mut['mut_consistency']
                final_ans_dict = selected_mut['ans_dict']
                ans_by_maj = self.get_majority_answer(final_ans_dict[target_modality])

                if any(final_ans_dict[target_modality]) and \
                    ans_by_maj is not None and \
                    selected_m_resp is not None:
                    selected_mutations.append(selected_mut)
                    ind_final_ans = final_ans_dict[target_modality].index(ans_by_maj)
                    selected_m_resp = selected_mut['mut'][target_modality]

                    # update prompts over modalities with selected mutation
                    # update question
                    _demo_str = f"Q: {prompt} {selected_m_q}\nA: {selected_m_resp.strip()}\n"
                    if is_prompt_append:
                        _demo_str = f"Q: {selected_m_q} {prompt}\nA: {selected_m_resp.strip()}\n"
                    # end if
                    demo_str_over_modals[target_modality] += _demo_str
                    target_question_response_dict[target_modality] = selected_mut['mut_response'][target_modality]
                # end if
            else:
                break
            # end if
            steps_for_finding_muts += 1
        # end while
        return target_question_response_dict, selected_mutations, cons_orig, cons_final, final_ans_dict
    
    @classmethod
    def main(
        cls,
        dataset_name: str,
        llm_name: str,
        target_modality: str,
        temp: float = Macros.resp_temp,
        top_k: int = 3
    ) -> None:
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_dir = Macros.result_dir / 'genetic'/ dataset_name / 'evaluate_consistency' / llm_name
        genetic_dir.mkdir(parents=True, exist_ok=True)
        eval_res_files = sorted([
            f_name for f_name in os.listdir(str(eval_dir))
            if f_name.endswith('.json') and \
                f_name.startswith('eval-') and \
                (not f_name.startswith('eval-results-w'))
        ])
        f_index = 0
        result = dict()
        if os.path.exists(str(genetic_dir / f"final_answers_w_self_consistency_over_{target_modality}.json")):
            result = Utils.read_json(genetic_dir / f"final_answers_w_self_consistency_over_{target_modality}.json")
        # end if
        for eval_res_file in eval_res_files:
            gt_answer = Utils.read_json(eval_dir / eval_res_file)['orig']['answer']
            cksum_val = eval_res_file.split('-')[-1].split('.json')[0].strip()
            if cksum_val not in result.keys():
                print(f"{f_index} OUT OF {len(eval_res_files)}::{cksum_val}")
                obj = cls(eval_res_file, eval_dir)

                # get initial zero-shot response
                target_question = obj.eval_res['orig']['question']
                prompt, is_prompt_append = Macros.prompts_over_modals[target_modality]

                input_text = f"Q: {prompt} {target_question}\nA: "
                if is_prompt_append:
                    input_text = f"Q: {target_question} {prompt}\nA: "
                # end if
                if llm_name=='gpt3.5':
                    model_name = Macros.gpt3d5_engine_name 
                elif llm_name=='gpt4':
                    model_name = Macros.gpt4_engine_name
                elif llm_name=='gpt4omini':
                    model_name = Macros.gpt4_engine_name
                else:
                    model_name = PaLMModel.engine_name
                # end if
                # orig_response = obj.get_openai_model_response(
                #     input_text,
                #     demo_str=None,
                #     temp=temp,
                #     top_k=top_k
                # )
                orig_response = obj.get_llm_response(
                    llm_name,
                    model_name,
                    input_text,
                    demo_str=None,
                    temp=temp,
                    top_k=top_k
                )

                responses = {
                    'question': target_question,
                    target_modality: orig_response
                }

                final_response_dict, \
                selected_mutations, \
                orig_cons_score, \
                final_cons_score, \
                final_answer_dict = obj.get_final_answer(
                    responses,
                    obj.eval_res['mutation'],
                    llm_name,
                    dataset_name,
                    target_modality=target_modality,
                    temp=temp,
                    top_k=top_k
                )
                result[cksum_val] = {
                    'question': target_question,
                    'origal_response': orig_response,
                    'final_response': final_response_dict,
                    'fianl_answer': final_answer_dict,
                    'answer': gt_answer,
                    'original_consistency_score': orig_cons_score,
                    'final_consistency_score': final_cons_score,
                    'mutation': selected_mutations
                }
                Utils.write_json(
                    result,
                    genetic_dir / f"final_answers_w_self_consistency_over_{target_modality}.json",
                    pretty_format=True
                )
            # end if
            f_index += 1
        # end for
        Utils.write_json(
            result,
            genetic_dir / f"final_answers_w_self_consistency_over_{target_modality}.json",
            pretty_format=True
        )
        return