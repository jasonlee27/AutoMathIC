
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
from ..llmut.palm import PaLMModel
# from ..llmut.openai import OpenAiModel
from ..consistency.selfconsistency import SelfConsistency

from .multimodals import Multimodals

client = OpenAI()


class GeneticFineGrained:

    def __init__(
        self, 
        eval_file_name: str,
        eval_dir: Path,
        llm_name: str,
        cksum_val: str
    ):
        self.eval_file_name = eval_file_name
        self.eval_dir = eval_dir
        self.llm_name = llm_name

        if llm_name=='gpt3.5':
            self.model_name = Macros.gpt3d5_engine_name
        elif llm_name=='gpt4':
            self.model_name = Macros.gpt4_engine_name
        elif llm_name=='gpt4omini':
            self.model_name = Macros.gpt4omini_engine_name
        elif llm_name=='palm':
            self.model_name = PaLMModel.engine_name
        # end if

        self.cksum_val = cksum_val
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
        
        eval_res = Utils.read_json(self.eval_dir / self.eval_file_name)
        
        # fix the problem with formatting of eval_res
        # =====
        formatting_changes = False
        for mod_name in Macros.prompts_over_modals.keys():
            if type(eval_res['orig'][mod_name])==dict:
                eval_res['orig'][mod_name] = eval_res['orig'][mod_name]['msg']
                formatting_changes = True
            # end if
        # end for
        for m_i in range(len(eval_res['mutation'])):
            for mod_name in Macros.prompts_over_modals.keys():
                if type(eval_res['mutation'][m_i][mod_name])==dict:
                    eval_res['mutation'][m_i][mod_name] = eval_res['mutation'][m_i][mod_name]['msg']
                    formatting_changes = True
                # end if
            # end for
        # end for
        if formatting_changes:
            Utils.write_json(
                eval_res,
                self.eval_dir / self.eval_file_name,
                pretty_format=True
            )
        # end if
        # =====
        orig_res = eval_res['orig']

        # check if there are top-k responses
        if not os.path.exists(self.eval_dir / f"fg-eval-{self.cksum_val}.json"):
            for mod_name in Macros.prompts_over_modals.keys():
                prompt, is_prompt_append = Macros.prompts_over_modals[mod_name]

                q = orig_res['question']

                input_text = f"Q: {prompt} {q}\nA: "
                if is_prompt_append:
                    input_text = f"Q: {q} {prompt}\nA: "
                # end if
                eval_res['orig'][mod_name] = self.get_llm_response(
                    input_text,
                    top_k=Macros.self_consistency_top_k
                )
            # end for
            Utils.write_json(
                eval_res,
                self.eval_dir / f"fg-eval-{self.cksum_val}.json",
                pretty_format=True
            )
        else:
            eval_res = Utils.read_json(self.eval_dir / f"fg-eval-{self.cksum_val}.json")
        # end if
        return eval_res

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
                    answer = self.get_answer_from_cot_resp(resp)
                elif modal_name=='code_response':
                    answer = self.get_answer_from_code_resp(resp, dataset_name, model_name=model_name)
                else:
                    answer = self.get_answer_from_eqn_resp(resp)
                # end if
                if answer is not None:
                    answer_list.append(answer)
                # end if
            # end for
        else:
            if modal_name=='cot_response':
                answer = self.get_answer_from_cot_resp(response_list)
            elif modal_name=='code_response':
                answer = self.get_answer_from_code_resp(response_list, dataset_name, model_name=model_name)
            else:
                answer = self.get_answer_from_eqn_resp(response_list)
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

    def get_answer_by_cons_scores(
        self, 
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

    def get_llm_response(
        self,
        input_text,
        demo_prompt=None,
        top_k=Macros.self_consistency_top_k
    ) -> str:
        responses = None
        if self.llm_name=='gpt3.5' or \
            self.llm_name=='gpt4' or \
            self.llm_name=='gpt4omini':
            responses = self.get_openai_model_response(
                self.model_name,
                input_text,
                demo_str=demo_prompt,
                top_k=top_k
            )
        elif self.llm_name=='palm':
            responses = PaLMModel.get_palm_model_response(
                self.model_name,
                input_text,
                demo_str=demo_prompt,
                top_k=top_k
            )
        # end if
        return responses

    def get_openai_model_response(
        self,
        model_name: str,
        input_text: str,
        demo_str: str=None,
        temp: float=Macros.resp_temp_for_self_consistency,
        top_k: int=1
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
            if top_k==1:
                # logprobs = response.choices[0].logprobs.content
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
            response = self.get_llm_response(
                input_text,
                demo_prompt=demo_prompt,
                top_k=Macros.self_consistency_top_k
            )
            response_dict[mod_name] = response
        # end for
        return response_dict

    def compute_consistency(
        self,
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
                answer_maj, answer_freqs = self.get_answer_from_resp(
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

    def compute_mut_modal_consistency(
        self, 
        response_dict: Dict,
        dataset_name: str,
        model_name: str,
        answer_dict: Dict=None
    ) -> float:
        # modal_consistency = number of unique answers over the modals/number of modals(prompts)
        if answer_dict is None:
            answer_dict = dict()
            num_correct_modal = 0
            for modal_name in Macros.prompts_over_modals.keys():
                answer, answer_freqs = self.get_answer_from_resp(
                    response_dict[modal_name], 
                    dataset_name, 
                    modal_name, 
                    model_name=model_name
                )
                answer_dict[modal_name] = answer
            # end for
        # end if
        num_unique_answers = len(set(answer_dict.values()))
        num_all_answers = len(answer_dict.keys())
        modal_consistency = 1. - (num_all_answers-num_unique_answers)*1. / num_all_answers
        return modal_consistency, answer_dict

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
        prev_selected_mutated_questions = [
            m['mut']['question'] for m in prev_selected_mutations
        ]
        prev_ans_by_scores = self.get_answer_by_cons_scores(prev_ans_dict)
        positive_mutations = list()
        for m in mutations:
            m_q = m['question']
            answer_dict = None # m.get('ans_dict', None)
            if answer_dict is not None:
                m_cons_mut, _ = self.compute_mut_modal_consistency(
                    m, 
                    dataset_name,
                    model_name=llm_name if llm_name=='gpt4' else None,
                    answer_dict=answer_dict
                )
            else:
                m_cons_mut, answer_dict = self.compute_mut_modal_consistency(
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
                orig_cons_with_mut, \
                orig_ans_dict_with_mut, \
                cons_score_dict_over_answers = self.compute_consistency(
                    m_response_dict, 
                    dataset_name, 
                    model_name=llm_name if llm_name=='gpt4' else None
                )
                ans_by_scores = self.get_answer_by_cons_scores(cons_score_dict_over_answers)
                if prev_consistency<orig_cons_with_mut:
                    # 1. if consistency gets better, then we take the mutation
                    selected_mut_dict = {
                        'mut': m,
                        'mut_response': m_response_dict,
                        'mut_mod_consistency': m_cons_mut,
                        'mod_consistency_after_append_mut': orig_cons_with_mut,
                        'ans_dict': orig_ans_dict_with_mut,
                        'cons_score_dict_over_answers': cons_score_dict_over_answers
                    }
                    if selected_mut_dict not in prev_selected_mutations:
                        positive_mutations.append(selected_mut_dict)
                    # end if
                elif prev_consistency==orig_cons_with_mut and \
                    prev_ans_by_scores!=ans_by_scores:
                    # 2. if consistency is same as the previous one, 
                    # but the answer is different from previous one, 
                    # then, we take the mutation because of empirical observation 
                    # that the demonstration is likely to correct the answer for the math problem
                    selected_mut_dict = {
                        'mut': m,
                        'mut_response': m_response_dict,
                        'mut_mod_consistency': m_cons_mut,
                        'mod_consistency_after_append_mut': orig_cons_with_mut,
                        'ans_dict': orig_ans_dict_with_mut,
                        'cons_score_dict_over_answers': cons_score_dict_over_answers
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
        cons_orig, \
        final_ans_dict, \
        score_dict_over_unique_answers = self.compute_consistency(
            target_question_response_dict, 
            dataset_name,
            model_name=llm_name if llm_name=='gpt4' else None
        )
        cons_final = cons_orig
        cons_prev = cons_orig
        max_cons_val = 1.
        # max_mod_cons_val = 1.-(1./len(Macros.prompts_over_modals.keys()))
        max_mod_cons_val = 1.
        while steps_for_finding_muts<max_steps_for_finding_muts and \
            cons_final<max_cons_val:
            print(f"\tFinding Mutations...")
            # if steps_for_finding_muts==0:
            #     cons_final = cons_orig
            # else:
            #     cons_final, ans_dict = self.compute_consistency(target_question_response_dict, dataset_name)
            # end if
            
            positive_muts = self.select_mutations_by_modal_consistency(
                target_question_response_dict,
                mutations,
                score_dict_over_unique_answers,
                cons_final,
                selected_mutations,
                llm_name,
                demo_str_over_modals,
                dataset_name,
                target_response_type=target_response_type,
                max_mod_const_val=max_mod_cons_val
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

                # TODO: LLM (especially PaLM) can have no response e.g. 013d407 at GSM8K
                selected_m_resp = selected_mut['mut'][target_response_type] if selected_mut['mut'][target_response_type] is not None else ''
                cons_prev = cons_final
                cons_final = selected_mut['mod_consistency_after_append_mut']
                final_ans_dict = selected_mut['ans_dict']
                score_dict_over_unique_answers = selected_mut['cons_score_dict_over_answers']

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
                        _selected_m_resp = selected_mut['mut'][key] if selected_mut['mut'][key] is not None else ''
                        _demo_str = f"Q: {_prompt} {selected_m_q}\nA: {_selected_m_resp.strip()}\n"
                        if _is_prompt_append:
                            _demo_str = f"Q: {selected_m_q} {_prompt}\nA: {_selected_m_resp.strip()}\n"
                        # end if
                        demo_str_over_modals[key] += _demo_str
                    # end if
                    target_question_response_dict[key] = selected_mut['mut_response'][key]
                # end for
                print(f"\t----- Step{steps_for_finding_muts} -----")
                print(f"\t#muts_selected:{len(selected_mutations)}")
                print(f"\tcons_prev:{cons_prev}, cons_final:{cons_final}")
                print(f"\tscore_dict_over_unique_answers:{score_dict_over_unique_answers}")
            else:
                break
            # end if
            steps_for_finding_muts += 1
        # end while

        return target_question_response_dict, selected_mutations, cons_orig, cons_final, final_ans_dict, score_dict_over_unique_answers

    @classmethod
    def main(
        cls,
        dataset_name: str,
        llm_name: str
    ) -> None:
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_dir = Macros.result_dir / 'genetic_fg'/ dataset_name / 'evaluate_consistency' / llm_name
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
            if cksum_val not in result.keys() and f_index not in [842,845,967,968]:
                # f_index=842,845: openai.InternalServerError
                print(f"GeneticFineGrained::{f_index} OUT OF {len(eval_res_files)}::{cksum_val}")
                obj = cls(eval_res_file, eval_dir, llm_name, cksum_val)

                final_response_dict, \
                selected_mutations, \
                orig_cons_score, \
                final_cons_score, \
                final_answer_dict, \
                score_dict_over_unique_answers = obj.get_final_answer(
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
                    'mutation': selected_mutations,
                    'score_dict_over_unique_answers': score_dict_over_unique_answers
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


class GeneticFineGrainedWithValidationCls:

    def __init__(
        self,
        mut_dir: Path,
        eval_dir: Path,
        llm_name: str,
        cksum_val: str
    ):
        self.mut_dir = mut_dir
        self.eval_dir = eval_dir
        self.model_name = Macros.gpt3d5_engine_name if llm_name=='gpt3.5' else Macros.gpt4_engine_name
        self.cksum_val = cksum_val
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
        mut_nl_res = Utils.read_json(self.mut_dir / f"mut-nl-{self.cksum_val}.json")
        orig_res = mut_nl_res['orig']
        mut_res = mut_nl_res['mutations']
        eval_res = {
            'orig': dict(),
            'mutation': [dict() for _ in mut_res]
        }
        if os.path.exists(self.eval_dir / f"fg-w-val-cls-eval-{self.cksum_val}.json"):
            eval_res = Utils.read_json(self.eval_dir / f"fg-w-val-cls-eval-{self.cksum_val}.json")
        else:
            # write results of original questions
            q = orig_res['question']
            eval_res['orig']['question'] = orig_res['question']
            eval_res['orig']['answer'] = orig_res['answer']
            for mod_name in Macros.prompts_over_modals.keys():
                prompt, is_prompt_append = Macros.prompts_over_modals[mod_name]

                if eval_res['orig'].get(mod_name, None) is None:
                    input_text = f"Q: {prompt} {q}\nA: "
                    if is_prompt_append:
                        input_text = f"Q: {q} {prompt}\nA: "
                    # end if
                    
                    eval_res['orig'][mod_name] = self.get_openai_model_response(
                        self.model_name,
                        input_text,
                        top_k=Macros.self_consistency_top_k
                    )
                # end if
            # end for

            # write results of mutated questions
            for mut_i, mut in enumerate(mut_res):
                q = mut['question']
                eval_res['mutation'][mut_i]['question'] = mut['question']
                eval_res['mutation'][mut_i]['answer'] = mut_res[mut_i]['answer'] # must be null
                for mod_name in Macros.prompts_over_modals.keys():
                    prompt, is_prompt_append = Macros.prompts_over_modals[mod_name]

                    if eval_res['mutation'][mut_i].get(mod_name, None) is None:
                        input_text = f"Q: {prompt} {q}\nA: "
                        if is_prompt_append:
                            input_text = f"Q: {q} {prompt}\nA: "
                        # end if
                        eval_res['mutation'][mut_i][mod_name] = self.get_openai_model_response(
                            self.model_name,
                            input_text,
                            top_k=1
                        )
                    # end if
                # end for
            # end for
            Utils.write_json(
                eval_res,
                self.eval_dir / f"fg-w-val-cls-eval-{self.cksum_val}.json",
                pretty_format=True
            )
        # end if
        return eval_res

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
        response_list: List[str],
        dataset_name: str,
        modal_name: str,
        model_name: str=None
    ) -> Any:
        answer_list = list()
        if type(response_list) is not str:
            for resp in response_list:
                if modal_name=='cot_response':            
                    answer = self.get_answer_from_cot_resp(resp)
                elif modal_name=='code_response':
                    answer = self.get_answer_from_code_resp(resp, dataset_name, model_name=model_name)
                else:
                    answer = self.get_answer_from_eqn_resp(resp)
                # end if
                answer_list.append(answer)
            # end for
        else:
            if modal_name=='cot_response':
                answer = self.get_answer_from_cot_resp(response_list)
            elif modal_name=='code_response':
                answer = self.get_answer_from_code_resp(response_list, dataset_name, model_name=model_name)
            else:
                answer = self.get_answer_from_eqn_resp(response_list)
            # end if
            answer_list.append(answer)
        # end if
        c = Counter(answer_list)
        answer_freqs = c.most_common()
        answer = answer_freqs[0][0]
        return answer, answer_freqs

    def get_answer_by_cons_scores(
        self, 
        cons_score_dict_over_answers: Dict
    ):
        final_ans = list()
        max_cons_score = max(cons_score_dict_over_answers.values())
        for ans_per_mod, score in cons_score_dict_over_answers.items():
            if score==max_cons_score:
                final_ans.append(ans_per_mod)
            # end if
        # end for
        return final_ans[0]

    def get_openai_model_response(
        self,
        model_name: str,
        input_text: str,
        demo_str: str=None,
        temp: float=Macros.resp_temp_for_self_consistency,
        top_k: int=1
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
            if top_k==1:
                # logprobs = response.choices[0].logprobs.content
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
                demo_str=demo_prompt,
                top_k=Macros.self_consistency_top_k
            )
            response_dict[mod_name] = response
        # end for
        return response_dict

    def compute_consistency(
        self, 
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
                answer_maj, answer_freqs = self.get_answer_from_resp(
                    resp_list, 
                    dataset_name, 
                    modal_name, 
                    model_name=model_name
                )
                self_consistency_score = answer_freqs[0][1]*1./len(resp_list)
                # print(len(resp_list), answer_freqs)
                answer_dict[modal_name] = {
                    'answer': answer_maj,
                    'self-consistency': self_consistency_score
                }
                if answer_maj not in score_dict_over_unique_answers.keys():
                    score_dict_over_unique_answers[answer_dict[modal_name]['answer']] = modal_weights[modal_name]*self_consistency_score
                else:
                    score_dict_over_unique_answers[answer_dict[modal_name]['answer']] += modal_weights[modal_name]*self_consistency_score
                # end if
            # end for
        else:
            for modal_name in Macros.prompts_over_modals.keys():
                answer_maj = answer_dict[modal_name]
                if answer_maj not in score_dict_over_unique_answers.keys():
                    score_dict_over_unique_answers[answer_dict[modal_name]['answer']] = modal_weights[modal_name]*answer_dict[modal_name]['self-consistency']
                else:
                    score_dict_over_unique_answers[answer_dict[modal_name]['answer']] += modal_weights[modal_name]*answer_dict[modal_name]['self-consistency']
                # end if
            # end for
        # end if

        # Normalizing scores
        sum_scores = sum([
            score_dict_over_unique_answers[answer_per_modal]
            for answer_per_modal in score_dict_over_unique_answers.keys()
        ])
        for answer_per_modal in score_dict_over_unique_answers.keys():
            score_dict_over_unique_answers[answer_per_modal] = score_dict_over_unique_answers[answer_per_modal]/sum_scores
        # end for

        max_score = max(score_dict_over_unique_answers.values())
        return max_score, answer_dict, score_dict_over_unique_answers

    def compute_mut_modal_consistency(
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
                answer, answer_freqs = self.get_answer_from_resp(
                    response_dict[modal_name], 
                    dataset_name, 
                    modal_name, 
                    model_name=model_name
                )
                answer_dict[modal_name] = answer
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
        prev_selected_mutated_questions = [
            m['mut']['question'] for m in prev_selected_mutations
        ]
        prev_ans_by_scores = self.get_answer_by_cons_scores(prev_ans_dict)
        positive_mutations = list()
        for m in mutations:
            m_q = m['question']
            answer_dict = None # m.get('ans_dict', None)
            if answer_dict is not None:
                m_cons_mut, _ = self.compute_mut_modal_consistency(
                    m, 
                    dataset_name,
                    model_name=llm_name if llm_name=='gpt4' else None,
                    answer_dict=answer_dict
                )
            else:
                m_cons_mut, answer_dict = self.compute_mut_modal_consistency(
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
                orig_cons_with_mut, \
                orig_ans_dict_with_mut, \
                cons_score_dict_over_answers = self.compute_consistency(
                    m_response_dict, 
                    dataset_name, 
                    model_name=llm_name if llm_name=='gpt4' else None
                )
                ans_by_scores = self.get_answer_by_cons_scores(cons_score_dict_over_answers)
                if prev_consistency<orig_cons_with_mut:
                    # 1. if consistency gets better, then we take the mutation
                    selected_mut_dict = {
                        'mut': m,
                        'mut_response': m_response_dict,
                        'mut_mod_consistency': m_cons_mut,
                        'mod_consistency_after_append_mut': orig_cons_with_mut,
                        'ans_dict': orig_ans_dict_with_mut,
                        'cons_score_dict_over_answers': cons_score_dict_over_answers
                    }
                    if selected_mut_dict not in prev_selected_mutations:
                        positive_mutations.append(selected_mut_dict)
                    # end if
                elif prev_consistency==orig_cons_with_mut and \
                    prev_ans_by_scores!=ans_by_scores:
                    # 2. if consistency is same as the previous one, 
                    # but the answer is different from previous one, 
                    # then, we take the mutation because of empirical observation 
                    # that the demonstration is likely to correct the answer for the math problem
                    selected_mut_dict = {
                        'mut': m,
                        'mut_response': m_response_dict,
                        'mut_mod_consistency': m_cons_mut,
                        'mod_consistency_after_append_mut': orig_cons_with_mut,
                        'ans_dict': orig_ans_dict_with_mut,
                        'cons_score_dict_over_answers': cons_score_dict_over_answers
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
        cons_orig, final_ans_dict, score_dict_over_unique_answers = self.compute_consistency(
            target_question_response_dict, 
            dataset_name,
            model_name=llm_name if llm_name=='gpt4' else None
        )
        cons_final = cons_orig
        max_cons_val = 1.
        max_mod_cons_val = 1.-(1./len(Macros.prompts_over_modals.keys()))
        while steps_for_finding_muts<max_steps_for_finding_muts and \
            cons_final<max_cons_val:

            print(steps_for_finding_muts, cons_orig, cons_final, len(selected_mutations), final_ans_dict)

            if steps_for_finding_muts==0:
                cons_final = cons_orig
            # else:
            #     cons_final, ans_dict = self.compute_consistency(target_question_response_dict, dataset_name)
            # end if
            
            positive_muts = self.select_mutations_by_modal_consistency(
                target_question_response_dict,
                mutations,
                score_dict_over_unique_answers,
                cons_final,
                selected_mutations,
                llm_name,
                demo_str_over_modals,
                dataset_name,
                target_response_type=target_response_type,
                max_mod_const_val=max_mod_cons_val
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
                score_dict_over_unique_answers = selected_mut['cons_score_dict_over_answers']

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

        return target_question_response_dict, selected_mutations, cons_orig, cons_final, final_ans_dict, score_dict_over_unique_answers

    @classmethod
    def main(
        cls,
        dataset_name: str,
        llm_name: str
    ) -> None:
        mut_dir = Macros.result_dir / 'nl2nl' / dataset_name / 'mutation_with_validation_cls' / llm_name
        eval_dir = Macros.result_dir / 'nl2nl'/ dataset_name / 'evaluate_consistency' / llm_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        genetic_dir = Macros.result_dir / 'genetic_fg_with_validation_cls'/ dataset_name / 'evaluate_consistency' / llm_name
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
                print(f"GeneticFineGrainedWithValidationCls::{f_index} OUT OF {len(eval_res_files)}::{cksum_val}")
                obj = cls(
                    mut_dir,
                    eval_dir,
                    llm_name,
                    cksum_val
                )
                final_response_dict, \
                selected_mutations, \
                orig_cons_score, \
                final_cons_score, \
                final_answer_dict, \
                score_dict_over_unique_answers = obj.get_final_answer(
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
                    'mutation': selected_mutations,
                    'score_dict_over_unique_answers': score_dict_over_unique_answers
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

