
import os
import re
import random
import subprocess

from typing import *
from pathlib import Path

from .validation import MutationValidataionCls

from ..dataset.asdiv import Asdiv
from ..dataset.gsm8k import Gsm8k
from ..dataset.svamp import Svamp
from ..dataset.multiarith import MultiArith
from ..dataset.addsub import Addsub
from ..dataset.singleeq import SingleEq
from ..prog_synthesis.varfinder import VarFinder
from ..llmut.llama_model import LlamaModel
from ..llmut.evaluate import EvaluateWithMultimodals

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


NUMBER_IN_ENGLISH = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13, 
    'fourteen': 14, 
    'fifteen': 15,
    'sixteen': 16, 
    'seventeen': 17, 
    'eighteen': 18, 
    'nineteen': 19,
    'twenty': 20, 
    'thirty': 30, 
    'forty': 40, 
    'fifty': 50, 
    'sixty': 60, 
    'seventy': 70, 
    'eighty': 80, 
    'ninety': 90,
    'hundred': 100, 
    'thousand': 1000, 
    'million': 1000000, 
    'billion': 1000000000, 
    'trillion': 1000000000000
}

class Mutate2nl:

    def __init__(
        self, 
        cksum_val: str,
        dataset_name: str,
        pl_type='python'
    ):
        self.cksum_val: str = cksum_val
        self.dataset_name = dataset_name
        self.pl_type: str = pl_type
        self.mut_dir: Path = Macros.result_dir / 'eq2code' / dataset_name / pl_type / 'mutation'
        self.orig_data: Dict = self.find_target_orig_data()
        self.orig_vars = Utils.read_json(self.mut_dir.parent / f"vars-{cksum_val}.json")
        self.mut_res: List[Dict] = self.read_mut_res()
        
    def read_mut_res(self) -> List[Dict]:
        '''
        mut_res_file: List[Dict]
        each dict in the list is a mutation result
        the mutation result consists of "code" and "var"
        "code" is the mutated code
        "var" is the variables mutated in the "code"
        ''' 
        mut_res_file = self.mut_dir / f"mut-{self.cksum_val}.json"
        return Utils.read_json(mut_res_file)

    def read_dataset(self) -> List[Dict]:
        dataset_obj = None
        if self.dataset_name=='asdiv':
            dataset_obj = Asdiv()
        elif self.dataset_name=='gsm8k':
            dataset_obj = Gsm8k()
        elif self.dataset_name=='svamp':
            dataset_obj = Svamp()
        elif self.dataset_name=='multiarith':
            dataset_obj = MultiArith()
        elif self.dataset_name=='addsub':
            dataset_obj = Addsub()
        elif self.dataset_name=='singleeq':
            dataset_obj = SingleEq()
        # end if
        return dataset_obj

    def find_target_orig_data(self) -> Dict:
        orig_data: List[Dict] = self.read_dataset()
        for d in orig_data:
            _cksum_val = Utils.get_cksum(d['id'], length=7)
            if self.cksum_val==_cksum_val:
                return d
            # end if
        # end for
        return

    def generate_mutated_nl_questions(self) -> List[Dict]:
        mut_nl_questions = list()
        for mut in self.mut_res:
            mut_nl_q = self.generate_nl_question_per_mutation(mut)
            mut_nl_questions.append(mut_nl_q)
        # end for
        return mut_nl_questions

    def generate_nl_q_by_replacing_values(
        self, 
        nl_q: str,
        noun_phrase: str, 
        orig_val: str, 
        mut_val: str
    ) -> str:
        mut_noun_phrase = noun_phrase.replace(orig_val, mut_val)
        # mut_nl_q = nl_q.replace(noun_phrase, mut_noun_phrase)
        _noun_phrase = noun_phrase.replace('$', '\$') # ([^\d])\$ 3
        _mut_noun_phrase = mut_noun_phrase.replace('$', '\$')
        pat1 = r'([^\d])'+_noun_phrase+r'([^\d])'
        pat2 = r'^'+_noun_phrase+r'([^\d])'
        pat3 = r'([^\d])'+_noun_phrase+r'$'
        char_search1 = re.search(pat1, nl_q)
        char_search2 = re.search(pat2, nl_q)
        char_search3 = re.search(pat3, nl_q)

        if char_search1 is not None:
            temp_char_before = char_search1.group(1)
            temp_char_after = char_search1.group(2)
            pat_from = temp_char_before+_noun_phrase+temp_char_after
            pat_to = temp_char_before+mut_noun_phrase+temp_char_after
        elif char_search2 is not None:
            temp_char_before = r'^'
            temp_char_after = char_search2.group(1)
            pat_from = temp_char_before+_noun_phrase+temp_char_after
            pat_to = mut_noun_phrase+temp_char_after
        else: # char_search3 is not None
            temp_char_before = char_search3.group(1)
            temp_char_after = r'$'
            pat_from = temp_char_before+_noun_phrase+temp_char_after
            pat_to = temp_char_before+mut_noun_phrase
        # end if
        
        # for regex sub
        pat_from = pat_from.replace('.', '\.')
        # pat_to = pat_to.replace(',', '\,')
        
        mut_nl_q = re.sub(pat_from, pat_to, nl_q)
        return mut_nl_q

    def generate_nl_question_per_mutation(
        self,
        mutation_dict: Dict
    ) -> Dict:
        mut_code = mutation_dict['code']
        mut_vars = mutation_dict['var']
        orig_nl_q = f"{self.orig_data['body']} {self.orig_data['question']}"
        mut_nl_q = orig_nl_q.lower()
        mut_ans = mutation_dict['answer']
        for var_name in mut_vars.keys():
            _mut_nl_q = mut_nl_q
            # for each variable, convert variable name to noun phrase
            noun_phrase = VarFinder.convert_snake_case_var_to_name(var_name)
            orig_val = self.orig_vars[var_name]

            # 1.replace original value with mutated value
            # 2.replace original noun phrase with mutated noun phrase in the nl question
            mut_nl_q = self.generate_nl_q_by_replacing_values(
                mut_nl_q,
                noun_phrase,
                str(orig_val), 
                str(mut_vars[var_name])
            )
        # end for
        if mut_nl_q==orig_nl_q:
            print(mut_nl_q)
            print()
            print(mutation_dict['var'])
            print()
            raise()
        # end if
        return {
            'question': mut_nl_q,
            'answer': mut_ans,
            'code': mut_code,
            'var': mut_vars
        }

    def write_nl(self, mut_nls) -> None:
        mut_res = list()
        # self.mut_dir.mkdir(parents=True, exist_ok=True)
        orig_nl_q = f"{self.orig_data['body']} {self.orig_data['question']}"
        orig_nl_ans = self.orig_data['answer']
        if any(mut_nls):
            res = {
                'orig': {
                    'question': orig_nl_q,
                    'answer': orig_nl_ans
                },
                'mutations': mut_nls
            }
            Utils.write_json(
                res, 
                self.mut_dir / f"mut-nl-{self.cksum_val}.json"
            )
        # end if
        return

    @classmethod
    def mutate(
        cls,
        dataset_name: str,
        pl_type: str='python'
    ) -> None:
        res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        all_orig_codes = [
            f for f in os.listdir(str(res_dir))
            if f.endswith('.py')
        ]
        for orig_code in all_orig_codes:
            cksum_val = orig_code.split('.py')[0]
            if cksum_val is not None:
                print(cksum_val)
                mut_nl_obj = cls(
                    cksum_val=cksum_val,
                    dataset_name=dataset_name,
                    pl_type=pl_type
                )
                mut_nl_res = mut_nl_obj.generate_mutated_nl_questions()
                mut_nl_obj.write_nl(mut_nl_res)
        # end for
        return


class Mutate2nlWoEquation:

    def __init__(
        self,
        dataset_name: str,
    ):
        self.dataset_name = dataset_name
        self.mut_dir: Path = Macros.result_dir / 'nl2nl' / dataset_name / 'mutation'
        self.mut_dir.mkdir(parents=True, exist_ok=True)
        self.orig_data: Lict[Dict] = self.read_dataset()

    def read_dataset(self) -> List[Dict]:
        dataset_obj = None
        if self.dataset_name=='asdiv':
            dataset_obj = Asdiv()
        elif self.dataset_name=='gsm8k':
            dataset_obj = Gsm8k()
        elif self.dataset_name=='svamp':
            dataset_obj = Svamp()
        elif self.dataset_name=='multiarith':
            dataset_obj = MultiArith()
        elif self.dataset_name=='addsub':
            dataset_obj = Addsub()
        elif self.dataset_name=='singleeq':
            dataset_obj = SingleEq()
        # end if
        return dataset_obj

    def find_target_orig_data(self, cksum_val: str) -> Dict:
        for d in self.orig_data:
            _cksum_val = Utils.get_cksum(d['id'], length=7)
            if self.cksum_val==_cksum_val:
                return d
            # end if
        # end for
        return

    def convert_string_to_number(self, str_num: str) -> bool:
        try:
            num_complex = complex(str_num)
            return num_complex.real
        except ValueError:
            return NUMBER_IN_ENGLISH.get(str_num.lower(), None)
        # end try
        return 

    def generate_mutated_nl_questions(
        self, 
        data: Dict,
        res_file: Path,
        max_bound: Any,
        min_bound: Any,
        num_mutations: int=Macros.num_mutations,
        num_max_iter: int=Macros.num_max_mutation_iter
    ) -> List[str]:
        res = Utils.read_json(res_file)
        mutations = list()
        if res is not None:
            mutations = [
                r['question'] for r in res['mutations']
            ]
        # end if
        num_iter = 0
        while len(mutations)<num_mutations and num_iter<num_max_iter:
            m = self.generate_mutated_nl_question_per_mutation(
                data, max_bound, min_bound
            )
            if m not in mutations:
                mutations.append(m)
            else:
                tokens = Utils.tokenize(f"{data['body']} {data['question']}")
                orig_nl_q = Utils.detokenize(tokens)
                if orig_nl_q==m:
                    break
                # end if
            # end if
            num_iter += 1
        # end for
        return mutations

    def generate_mutated_nl_question_per_mutation(
        self, 
        data: Dict,
        max_bound: Any,
        min_bound: Any,
    ) -> List[str]:
        tokens, val_token_inds = self.generate_tokens_w_values(data)
        for t_i in val_token_inds:
            val = tokens[t_i]
            if val==int(val):
                # if val is integer
                tokens[t_i] = str(random.randint(min_bound, max_bound))
            else:
                # if val is float
                whole = random.randint(min_bound, max_bound-1)
                fraction = random.random() # sample random number between 0 and 1
                tokens[t_i] = str(whole*1.+fraction)
            # end if
        # end for
        return Utils.detokenize(tokens)
    
    def generate_tokens_w_values(
        self, 
        data: Dict,
    ) -> List[str]:
        tokens_w_vals = list()
        val_token_inds = list()
        orig_nl_q = f"{data['body']} {data['question']}"
        tokens = Utils.tokenize(orig_nl_q)
        for t_i, t in enumerate(tokens):
            t_val = self.convert_string_to_number(t)
            if t_val is not None:
                tokens_w_vals.append(t_val)
                val_token_inds.append(t_i)
            else:
                tokens_w_vals.append(t)
            # end if
        # end for
        return tokens_w_vals, val_token_inds

    def write_nl(
        self, 
        orig_data: Dict,
        mut_nls: List[str], 
        cksum_val: str,
        res_file: Path
    ) -> None:
        mut_res = list()
        # self.mut_dir.mkdir(parents=True, exist_ok=True)
        orig_nl_q = f"{orig_data['body']} {orig_data['question']}"
        orig_nl_ans = orig_data['answer']
        _mut_nls = list()
        if len(set(mut_nls))>0:
            for m in list(set(mut_nls)):
                if m!=orig_nl_q:
                    _mut_nls.append(m)
                # end if
            # end for
        # end if
        if any(_mut_nls):
            mut_nls = _mut_nls
        # end if

        if any(mut_nls):
            res = {
                'orig': {
                    'question': orig_nl_q,
                    'answer': orig_nl_ans
                },
                'mutations': [
                    {
                        'question': m,
                        'answer': None
                    } for m in mut_nls
                ]
            }
            Utils.write_json(
                res, 
                res_file,
                pretty_format=True
            )
        elif not any(mut_nls) and os.path.exists(str(res_file)):
            print(f"{cksum_val}::NO_MUTATIONS")
            os.remove(str(res_file))
        else:
            print(f"{cksum_val}::NO_MUTATIONS")
        # end if
        return

    @classmethod
    def mutate(
        cls,
        dataset_name: str,
    ) -> None:
        mut_nl_obj = cls(dataset_name=dataset_name)
        i = 0
        for d in mut_nl_obj.orig_data:
            cksum_val = Utils.get_cksum(str(d['id']), length=7)
            res_file = mut_nl_obj.mut_dir / f"mut-nl-{cksum_val}.json"
            if not os.path.exists(str(res_file)):
                print(f"Mutate2nlWoEquation.mutate::{cksum_val}")
                mut_nl_res = mut_nl_obj.generate_mutated_nl_questions(
                    d,
                    res_file,
                    max_bound=100,
                    min_bound=0,
                    num_mutations=Macros.num_mutations
                )
                mut_nl_obj.write_nl(d, mut_nl_res, cksum_val, res_file)
            # end if
            i += 1
        # end for
        return


class Mutate2nlWoEquationNModConsistency:

    def __init__(
        self,
        dataset_name: str,
    ):
        self.dataset_name = dataset_name
        self.mut_dir: Path = Macros.result_dir / 'nl2nl' / dataset_name / 'mutation'
        self.mut_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_over_modals = Macros.prompts_over_modals
        self.orig_data: Lict[Dict] = self.read_dataset()

    def read_dataset(self) -> List[Dict]:
        dataset_obj = None
        if self.dataset_name=='asdiv':
            dataset_obj = Asdiv()
        elif self.dataset_name=='gsm8k':
            dataset_obj = Gsm8k()
        elif self.dataset_name=='svamp':
            dataset_obj = Svamp()
        elif self.dataset_name=='multiarith':
            dataset_obj = MultiArith()
        elif self.dataset_name=='addsub':
            dataset_obj = Addsub()
        # end if
        return dataset_obj

    def find_target_orig_data(self, cksum_val: str) -> Dict:
        for d in self.orig_data:
            _cksum_val = Utils.get_cksum(d['id'], length=7)
            if self.cksum_val==_cksum_val:
                return d
            # end if
        # end for
        return

    def convert_string_to_number(self, str_num: str) -> bool:
        try:
            num_complex = complex(str_num)
            return num_complex.real
        except ValueError:
            return NUMBER_IN_ENGLISH.get(str_num.lower(), None)
        # end try
        return 

    @classmethod
    def get_openai_model_response(
        self,
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
    
    def get_response_over_modals(
        self, 
        llm_name: str, 
        input_text: str,
        llm_generator: Any=None
    ) -> Dict:
        resp_over_modals = dict()
        for mod_name in self.prompts_over_modals.keys():
            if llm_name==Macros.gpt3d5_engine_name or \
                llm_name==Macros.gpt4_engine_name:
                resp_per_mod = self.get_openai_model_response(
                    llm_name,
                    input_text,
                    prompt=self.prompts_over_modals[mod_name][0],
                    prompt_append=self.prompts_over_modals[mod_name][1]
                )
                resp_over_modals[mod_name] = resp_per_mod
            elif (llm_name==Macros.llama_model_name) and \
                (llm_generator is not None):
                resp_per_mod = LlamaModel.predict(
                    llm_generator,
                    input_text,
                    prompt=self.prompts_over_modals[mod_name][0],
                    prompt_append=self.prompts_over_modals[mod_name][1]
                )
                resp_over_modals[mod_name] = resp_per_mod[0]
            # end if
            
        # end for
        return resp_over_modals

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
    
    def compute_modal_consistency(
        self, 
        response_dict: Dict,
        dataset_name: str,
        model_name: str
    ) -> float:
        # modal_consistency = number of unique answers over the modals/number of modals(prompts)
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

    def generate_mutated_nl_questions(
        self, 
        data: Dict,
        res_file: Path,
        max_bound: Any,
        min_bound: Any,
        llm_name: str,
        dataset_name: str,
        num_mutations: int=Macros.num_mutations,
        num_max_iter: int=Macros.num_max_mutation_iter,
        llm_generator: Any=None
    ) -> List[str]:
        res = Utils.read_json(res_file)
        mutations = list()
        mut_questions = list()
        if res is not None:
            mut_questions = [
                r['mut_question'] for r in res['mutations']
            ]
        # end if
        num_iter = 0
        while len(mutations)<num_mutations and num_iter<num_max_iter:
            m = self.generate_mutated_nl_question_per_mutation(
                data, max_bound, min_bound
            )
            if m not in mut_questions:
                # get response of LLM for the mutated questions and see if they are consistent
                resp_over_modals = self.get_response_over_modals(
                    llm_name, 
                    m, 
                    llm_generator=llm_generator
                )
                mod_consistent, ans_dict = self.compute_modal_consistency(
                    resp_over_modals,
                    dataset_name,
                    llm_name
                )

                # if consistent, append the mutation and its responses
                if mod_consistent:
                    mutations.append({
                        'mut_question': m.strip(),
                        'responses': resp_over_modals,
                        'answer': ans_dict
                    })
                    mut_questions.append(m.strip())
                # end if
            else:
                tokens = Utils.tokenize(f"{data['body']} {data['question']}")
                orig_nl_q = Utils.detokenize(tokens)
                if orig_nl_q==m or orig_nl_q==m.strip():
                    break
                # end if
            # end if
            print('.', end='')
            num_iter += 1
        # end for
        print()
        return mutations

    def generate_mutated_nl_question_per_mutation(
        self, 
        data: Dict,
        max_bound: Any,
        min_bound: Any,
    ) -> List[str]:
        tokens, val_token_inds = self.generate_tokens_w_values(data)
        for t_i in val_token_inds:
            val = tokens[t_i]
            if val==int(val):
                # if val is integer
                tokens[t_i] = str(random.randint(min_bound, max_bound))
            else:
                # if val is float
                whole = random.randint(min_bound, max_bound-1)
                fraction = random.random() # sample random number between 0 and 1
                tokens[t_i] = str(whole*1.+fraction)
            # end if
        # end for
        return Utils.detokenize(tokens)
    
    def generate_tokens_w_values(
        self, 
        data: Dict,
    ) -> List[str]:
        tokens_w_vals = list()
        val_token_inds = list()
        orig_nl_q = f"{data['body']} {data['question']}"
        tokens = Utils.tokenize(orig_nl_q)
        for t_i, t in enumerate(tokens):
            t_val = self.convert_string_to_number(t)
            if t_val is not None:
                tokens_w_vals.append(t_val)
                val_token_inds.append(t_i)
            else:
                tokens_w_vals.append(t)
            # end if
        # end for
        return tokens_w_vals, val_token_inds

    def write_nl(
        self, 
        orig_data: Dict,
        mut_nls: List[str], 
        cksum_val: str,
        res_file: Path
    ) -> None:
        mut_res = list()
        # self.mut_dir.mkdir(parents=True, exist_ok=True)
        orig_nl_q = f"{orig_data['body']} {orig_data['question']}"
        orig_nl_ans = orig_data['answer']
        _mut_nls = list()
        if len(mut_nls)>0:
            for m in list(mut_nls):
                if m['mut_question']!=orig_nl_q:
                    _mut_nls.append(m)
                # end if
            # end for
        # end if
        if any(_mut_nls):
            mut_nls = _mut_nls
        # end if

        if any(mut_nls):
            res = {
                'orig': {
                    'question': orig_nl_q,
                    'answer': orig_nl_ans
                },
                'mutations': mut_nls
            }
            Utils.write_json(
                res, 
                res_file,
                pretty_format=True
            )
            print()
        elif not any(mut_nls) and os.path.exists(str(res_file)):
            print(f"NO_MUTATIONS")
            os.remove(str(res_file))
        else:
            print(f"NO_MUTATIONS")
        # end if
        return

    @classmethod
    def mutate(
        cls,
        dataset_name: str,
        llm_name: str
    ) -> None:
        mut_nl_obj = cls(dataset_name=dataset_name)
        generator = None
        if llm_name!=Macros.gpt3d5_engine_name and \
            llm_name!=Macros.gpt4_engine_name:
            if llm_name=='llama':
                generator = LlamaModel.load_model(
                    ckpt_dir = Macros.llama_model_dir,
                    tokenizer_path = Macros.llama_tokenizer_path
                )
            # end if
        # end if
        data_size = len(mut_nl_obj.orig_data)
        d_i = 1
        for d in sorted(mut_nl_obj.orig_data, key=lambda x: x['id']):
            cksum_val = Utils.get_cksum(str(d['id']), length=7)
            res_file = mut_nl_obj.mut_dir / f"mut-nl-{cksum_val}.json"
            if not os.path.exists(str(res_file)):
                print(f"Mutate2nlWoEquationNModConsistency::{d_i}_OUT_OF_{data_size}::{cksum_val}", end='')
                mut_nl_res = mut_nl_obj.generate_mutated_nl_questions(
                    d,
                    res_file,
                    dataset_name=dataset_name,
                    max_bound=100,
                    min_bound=0,
                    num_mutations=Macros.num_mutations,
                    llm_name=llm_name,
                    llm_generator=generator
                )
                mut_nl_obj.write_nl(d, mut_nl_res, cksum_val, res_file)
            # end if
            d_i += 1
        # end for
        return


class Mutate2nlWoEquationNValidationCls:

    def __init__(
        self,
        dataset_name: str,
        model_name: str
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.mut_dir: Path = Macros.result_dir / 'nl2nl' / dataset_name / 'mutation_with_validation_cls' / model_name
        self.mut_dir.mkdir(parents=True, exist_ok=True)
        self.orig_data: Lict[Dict] = self.read_dataset()

    def read_dataset(self) -> List[Dict]:
        dataset_obj = None
        if self.dataset_name=='asdiv':
            dataset_obj = Asdiv()
        elif self.dataset_name=='gsm8k':
            dataset_obj = Gsm8k()
        elif self.dataset_name=='svamp':
            dataset_obj = Svamp()
        elif self.dataset_name=='multiarith':
            dataset_obj = MultiArith()
        elif self.dataset_name=='addsub':
            dataset_obj = Addsub()
        elif self.dataset_name=='singleeq':
            dataset_obj = SingleEq()
        # end if
        return dataset_obj

    def find_target_orig_data(self, cksum_val: str) -> Dict:
        for d in self.orig_data:
            _cksum_val = Utils.get_cksum(d['id'], length=7)
            if self.cksum_val==_cksum_val:
                return d
            # end if
        # end for
        return

    def convert_string_to_number(self, str_num: str) -> bool:
        try:
            num_complex = complex(str_num)
            return num_complex.real
        except ValueError:
            return NUMBER_IN_ENGLISH.get(str_num.lower(), None)
        # end try
        return 

    def generate_mutated_nl_questions(
        self, 
        data: Dict,
        res_file: Path,
        max_bound: Any,
        min_bound: Any,
        num_mutations: int=Macros.num_mutations,
        num_max_iter: int=Macros.num_max_mutation_iter
    ) -> List[str]:
        res = Utils.read_json(res_file)
        mutations = list()
        mutations_not_validated = list()
        if res is not None:
            mutations = [
                r['question'] for r in res['mutations']
            ]
        # end if
        tokens = Utils.tokenize(f"{data['body']} {data['question']}")
        orig_nl_q = Utils.detokenize(tokens)
        num_iter = 0
        while len(mutations)<num_mutations and num_iter<num_max_iter:
            m = self.generate_mutated_nl_question_per_mutation(
                data, max_bound, min_bound
            )
            # validate if the mutated question is reasonable in real world
            is_mut_reasonable = MutationValidataionCls.mutation_validataion(
                m,
                self.model_name
            )
            if is_mut_reasonable and orig_nl_q!=m:
                if m not in mutations:
                    mutations.append(m)
                # else:
                #     tokens = Utils.tokenize(f"{data['body']} {data['question']}")
                #     orig_nl_q = Utils.detokenize(tokens)
                #     if orig_nl_q==m:
                #         break
                #     # end if
                # end if
            elif not is_mut_reasonable and orig_nl_q!=m:
                mutations_not_validated.append(m)
            # end if
            num_iter += 1
        # end for
        if len(mutations)<num_mutations and any(mutations_not_validated):
            num_muts_needed_to_fill = num_mutations-len(mutations)
            mutations_not_validated_sample = random.sample(
                mutations_not_validated, 
                num_muts_needed_to_fill
            )
            mutations.extend(mutations_not_validated_sample)
        # end if
        return mutations

    def generate_mutated_nl_question_per_mutation(
        self, 
        data: Dict,
        max_bound: Any,
        min_bound: Any,
    ) -> List[str]:
        tokens, val_token_inds = self.generate_tokens_w_values(data)
        for t_i in val_token_inds:
            val = tokens[t_i]
            if val==int(val):
                # if val is integer
                tokens[t_i] = str(random.randint(min_bound, max_bound))
            else:
                # if val is float
                whole = random.randint(min_bound, max_bound-1)
                fraction = round(random.random(), 2) # sample random number between 0 and 1 with two decimals
                tokens[t_i] = str(whole*1.+fraction)
            # end if
        # end for
        return Utils.detokenize(tokens)
    
    def generate_tokens_w_values(
        self, 
        data: Dict,
    ) -> List[str]:
        tokens_w_vals = list()
        val_token_inds = list()
        orig_nl_q = f"{data['body']} {data['question']}"
        tokens = Utils.tokenize(orig_nl_q)
        for t_i, t in enumerate(tokens):
            t_val = self.convert_string_to_number(t)
            if t_val is not None:
                tokens_w_vals.append(t_val)
                val_token_inds.append(t_i)
            else:
                tokens_w_vals.append(t)
            # end if
        # end for
        return tokens_w_vals, val_token_inds

    def write_nl(
        self, 
        orig_data: Dict,
        mut_nls: List[str], 
        cksum_val: str,
        res_file: Path
    ) -> None:
        mut_res = list()
        # self.mut_dir.mkdir(parents=True, exist_ok=True)
        orig_nl_q = f"{orig_data['body']} {orig_data['question']}"
        orig_nl_ans = orig_data['answer']
        _mut_nls = list()
        if len(set(mut_nls))>0:
            for m in list(set(mut_nls)):
                if m!=orig_nl_q:
                    _mut_nls.append(m)
                # end if
            # end for
        # end if
        if any(_mut_nls):
            mut_nls = _mut_nls
        # end if

        if any(mut_nls):
            res = {
                'orig': {
                    'question': orig_nl_q,
                    'answer': orig_nl_ans
                },
                'mutations': [
                    {
                        'question': m,
                        'answer': None
                    } for m in mut_nls
                ]
            }
            Utils.write_json(
                res, 
                res_file,
                pretty_format=True
            )
        elif not any(mut_nls) and os.path.exists(str(res_file)):
            print(f"{cksum_val}::NO_MUTATIONS")
            os.remove(str(res_file))
        else:
            print(f"{cksum_val}::NO_MUTATIONS")
        # end if
        return

    @classmethod
    def mutate(
        cls,
        dataset_name: str,
        model_name: str
    ) -> None:
        mut_nl_obj = cls(
            dataset_name=dataset_name,
            model_name=model_name
        )
        i = 0
        for d in mut_nl_obj.orig_data:
            cksum_val = Utils.get_cksum(str(d['id']), length=7)
            res_file = mut_nl_obj.mut_dir / f"mut-nl-{cksum_val}.json"
            if not os.path.exists(str(res_file)):
                print(f"Mutate2nlWoEquationNValidationCls.mutate::{cksum_val}")
                mut_nl_res = mut_nl_obj.generate_mutated_nl_questions(
                    d,
                    res_file,
                    max_bound=100,
                    min_bound=0,
                    num_mutations=Macros.num_mutations
                )
                mut_nl_obj.write_nl(d, mut_nl_res, cksum_val, res_file)
            # end if
            i += 1
        # end for
        return

