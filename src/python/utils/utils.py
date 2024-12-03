from typing import *
from pathlib import Path

import re, os
import csv
import sys
import json
import torch
import pickle
import string
import hashlib
import statistics
import subprocess
import numpy as np
import wolframalpha
import contractions

from nltk.corpus import treebank
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from dataclasses import dataclass

from .macros import Macros

wolframalpha_api_key = os.environ["WOLFRAMALPHA_API_KEY"]
wa_client = wolframalpha.Client(wolframalpha_api_key)


class Utils:
    
    @classmethod
    def fix_contractions(cls, sent):
        _sent = contractions.fix(sent)
        _sent = re.sub(r" is been ", r" has been ", _sent)
        return _sent
    
    @classmethod
    def tokenize(cls, sent: str)->list:
        return word_tokenize(sent)

    @classmethod
    def detokenize(cls, tokens: list)->str:
        tokens = ['"' if (t=='``' or t=='\'\'') else t for t in tokens]
        sent = TreebankWordDetokenizer().detokenize(tokens)
        sent = re.sub(r"(.+)\-\-(.+)", r"\1 -- \2", sent)
        sent = re.sub(r"(.+)\.\.\.(.+)", r"\1 ... \2", sent)
        sent = cls.fix_contractions(sent)
        return sent
    
    @classmethod
    def read_txt(cls, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        #end with
        return lines

    @classmethod
    def read_csv(cls, data_file, sep=',', split_headers=False):
        with open(data_file, newline='') as file:
            reader = csv.reader(file, delimiter=sep)
            headers = None
            if split_headers:
                # store the headers in a separate variable,
                # move the reader object to point on the next row
                headers = next(reader)
            # end if
            # output list to store all rows
            rows = list()
            for row in reader:
                rows.append(row[:])
            # end for
        # end with
        return rows, headers

    @classmethod
    def write_txt(cls, input_str, data_file):
        with open(data_file, 'w') as f:
            lines = f.write(input_str)
        #end with
        return lines

    @classmethod
    def write_pkl(cls, results, pkl_file):
        with open(pkl_file, 'wb+') as f:
            pickle.dump(results, f)
        # end with
        return
    
    @classmethod
    def read_json(cls, json_file):
        # read cfg json file
        if os.path.exists(str(json_file)):
            with open(json_file, 'r') as f:
                return json.load(f)
            # end with
        # end if
        return
    
    @classmethod
    def write_json(cls, input_dict, json_file, pretty_format=False):
        with open(json_file, 'w') as f:
            if pretty_format:
                json.dump(input_dict, f, indent=4, cls=NpEncoder)
            else:
                json.dump(input_dict, f, cls=NpEncoder)
            # end if
        # end with
        return

    @classmethod
    def get_cksum(cls, input_str: str, length=7):
        return hashlib.md5(input_str.encode('utf-8')).hexdigest()[:length]

    @classmethod
    def save_checkpoint(cls, state, save_file='checkpoint.pth.tar'):
        torch.save(state, save_file)
        return

    @classmethod
    def normalize_string(cls, s):
        s = s.lower().strip()
        s = re.sub(r"<br />",r" ",s)
        s = re.sub(r'(\W)(?=\1)', '', s)
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)    
        return s

    @classmethod
    def execute_code(
        cls, 
        code_str: str, 
        dataset_name: str,
        model_name: str=None,
        func_name: str=None,
        pl_type: str='python'
    ) -> str:
        temp_pl_file = None
        if code_str is None:
            return None
        # end if
        if any(code_str):
            code_str = code_str.lstrip("```python\n")
            code_str = code_str.rstrip("\n```")
            if pl_type=='python':
                # execute the llm_temp.py
                code_str_splitlines = code_str.splitlines()
                if any(code_str_splitlines):
                    last_stmt = code_str.splitlines()[-1]
                else:
                    return None
                # end if
                print_stmt = 'print(func())' if func_name is None else f"print({func_name}())"
                if 'print(' in last_stmt:
                    code_str = f"import math\nimport datetime\n{code_str}\n"
                else:
                    code_str = f"import math\nimport datetime\n{code_str}\n{print_stmt}\n"
                # end if
                temp_pl_file = Macros.result_dir / f"llm_{dataset_name}_temp.py"
                if model_name is not None:
                    temp_pl_file = Macros.result_dir / f"llm_{dataset_name}_{model_name}_temp.py"
                # end if
                Utils.write_txt(code_str, temp_pl_file)
                cmd = f"python {str(temp_pl_file)}"
            # end if
            try:
                output = subprocess.check_output(cmd, shell=True, timeout=5).strip() # timeout 5 sec
                if os.path.exists(str(temp_pl_file)):
                    os.remove(str(temp_pl_file))
                # end if
                return output.decode()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"<ERROR_CODE>\n{code_str}<\ERROR_CODE>")
                if os.path.exists(str(temp_pl_file)):
                    os.remove(str(temp_pl_file))
                # end if
                return None
            # end try
        else:
            return None
        # end if
    
# ==========
# Stats

    @classmethod
    def avg(cls, nums: list, decimal=3):
        return round(sum(nums) / len(nums), decimal)

    @classmethod
    def median(cls, nums: list, decimal=3):
        return round(statistics.median(nums), decimal)

    @classmethod
    def stdev(cls, nums: list, decimal=3):
        return round(statistics.stdev(nums), decimal)

    @classmethod
    def lod_to_dol(cls, list_of_dict: List[dict]) -> Dict[Any, List]:
        """
        Converts a list of dict to a dict of list.
        """
        keys = set.union(*[set(d.keys()) for d in list_of_dict])
        return {k: [d.get(k) for d in list_of_dict] for k in keys}

# ==========
# LLM answer

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        if str_answer is None:
            return '<N/A>'
        # end if
        if type(str_answer)!=str:
            str_answer = str(str_answer)
        else:
            str_answer = str_answer.replace(',', '')
        # end if
        ans_search = re.search(r'([-]?\d+(?:\.\d+)?)', str_answer)
        if ans_search is not None:
            return ans_search.group(1).strip()
        # end if
        translator = str.maketrans("", "", string.punctuation)
        return str(float(str_answer.translate(translator).strip().lower()))

    @classmethod
    def get_answer_from_cot_resp(cls, cot_resp: str) -> str:
        for l in cot_resp.strip().splitlines()[::-1]:
            l = l.lower()
            if 'the answer is' in l:
                l = l.split('the answer is')[-1]
                l = l.replace(',', '')
                l_search = re.search(r'([-|\$]?\d+(?:\.\d+)?)', l)
                if l_search is not None:
                    ans_str = l_search.group(0).strip().replace('$','')
                    return str(float(ans_str))
                else:
                    translator = str.maketrans("", "", string.punctuation)
                    try:
                        ans = l.translate(translator).strip().lower()
                        return str(float(ans))
                    except ValueError:
                        return 
                    # end try
                # end if
            # end if
        # end for
        l_search = re.search(r'([-|\$]?\d+(?:\.\d+)?)', cot_resp.strip())
        if l_search is not None:
            ans_str = l_search.group(0).strip().replace('$','')
            return str(float(ans_str))
        else:
            translator = str.maketrans("", "", string.punctuation)
            try:
                ans = l.translate(translator).strip().lower()
                return str(float(ans))
            except ValueError:
                return 
            # end try
        # end if
        return

    @classmethod
    def get_func_name_from_code_resp(cls, code_resp: str):
        func_name = None
        for l in code_resp.splitlines():
            l_search = re.search(r'def ([a-zA-Z_{1}][a-zA-Z0-9_]+)\(\)\:', l)
            if l_search is not None:
                func_name = l_search.group(1)
            # end if
        # end for
        return func_name

    @classmethod
    def get_answer_from_code_resp(
        cls, 
        code_resp: str, 
        dataset_name: str,
        model_name: str=None
    ) -> str:
        if 'the answer is' in code_resp.lower():
            l_search = re.search(r'the answer is ([-|\$]?\d+(?:\.\d+)?)', code_resp)
            if l_search is not None:
                ans_str = l_search.group(1).strip()
                if ans_str.startswith('$'):
                    ans_str = ans_str.replace('$','')
                # end if
                return str(float(ans_str))
                # return l_search.group(1).strip()
            else:
                translator = str.maketrans("", "", string.punctuation)
                l = code_resp.split('the answer is')[-1].strip().lower()
                try:
                    ans = l.translate(translator)
                    return str(float(ans))
                except ValueError:
                    return '<N/A>'
                # end try
            # end if
        else:
            try:
                func_name = cls.get_func_name_from_code_resp(code_resp)
                return_val = cls.execute_code(
                    code_resp, 
                    dataset_name, 
                    pl_type='python', 
                    model_name=model_name,
                    func_name=func_name
                )
                return str(float(return_val)) if return_val is not None else None
            except ValueError:
                return
            # try
        # end if

    @classmethod
    def is_str_float(cls, string):
        try:
            float(string)
            return True
        except ValueError:
            return False
        # end try

    @classmethod
    def remove_units(cls, eqn_str):
        for unit in Macros.units:
            eqn_str = eqn_str.replace(unit, '').strip()
        # end for
        return eqn_str

    @classmethod
    def check_paranthesis(cls, my_string: str):
        # brackets = ['()', '{}', '[]']
        brackets = ['()']
        while any(x in my_string for x in brackets):
            for br in brackets:
                my_string = my_string.replace(br, '')
        return not my_string

    @classmethod
    def get_answer_from_eqn_resp(cls, eqn_resp: str) -> str:
        if 'the answer is' in eqn_resp.lower():
            _eqn_resp = eqn_resp.lower().split('answer =')[-1].splitlines()
            if any(eqn_resp):
                eqn_resp = _eqn_resp[0].strip()
            else:
                eqn_resp = eqn_resp.lower().strip()
            # end if
        elif 'answer = ' in eqn_resp.lower():
            _eqn_resp = eqn_resp.lower().split('answer =')[-1].splitlines()
            if any(_eqn_resp):
                eqn_resp = _eqn_resp[0].strip()
            else:
                eqn_resp = eqn_resp.lower().strip()
            # end if
        else:
            eqn_resp = eqn_resp.lower().strip()
        # end if
        eqn_resp = eqn_resp.replace("'","")
        eqn_resp = cls.remove_units(eqn_resp)
        # eqn_resp = re.sub(r'[a-z]+', '', eqn_resp).strip()
        if eqn_resp.isnumeric() or cls.is_str_float(eqn_resp):
            return str(float(eqn_resp.strip()))
        else:
            result, ret_val = None, None
            _eqn_resp = eqn_resp.strip()
            while _eqn_resp!='':
                if _eqn_resp.isnumeric() or cls.is_str_float(_eqn_resp):
                    ret_val = _eqn_resp.strip()
                    return str(float(ret_val))
                else:
                    try:
                        ret_val = eval(_eqn_resp)
                        return str(float(ret_val))
                    except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                        pass
                    # end try

                    # solve the equation with wolframalpha api
                    if '(' not in _eqn_resp and ')' not in _eqn_resp:
                        q = f"solve({_eqn_resp})"
                    elif cls.check_paranthesis(_eqn_resp) and not _eqn_resp.startswith('solve('):
                        q = f"solve({_eqn_resp})"
                    elif not cls.check_paranthesis(_eqn_resp) and _eqn_resp.startswith('solve('):
                        q = f"{_eqn_resp})"
                    elif not cls.check_paranthesis(_eqn_resp) and not _eqn_resp.startswith('solve('):
                        q = f"solve({_eqn_resp})"
                    else:
                        q = _eqn_resp
                    # end if
                    resp = wa_client.query(q)
                    
                    for result in resp.results:
                        pass
                    # end for
                    if result is None:
                        # _eqn_resp = ' '.join(_eqn_resp.split()[:-1]).strip()
                        try:
                            return str(float(_eqn_resp))
                        except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                            return str(_eqn_resp)
                        # end try
                    elif result.text is not None:
                        # print(_eqn_resp, q, result.text)
                        if result.text.isnumeric() or cls.is_str_float(result.text):
                            try:
                                ret_val = result.text.strip()
                                return str(float(ret_val))
                            except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                                return str(ret_val)
                            # end try
                        elif '=' in result.text:
                            ret_val = result.text.split('=')[-1].strip()
                            try:
                                return str(float(ret_val))
                            except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                                return str(ret_val)
                            # end try
                            break
                        elif 'no solutions exist' in result.text:
                            ret_val = result.text.strip()
                            return str(ret_val)
                        else:
                            # _eqn_resp = ' '.join(_eqn_resp.split()[:-1]).strip()
                            try:
                                return str(float(_eqn_resp))
                            except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                                return str(_eqn_resp)
                            # end try
                        # end if
                    else:
                        return str(_eqn_resp)
                    # end if
                # end if
            # end while
            if ret_val is not None:
                try:
                    return str(float(ret_val))
                except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                    return str(ret_val)
                # end try
            else:
                translator = str.maketrans("", "", string.punctuation)
                l = eqn_resp.strip().lower()
                try:
                    ans = l.translate(translator)
                    return str(float(ans))
                except ValueError:
                    return '<N/A>'
                # end try
            # end if
        # end if


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
