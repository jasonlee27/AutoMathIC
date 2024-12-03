
import os
import re
import spacy
import numpy
import benepar

from typing import *
from pathlib import Path

from word2number import w2n

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger

# download benepar package for english
benepar.download('benepar_en3')

SPECIAL_SYMBOL_MAP = {
    '-': 'HYPHEN',
    ' ': 'SPACE',
    '$': 'DOLLAR'
}


class VarFinder:

    def __init__(
        self, 
        data: Dict[str, str],
        nlp_parser):
        self.body = data.get('body', '')
        self.question = data['question']
        self.question = f"{self.body} {self.question}".strip()
        self.equation = data['equation']
        # self.code = data['code']
        self.doc = nlp_parser(self.question) # doc =nlp('Bananas are an excellent source of potassium.')
        self.np_list = list()
        self.var_name_header = 'var_'
        sents = list(self.doc.sents)
        for sent in sents:
            self.find_np_in_question(sent, self.np_list)
        # end for
        self.np_w_vals = self.find_np_with_numbers()
        self.var_w_vals = self.convert_np_to_var_name()
        # print(f"VAR_W_VALS: {self.var_w_vals}")
    
    def is_lowest_level_np(self, noun_phrase):
        children = list(noun_phrase._.children)
        if not any(children):
            return True
        else:
            for ch in children:
                if any(ch._.labels): 
                    if ch._.labels[0] in ['NP', 'QP']:
                        return False
                    # end if
                # end if
            # end for
            for ch in children:
                if not self.is_lowest_level_np(ch):
                    return False
                # end if
            # end for
            return True
        # end if

    def find_np_in_question(self, sent, np_list: List):
        if any(sent._.labels) and \
            sent._.labels[0] in ['NP', 'QP', 'CD'] and \
            self.is_lowest_level_np(sent):
            # hardcoding: split NP with the delimeter of 'and'/'or'
            # becuase of wrong neural parse tree of the berkley neural parser
            and_pat = re.search(r'([^0-9]+)?(\d+)([^0-9]+)?\sand\s([^0-9]+)?(\d+)([^0-9]+)?', str(sent))
            or_pat = re.search(r'([^0-9]+)?(\d+)([^0-9]+)?\sor\s([^0-9]+)?(\d+)([^0-9]+)?', str(sent))
            of_pat = re.search(r'([^0-9]+)?(\d+)([^0-9]+)?\sof\s([^0-9]+)?(\d+)([^0-9]+)?', str(sent))
            
            if and_pat:
                for np in str(sent).split(' and '):
                    np_list.append(np)
                # end for
            elif or_pat:
                for np in str(sent).split(' or '):
                    np_list.append(np)
                # end for
            elif of_pat:
                for np in str(sent).split(' of '):
                    np_list.append(np)
                # end for
            else:
                np_list.append(str(sent))
            # end if
        elif any(sent._.labels):
            # print(str(sent), sent._.labels, sent._.parse_string)
            childs = list(sent._.children)
            if any(childs):
                for ch in childs:
                    self.find_np_in_question(ch, np_list)
                # end for
            # end if
        elif not any(sent._.labels):
            tag = re.search(r'\(([^()]+)\s'+str(sent)+r'\)', sent._.parse_string).group(1)
            # print(tag, sent)
            if tag in ['CD']:
                np_list.append(str(sent))
            # end if
        # end if
        return

    def preprocess_np(self, np):
        # e.g. $ 4 -> 4 dollar
        # _np = re.sub(r'\$', '', np)
        # _np = re.sub(r'(\$)\s+?(\d+)', f"{SPECIAL_SYMBOL_MAP['$']} \2", np)
        return np

    def find_np_with_numbers(self):
        np_w_vals = dict()
        for _np in self.np_list:
            np = self.preprocess_np(_np)
            # doc = nlp_parser(np)
            # sent = list(doc.sents)[0]
            np_val = None
            # for token in list(sent._.children):
            for t_i, token in enumerate(np.split()):
                try:
                    np_val = w2n.word_to_num(str(token))
                    if np_val is not None:
                        # obj = re.sub(r'.*'+str(token), '', np).strip()
                        obj = self.var_name_header+np.strip()
                        if obj not in np_w_vals.keys():
                            np_w_vals[obj] = np_val
                        # end if
                        break
                    # end if
                except ValueError as e:
                    pass
                # end try
            # end for
        # end for
        return np_w_vals

    def convert_name_to_snake_case(self, noun_phrase):
        var_name = '_'.join(noun_phrase.lower().split())
        var_name = var_name.replace('-', '_'+SPECIAL_SYMBOL_MAP['-']+'_')
        var_name = var_name.replace('__', '_'+SPECIAL_SYMBOL_MAP[' ']+'_')
        var_name = var_name.replace('_$_', '_'+SPECIAL_SYMBOL_MAP['$']+'_')
        return var_name

    def convert_np_to_var_name(self):
        var_dict = dict()
        for np in self.np_w_vals.keys():
            var_name = self.convert_name_to_snake_case(np)
            var_dict[var_name] = self.np_w_vals[np]
        # end for
        return var_dict

    # @classmethod
    # def find_vars_over_questions(
    #     cls, 
    #     math_inputs: List[Dict]):
    #     spacy.prefer_gpu()
    #     nlp = spacy.load("en_core_web_md")
    #     if spacy.__version__.startswith('2'):
    #         nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    #     else:
    #         nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    #     # end if

    #     for math_input in math_inputs:
    #         question = math_input['question']
    #         eq = math_input['equation']
    #         code = math_input['code']
    #         varfinder = cls(question, eq, code, nlp)
    #         math_input['variables'] = varfinder.var_w_vals
    #     # end for
    #     return 

    @classmethod
    def find_vars_in_query(
        cls, 
        data,
        nlp_parser
    ):
        varfinder = cls(data, nlp_parser)
        return varfinder.var_w_vals

    @classmethod
    def convert_snake_case_var_to_name(cls, var_name):
        tokens = var_name.split('_')[1:] # remove var_ at the front of the variable name
        _tokens = list()
        t_i = 0
        while t_i<len(tokens):
            t = tokens[t_i]
            if t.islower() or t.isdigit():
                _tokens.append(t)
                t_i += 1
            else:
                _t_i = 0
                pos_t = t_i+_t_i
                _t = tokens[pos_t-1] if pos_t>0 else ''
                while (not tokens[t_i+_t_i].islower()):
                    if not tokens[t_i+_t_i].isdigit():
                        for key, val in SPECIAL_SYMBOL_MAP.items():
                            if tokens[t_i+_t_i]==val:
                                _t += key
                            # end if
                        # end for
                        _t_i += 1
                    else:
                        break
                    # end if
                # end while
                if pos_t>0:
                    _t += tokens[t_i+_t_i]
                    _tokens[t_i-1] = _t
                    t_i += _t_i+1
                else:
                    _tokens.append(_t)
                    t_i += _t_i
                # end if
            # end if
        # end while
        name = ' '.join(_tokens)
        return name
    
