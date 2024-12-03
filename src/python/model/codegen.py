

import os
import torch
import spacy
import numpy as np

from typing import *
from pathlib import Path

from ..dataset.asdiv import Asdiv
from ..dataset.gsm8k import Gsm8k
from ..dataset.svamp import Svamp
from ..dataset.multiarith import MultiArith
from ..dataset.addsub import Addsub
from ..prog_synthesis.varfinder import VarFinder
from ..prog_synthesis.eq2code import Eq2code
from ..prog_synthesis.llm4pl import Llm4plHf, Llm4plOpenai

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class Codegen_eq:

    @classmethod
    def load_nlp_parser(cls):
        spacy.prefer_gpu()
        nlp_parser = spacy.load("en_core_web_md")
        if spacy.__version__.startswith('2'):
            nlp_parser.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            nlp_parser.add_pipe("benepar", config={"model": "benepar_en3"})
        # end if
        return nlp_parser

    @classmethod
    def read_dataset(cls, dataset_name):
        dataset_obj = None
        if dataset_name=='asdiv':
            dataset_obj = Asdiv()
        elif dataset_name=='gsm8k':
            dataset_obj = Gsm8k()
        elif dataset_name=='svamp':
            dataset_obj = Svamp()
        elif dataset_name=='multiarith':
            dataset_obj = MultiArith()
        elif dataset_name=='addsub':
            dataset_obj = Addsub()
        # end if
        return dataset_obj

    @classmethod
    def main(cls, dataset_name='svamp'):
        dataset = cls.read_dataset(dataset_name)
        nlp_parser = cls.load_nlp_parser()
        for d_i in range(len(dataset)):
            d = dataset[d_i]
            # print(d_i+1)
            answer = d['answer']
            #svamp: chal-50: unsolvable, chal-680 answer is incorrect
            if d['id'] not in [
                'chal-50', 'chal-680'
            ]:
                # find variable from the question in natural language
                var_dict = VarFinder.find_vars_in_query(d, nlp_parser)
                cv_obj = Eq2code(d, var_dict, dataset_name)

                # evaluate the generated code
                cv_obj.write_code()

                # exevute the generated code
                out = cv_obj.execute_code()
                if eval(out)!=answer:
                    query = d['body']+'\n'+d['question']
                    eqn = d['equation']
                    print(d['id'])
                    print(f"TEXT: {query}")
                    print(f"EQN: {eqn}")
                    print('----------')
                    print(f"CODE_GENERATED:\n{cv_obj.code}")
                    print('----------')
                    print(f"OUT: {out}, ANSWER: {answer}")
                    print(eval(out)==answer)
                    print('==========')
                    print()
                # end if
            # end if
        # end for
        return


class Codegen_llm:
    # code generator using LLM
    @classmethod
    def read_dataset(cls, dataset_name):
        dataset_obj = None
        if dataset_name=='asdiv':
            dataset_obj = Asdiv()
        elif dataset_name=='gsm8k':
            dataset_obj = Gsm8k()
        elif dataset_name=='svamp':
            dataset_obj = Svamp()
        elif dataset_name=='multiarith':
            dataset_obj = MultiArith()
        elif dataset_name=='addsub':
            dataset_obj = Addsub()
        # end if
        return dataset_obj

    @classmethod
    def load_prog_synthesis_model(cls, model_name):
        model_obj = Llm4plOpenai(model_name)
        return model_obj

    @classmethod
    def generate_code_from_text(cls, query_text, model_obj):
        return model_obj.generate_code(query_text)

    @classmethod
    def main(
        cls,
        model_name='codet5p',
        dataset_name='asdiv'
    ):
        _model_name = Macros.llm_pl_models[model_name]
        model_obj = cls.load_prog_synthesis_model(_model_name)
        dataset = cls.read_dataset(dataset_name)
        for d_i in range(len(dataset[:10])):
            d = dataset[d_i]
            print(d)
            body, question = d['body'], d['question']
            query = f"{body} {question}"
            code = cls.generate_code_from_text(query, model_obj)
            print(f"TEXT: {query}")
            print('----------')
            print(f"CODE_GENERATED: {code}")
            print('==========')
            print()
        # end for
        return