
import os
import re
import random
import subprocess

from typing import *
from pathlib import Path

from ..dataset.asdiv import Asdiv
from ..dataset.gsm8k import Gsm8k
from ..dataset.svamp import Svamp
from ..dataset.multiarith import MultiArith
from ..dataset.addsub import Addsub
from ..dataset.singleeq import SingleEq

from ..prog_synthesis.varfinder import VarFinder
from ..llmut.llama_model import LlamaModel
from ..llmut.openai import OpenAiModel
from ..llmut.evaluate import EvaluateWithMultimodals

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class MutationValidataionCls:

    # def __init__(
    #     self, 
    #     cksum_val: str,
    #     dataset_name: str,
    #     pl_type='python'
    # ):
    #     self.cksum_val: str = cksum_val
    #     self.dataset_name = dataset_name
    #     self.pl_type: str = pl_type
    #     self.mut_dir: Path = Macros.result_dir / 'eq2code' / dataset_name / pl_type / 'mutation'
    #     self.orig_data: Dict = self.find_target_orig_data()
    #     self.orig_vars = Utils.read_json(self.mut_dir.parent / f"vars-{cksum_val}.json")
    #     self.mut_res: List[Dict] = self.read_mut_res()
    
    @classmethod
    def get_llm_validation_results(
        cls, 
        query_text: str,
        model_name: str
    ):
        text = f"\nQ: {query_text}\nA:"
        engine_name = Macros.gpt3d5_engine_name if model_name.lower().strip()=='gpt3.5' else Macros.gpt4_engine_name
        return OpenAiModel.predict(
            text,
            prompt=Macros.openai_mutation_validation_cls_prompt,
            prompt_append=False,
            model_name=engine_name,
            temperature=Macros.resp_temp_for_mutation_validation_cls,
            logprobs=False
        )

    @classmethod
    def validate_from_llm(
        cls, 
        query_text: str,
        model_name: str
    ):
        out = cls.get_llm_validation_results(
            query_text,
            model_name
        )
        if out.lower().strip()=='yes':
            return True
        # end if
        return False

    @classmethod
    def mutation_validataion(
        cls, 
        mutated_question: str,
        model_name: str
    ):
        return cls.validate_from_llm(
            mutated_question,
            model_name
        )
    