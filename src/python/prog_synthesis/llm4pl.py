

import os
import torch
import openai
import numpy as np

from typing import *
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, CodeGenTokenizer

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Llm4plHf:

    def __init__(
        self,
        model_name,
        prompt=None
    ):
        self.model_name = model_name
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.model, self.tokenizer = self.load_model_n_tokenizer(model_name)
        self.output_max_len = Macros.llm_output_max_len
        self.prompt = prompt 
        if self.prompt is None:
            self.prompt = Macros.llm_pl_prompt
        # end if

    def load_model_n_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        return model, tokenizer

    def generate_code(self, query_text):
        # input_text = f"{self.prompt} {query_text}"
        input_text = query_text
        encoding = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(encoding.input_ids, max_length=self.output_max_len)
        out_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return out_text


class Llm4plOpenai:

    def __init__(self, 
                 engine_name, 
                 prompt="Write a Python function to answer the following question.", 
                 max_token=150):
        self.engine_name = engine_name # "text-davinci-002" or "davinci"
        self.prompt = prompt # "Write a Python function to answer the following question."
        self.max_token = max_token

    def generate_code(self, query_text):
        text = f"{self.prompt} {query_text}"
        response = openai.Completion.create(
            engine=self.engine_name,
            prompt=text,
            temperature=0.7,
            max_tokens=self.max_token,
        )
        return response["choices"][0]["text"]
