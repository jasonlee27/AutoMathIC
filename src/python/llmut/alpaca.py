

import os
import torch
import numpy as np
import transformers

from typing import *
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from ..utils.macros import Macros
from ..utils.utils import Utils


class Alpaca:

    model_name = 'allenai/open-instruct-stanford-alpaca-7b'

    @classmethod
    def load_model(cls):
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        pipeline = transformers.pipeline(
            "text-generation",
            model=cls.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return pipeline, tokenizer

    @classmethod
    def predict(
        cls, 
        model, 
        tokenizer, 
        input_text: str
    ) -> List[str]:
        preds = list()
        sequences = model(
            f"{Macros.llama_prompt} {input_text}",
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=Macros.llm_resp_max_len,
        )

        for seq in sequences:
            # print(f"Result: {seq['generated_text']}")
            preds.append(seq['generated_text'])
        # end for
        return preds

    # @classmethod
    # def load_model(cls):
    #     tokenizer = AutoTokenizer.from_pretrained(cls.tokenizer_name)
    #     model = AutoModel.from_pretrained(
    #         cls.model_name,
    #         device_map='auto'
    #     )
    #     return model, tokenizer

    # @classmethod
    # def predict(cls, model, tokenizer, input_text: str) -> List[str]:
    #     preds = list()
    #     inputs = tokenizer.encode(input_text, return_tensors='pt')
    #     outputs = model.generate(
    #         inputs, 
    #         max_length=200,
    #         num_return_sequences=5,
    #         temperature=0.7
    #     )
    #     for output in outputs:
    #         out_seq = tokenizer.decode(output)
    #         print(f"Result: {out_seq}")
    #         preds.append(out_seq)
    #     # end for
    #     return preds
    