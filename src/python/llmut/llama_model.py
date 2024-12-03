

import os
import torch
import numpy as np
import transformers

from typing import *
from pathlib import Path

from .llama import Llama

from ..utils.macros import Macros
from ..utils.utils import Utils


class LlamaModel:

    @classmethod
    def load_model(
        cls,
        ckpt_dir: Path,
        tokenizer_path: Path
    ):
        generator = Llama.build(
            ckpt_dir=str(ckpt_dir),
            tokenizer_path=str(tokenizer_path),
            max_seq_len=Macros.llm_resp_max_len,
            max_batch_size=Macros.llama_max_batch_size,
        )
        return generator
    
    @classmethod
    def predict(
        cls, 
        generator, 
        query_text: Any,
        prompt: str,
        prompt_append: bool=False,
    ) -> List[str]:
        preds = list()
        if type(query_text)==list:
            input_text = [
                f"{prompt} {q}"
                for q in query_text
            ]
            if prompt_append:
                input_text = [
                    f"{q} {prompt}"
                    for q in query_text
                ]
            # end if
        else:
            input_text = [f"{prompt} {query_text}"]
            if prompt_append:
                input_text = [f"{query_text} {prompt}"]
            # end if
        # end if

        results = generator.text_completion(
            input_text,
            max_gen_len=Macros.llm_resp_max_len,
            temperature=Macros.resp_temp,
            top_p=0.9,
        )
        # results = generator.chat_completion(
        #     input_text,
        #     max_gen_len=Macros.llm_resp_max_len,
        #     temperature=Macros.resp_temp,
        #     top_p=1,
        # )

        for t, r in zip(input_text, results):
            print(t, r.keys())
            print(f"> {r['generation']}")
            print("\n==================================\n")
            preds.append(r['generation'])
        # end for
        raise()
        return preds
