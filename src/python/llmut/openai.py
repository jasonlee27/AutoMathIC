
import os
import openai
import numpy as np

from typing import *
from pathlib import Path
from openai import OpenAI

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI()


class OpenAiModel:

    engine_name = Macros.gpt3d5_engine_name # model name

    @classmethod
    def set_model_name(
        cls, 
        engine_name
    ) -> None:
        cls.engine_name = engine_name
        return

    @classmethod
    def predict(
        cls, 
        query_text: str,
        prompt: str='',
        prompt_append: bool=False,
        model_name: str=None,
        temperature=Macros.resp_temp,
        logprobs=True
    ) -> str:
        '''
        TODO: if i need to add period(.) between prompt and query text? does it make significant change on output?
        '''
        text = f"{prompt} {query_text}"
        if prompt_append:
            text = f"{query_text} {prompt}"
        # end if
        
        _model_name = cls.engine_name if model_name is None else model_name
        # response = openai.Completion.create(
        #     engine=_model_name,
        #     prompt=text,
        #     temperature=Macros.resp_temp,
        #     max_tokens=Macros.llm_resp_max_len
        # )
        response = client.chat.completions.create(
            model=_model_name,
            messages=[{
                'role': 'user', 
                'content': text
            }],
            top_p=1,
            temperature=temperature,
            logprobs=logprobs,
            max_tokens=Macros.llm_resp_max_len
        )
        # print(text)
        # print(response["choices"][0]["text"])
        # print()
        if logprobs:
            logprobs = [
                p.logprob
                for p in response.choices[0].logprobs.content
            ]
            return {
                'msg': response.choices[0].message.content, 
                'logprob': logprobs
            }
        else:
            return response.choices[0].message.content
        # end if
