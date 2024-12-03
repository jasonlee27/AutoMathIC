
import os
import sys
import time
import numpy as np

from typing import *
from pathlib import Path
import google.generativeai as palm
import google.api_core as palm_api_core

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


google_cloud_api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
palm.configure(api_key=google_cloud_api_key)
# models = [model for model in palm.list_models()]

# for model in models:
#     print(model.name)
# # end for

# PaLMModel -> GEMINI model
class PaLMModel:

    engine_name = Macros.palm_engine_name

    @classmethod
    def set_model_name(
        cls, 
        engine_name
    ) -> None:
        cls.engine_name = engine_name
        return model

    @classmethod
    def predict(
        cls, 
        query_text: str,
        prompt: str='',
        prompt_append: bool=False,
        model_name: str=None,
        temperature=Macros.resp_temp,
        top_k: int=1,
        logprobs: bool=True
    ) -> str:
        text = f"{prompt} {query_text}"
        if prompt_append:
            text = f"{query_text} {prompt}"
        # end if
        
        _model_name = cls.engine_name if model_name is None else model_name
        model = palm.GenerativeModel(cls.engine_name)
        success = False
        retries = 1
        while not success:
            try:
                response = model.generate_content(
                    contents=text,
                    generation_config=palm.GenerationConfig(
                        temperature=temperature,
                        top_k=top_k,
                        candidate_count=top_k,
                        max_output_tokens=Macros.llm_resp_max_len
                    )
                )
                success = True    
                return {
                    'msg': response.text
                }
                # end if
            except (
                palm_api_core.exceptions.ResourceExhausted
            ) as e:
                wait = retries * 11;
                print(f"Error! Waiting {wait} secs and re-trying...::{e}")
                sys.stdout.flush()
                time.sleep(wait)
                # retries += 1
            except (
                palm_api_core.exceptions.InternalServerError
            ) as e:
                wait = retries * 1;
                print(f"Error! Waiting {wait} secs and re-trying...::{e}")
                sys.stdout.flush()
                time.sleep(wait)
                # retries += 1
            # end try

            # try:
            #     response = palm.generate_text(
            #         model=_model_name,
            #         prompt=text,
            #         temperature=temperature,
            #         candidate_count=top_k,
            #         max_output_tokens=Macros.llm_resp_max_len
            #     )
            #     success = True    
            #     return {
            #         'msg': response.result
            #     }
            #     # end if
            # except (
            #     palm_api_core.exceptions.ResourceExhausted
            # ) as e:
            #     wait = retries * 11;
            #     print(f"Error! Waiting {wait} secs and re-trying...::{e}")
            #     sys.stdout.flush()
            #     time.sleep(wait)
            #     # retries += 1
            # except (
            #     palm_api_core.exceptions.InternalServerError
            # ) as e:
            #     wait = retries * 1;
            #     print(f"Error! Waiting {wait} secs and re-trying...::{e}")
            #     sys.stdout.flush()
            #     time.sleep(wait)
            #     # retries += 1
            # # end try
        # end while

    @classmethod
    def get_palm_model_response(
        cls,
        model_name: str,
        input_text: str,
        demo_str: str=None,
        temp: float=Macros.resp_temp_for_self_consistency,
        top_k: int=1
    ) -> str:
        if demo_str is not None:
            input_text = f"{demo_str}{input_text}"
        # end if
        success = False
        retries = 1
        while not success:
            try:
                response = model.generate_content(
                    contents=input_text,
                    generation_config=palm.GenerationConfig(
                        temperature=temperature,
                        candidate_count=top_k,
                        max_output_tokens=Macros.llm_resp_max_len
                    )
                )
                if top_k==1:
                    # logprobs = response.choices[0].logprobs.content
                    return response.text
                else:
                    return [
                        r.text for r in response.candidates[:top_k]
                    ]
                # end if

                # response = palm.generate_text(
                #     model=model_name,
                #     prompt=input_text,
                #     candidate_count=top_k,
                #     temperature=temp,
                #     max_output_tokens=Macros.llm_resp_max_len,
                # )
                # if top_k==1:
                #     # logprobs = response.choices[0].logprobs.content
                #     return response.result
                # else:
                #     return [
                #         r['output'] for r in response.candidates[:top_k]
                #     ]
                # # end if
            except (
                palm_api_core.exceptions.ResourceExhausted
            ) as e:
                wait = retries * 11;
                print(f"Error! Waiting {wait} secs and re-trying...::{e}")
                sys.stdout.flush()
                time.sleep(wait)
                # retries += 1
            except (
                palm_api_core.exceptions.InternalServerError
            ) as e:
                wait = retries * 1;
                print(f"Error! Waiting {wait} secs and re-trying...::{e}")
                sys.stdout.flush()
                time.sleep(wait)
                # retries += 1
            # end try
        # end while
        return
