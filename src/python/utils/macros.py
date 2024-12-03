
# This script is for defining all macros used in scripts in slpproject/python/coteval/

import os

from typing import *
from pathlib import Path


class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    root_dir: Path = this_dir.parent.parent.parent
    src_dir: Path = this_dir.parent.parent
    python_dir: Path = this_dir.parent
    storage_dir: Path = root_dir
    
    result_dir: Path = storage_dir / '_results'
    download_dir: Path = storage_dir / '_downloads'
    dataset_dir: Path = download_dir / 'dataset'
    log_dir: Path = storage_dir / '_logs'
    paper_dir: Path = root_dir / 'paper'
    
    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.5f}"
    RAND_SEED = 27

    # ===== LLM =====
    num_processes = 3
    llm_output_max_len = 100
    llm_pl_prompt = 'convert the following question into python code that returns the answer for the question:'
    llm_pl_models = {
        'codet5p': 'Salesforce/codegen-350M-mono',
        'text-davinci-002': 'text-davinci-002'
    }
    pl_indentation_dict = {
        'python': '    '
    }

    # ===== EQ2CODE =====
    code_header = {
        'python': [
            'def function(', '):\n    '
        ]
    }
    code_footer = {
        'python': [
            '\nprint(function(', '))'
        ]
    }
    math_operators = ['+', '-', '*', '/']
    ari_rs_dataset_names = ['asdiv', 'gsm8k', 'svamp', 'multiarith']
    num_pos = 'NUM'
    answer_key_for_var_dict = '<answer>'
    num_mutations = 20
    num_max_mutation_iter = 100
    # ==========

    # ===== LLM under test (e.g. GPT models) =====
    llm_result_dir = download_dir / 'llm'
    llama_dir = llm_result_dir / 'llama2'
    llm_resp_max_len = 500
    resp_temp = 0.
    resp_temp_for_mutation_validation_cls = 0.
    resp_temp_for_self_consistency = 0.7
    llama_max_batch_size = 8
    llama_model_name = 'llama-2-13b'
    llama_model_dir = llama_dir / llama_model_name

    gpt3d5_engine_name = 'gpt-3.5-turbo'
    gpt4_engine_name = 'gpt-4-turbo-preview'
    palm_engine_name = 'models/gemini-pro' # 'models/text-bison-001'
    gpt4omini_engine_name = 'gpt-4o-mini'

    openai_prompt = 'I want you to act like a mathematician. I will type mathematical question and you will respond with the answer of the question. I want you to answer only with the final amount and nothing else. Do not write explanations:'
    openai_prompt_w_exp = 'I want you to act like a mathematician. I will type mathematical question and you will respond with the answer of the question. I want you to answer with the following json format: { "answer":, "explanation":}. In "answer", you only answer with only with the final amount and nothing else. Do not write explanations. In "explanation", you only write mathematical equation for computing the answer. Question is the following:'
    
    openai_cot_prompt = "Let's think step by step and end your response with 'the answer is {answer}'"
    openai_code_prompt = "I want you to act like a mathematician. I will type mathematical question and you will respond with a function named with 'func' in python code that returns the answer of the question. the function should have no arguments. I want you to answer only with the final python code and nothing else. Do not write explanations:"
    openai_eqn_prompt = "Write a wolframalpha mathematical equation with no explanations and no units to the numbers in the equation. Generate the answer format starting with `Answer ='"

    prompts_over_modals = {
        'cot_response': (openai_cot_prompt, True),
        'code_response': (openai_code_prompt, False),
        'eqn_response': (openai_eqn_prompt, True)
    }

    units = [
        'kg', 'ft', 'hours', 'minutes', 'minutes//hour', '//hour'
        'cm', 'meters', 'centimeters', 'inches', 'emails', 'pizza'
        'dollar', 'cent', 'dollars', 'cents', 
    ]
    # =====
    
    # ===== self-consistency =====
    self_consistency_top_k = 3
    # =====

    # ===== genetic_fg=====
    genetic_fg_belief_weight_over_modals = {
        'cot_response': 0.4,
        'code_response': 0.3,
        'eqn_response': 0.3
    }
    # =====
