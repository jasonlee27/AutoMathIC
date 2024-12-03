
import os
import re
import numpy as np
# import pandas as pd

from typing import *
from pathlib import Path

from numpy.linalg import norm
# from datasets import load_dataset
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

from .externaldataset import ExternalDataset
from .embedding import Embedding

from ..consistency.consistency import ModalConsistency
from ..llmut.evaluate import Evaluate
from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class Retrieve:

    def __init__(
        self, 
        ext_dataset_name: str,
        retrival_method: str = 'embedding',
        topk: int = 5,
        embedding_model_name: str='openai',
        openai_embedding_model_name: str='text-embedding-ada-002',
        bert_emb_model_name: str='all-MiniLM-L6-v2'
    ):
        self.ext_dataset_name = ext_dataset_name
        self.ext_dataset = ExternalDataset.read_dataset(ext_dataset_name)
        self.topk = topk
        self.retrival_method = retrival_method
        self.emb_model = None
        res_dir = Macros.result_dir / 'suspiciousness'
        res_dir.mkdir(parents=True, exist_ok=True)
        self.score_dict_path = res_dir / f"{retrival_method}-{embedding_model_name}-ext-{self.ext_dataset_name}.json"
        self.emb_dict_over_ext_dataset = Utils.read_json(self.score_dict_path)
        self.score_dict_over_ext_dataset = dict()
        self.is_there_emb_dict_over_ext_dataset = False
        if retrival_method=='embedding':
            self.emb_model = Embedding(
                embedding_model_name=embedding_model_name,
                openai_embedding_model_name=openai_embedding_model_name,
                bert_emb_model_name=bert_emb_model_name
            )
            if self.emb_dict_over_ext_dataset is None:
                self.emb_dict_over_ext_dataset = dict()
            else:
                self.is_there_emb_dict_over_ext_dataset = True
            # end if
        # end if

    def cosine_sim(
        self, 
        vec_a: np.array, 
        vec_b: np.array
    ):
        return np.dot(vec_a, vec_b)/(norm(vec_a)*norm(vec_b))

    def get_answer_from_ground_truth(self, str_answer: str) -> str:
        if str_answer is None:
            return '<N/A>'
        # end if
        if type(str_answer)!=str:
            str_answer = str(str_answer)
        # end if
        ans_search = re.search(r'([-|$]?\d+)', str_answer)
        if ans_search is not None:
            return ans_search.group(1).strip().replace('$','')
        # end if
        return

    def retrieve(self, query_data: Dict):
        body = query_data.get('body', '')
        q = query_data.get('question', None)
        query_answer = query_data.get('answer', None)
        assert q is not None
        query_text = f"{body} {q}"
        query_emb = None
        if self.retrival_method=='embedding':
            query_emb = self.emb_model.get_embedding(query_text)
        # end if
        assert query_emb is not None

        score_dict = dict()
        for ed in self.ext_dataset:
            ext_body = ed.get('body', '')
            ext_q = ed.get('question', None)
            assert ext_q is not None
            ext_text = f"{ext_body} {ext_q}"

            if not self.is_there_emb_dict_over_ext_dataset:
                # when no embedding file
                ed_emb = self.emb_model.get_embedding(ext_text)
                self.emb_dict_over_ext_dataset.setdefault(
                    ext_text, {
                        'answer': ed['answer'],
                        'embedding': ed_emb
                    }
                )
            else:
                # when there exists embedding file
                if self.retrival_method=='embedding':
                    ed_emb = self.emb_dict_over_ext_dataset[ext_text]['embedding']
                # end if
            # end if
            self.score_dict_over_ext_dataset[ext_text] = {
                'answer': ed['answer'],
                'score': self.cosine_sim(query_emb, ed_emb)
            }
        # end for
        if not self.is_there_emb_dict_over_ext_dataset:
            Utils.write_json(
                self.emb_dict_over_ext_dataset,
                self.score_dict_path,
            )
            self.is_there_emb_dict_over_ext_dataset = True
        # end if
        sorted_ext_texts = sorted(
            list(self.score_dict_over_ext_dataset.keys()), 
            key=lambda x: self.score_dict_over_ext_dataset[x]['score'],
            reverse=True
        )
        return {
            et: self.score_dict_over_ext_dataset[et]
            for et in sorted_ext_texts[:self.topk]
        }, query_text, query_answer

    def evaluate_retrieved_data(
        self, 
        model_name: str, 
        retrieved_data: List[Dict]
    ):
        eval_res_list = list()
        model, tokenizer = Evaluate.load_model(model_name=model_name)
        cot_prompt = Macros.openai_cot_prompt
        for ret_d_dict in retrieved_data:
            eval_res = {
                'query_text': ret_d_dict['query_text'],
                'query_answer': ret_d_dict['query_answer'],
                'query_cot_resp': None,
                'query_correctness': None,
                'retrieved_texts_from_ext_dataset': dict()
            }
            query_cot_resp = Evaluate.get_response(
                model, 
                tokenizer, 
                ret_d_dict['query_text'],
                cot_prompt,
                prompt_append=True
            )
            answer_from_cot = ModalConsistency.get_answer_from_cot_resp(query_cot_resp)
            answer_gt = self.get_answer_from_ground_truth(ret_d_dict['query_answer'])
            correctness = answer_from_cot==answer_gt
            eval_res['query_cot_resp'] = query_cot_resp
            eval_res['query_correctness'] = correctness

            for ret_d in ret_d_dict['retrieved_texts_from_ext_dataset'].keys():
                ret_d_answer = ret_d_dict['retrieved_texts_from_ext_dataset'][ret_d]['answer']
                ret_d_score = ret_d_dict['retrieved_texts_from_ext_dataset'][ret_d]['score']
                ret_cot_resp = Evaluate.get_response(
                    model, 
                    tokenizer, 
                    ret_d,
                    cot_prompt,
                    prompt_append=True
                )
                answer_from_cot = ModalConsistency.get_answer_from_cot_resp(ret_cot_resp)
                # answer_from_cot = eval(answer_from_cot) if answer_from_cot is not None else answer_from_cot
                answer_gt = self.get_answer_from_ground_truth(ret_d_answer)
                correctness = answer_from_cot==answer_gt
                eval_res['retrieved_texts_from_ext_dataset'].setdefault(
                    ret_d, {
                        'score': ret_d_score,
                        'answer': ret_d_answer,
                        'cot_resp': ret_cot_resp,
                        'correctness': correctness
                    }
                )
            # end for
            eval_res_list.append(eval_res)
        # end for
        return eval_res_list
    
    @classmethod
    def exp_main(
        cls, 
        dataset_name: str,
        ext_dataset_name: str,
        model_name: str,
        retrival_method: str = 'embedding',
        embedding_model_name: str='openai',
        topk: int = 5,
    ):
        retrival_obj = cls(
            ext_dataset_name=ext_dataset_name,
            retrival_method=retrival_method,
            topk=topk,
            embedding_model_name=embedding_model_name
        )
        data_under_test: List[Dict] = ExternalDataset.read_dataset(dataset_name)
        retrieval_res = list()
        num_data_under_test = len(data_under_test)//5
        print(f"MODEL: {model_name}")
        print(f"#DATA_UNDER_TEST_FOR_SUSPICIOUSNESS: {num_data_under_test}")
        for d_i, d in enumerate(data_under_test):
            if d_i<num_data_under_test:
                cksum_val = Utils.get_cksum(d['id'], length=7)
                print(cksum_val)
                ret_texts, query_text, query_answer = retrival_obj.retrieve(d)
                retrieval_res.append({
                    'query_text': query_text,
                    'query_answer': query_answer,
                    'retrieved_texts_from_ext_dataset': ret_texts
                })
            # end if
        # end for
        retrieval_res = retrival_obj.evaluate_retrieved_data(
            model_name, 
            retrieval_res
        )
        res_dir = Macros.result_dir / 'suspiciousness'
        Utils.write_json(
            retrieval_res,
            res_dir / f"retrieved-data-using-{retrival_method}-{dataset_name}-from-ext-{ext_dataset_name}.json",
            pretty_format=True
        )
        return