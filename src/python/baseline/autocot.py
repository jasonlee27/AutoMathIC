
import os
# import openai
import pickle
import numpy as np
# import pandas as pd

from typing import *
from pathlib import Path

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from ..dataset.asdiv import Asdiv
from ..dataset.gsm8k import Gsm8k
from ..dataset.svamp import Svamp
from ..dataset.multiarith import MultiArith
from ..dataset.addsub import Addsub
from ..llmut.openai import OpenAiModel
from ..llmut.palm import PaLMModel

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class SentEmbedding:
    
    @classmethod
    def load_bert_embedding_model(
        cls, 
        emb_model_name: str='all-MiniLM-L6-v2'
    ) -> SentenceTransformer:
        return SentenceTransformer(emb_model_name)

    @classmethod
    def get_embedding(
        cls,
        bert_emb_model: SentenceTransformer,
        input_text: str, 
    ) -> List[float]:
        #Sentences are encoded by calling model.encode()
        if type(input_text)==list:
            return bert_emb_model.encode(input_text)
        else:
            embeddings = bert_emb_model.encode([input_text])
            return embeddings[0,:]
        # end if


class KMeansClustering:
    
    @classmethod
    def list_to_np_arr(
        cls, 
        input_list: List[List[float]]
    ) -> np.array:
        return np.array(input_list)

    @classmethod
    def cluster(
        cls, 
        embeddings: np.array,
        **kwargs
    ) -> Dict:
        kmeans_obj = KMeans(
            n_clusters=kwargs.get('n_clusters', 8), 
            random_state=kwargs.get('random_state', 0), 
            n_init=kwargs.get('n_init', 'auto')
        ).fit(embeddings)
        labels = kmeans_obj.labels_
        emb_dist = kmeans_obj.transform(embeddings)
        cluster_res = {
            str(l): list()
            for l in sorted(set(labels))
        }
        for emb_i in range(len(embeddings)):
            label = labels[emb_i]
            emb_dist_from_centroid = emb_dist[emb_i, label]
            cluster_res[str(label)].append(
                (emb_i, emb_dist_from_centroid)
            )
        # end for

        # sort cluster by its distance
        for key in cluster_res.keys():
            cluster_res[key] = sorted(
                cluster_res[key], 
                key=lambda x: x[1]
            )
        # end for
        return cluster_res


class AutoCot:

    @classmethod
    def load_emb_model(
        cls,
        emb_model_name: str='all-MiniLM-L6-v2'
    ) -> SentenceTransformer:
        return SentEmbedding.load_bert_embedding_model(
            emb_model_name=emb_model_name
        )

    @classmethod
    def read_dataset(cls, dataset_name: str) -> List[Dict]:
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
    def get_embeddings(
        cls,
        dataset_name: str,
        emb_model_name: str='all-MiniLM-L6-v2'
    ) -> List[float]:
        res_dir = Macros.result_dir / 'autocot' / 'embeddings' / dataset_name
        res_dir.mkdir(parents=True, exist_ok=True)
        res_file = res_dir / 'embeddings.pkl'
        if os.path.exists(str(res_file)):
            with open(str(res_file), 'rb') as fp:
                res = pickle.load(fp)
            # end with
        else:
            dataset_obj = cls.read_dataset(dataset_name=dataset_name)
            bert_emb_model = cls.load_emb_model(
                emb_model_name=emb_model_name
            )
            input_texts = list()
            ids = list()
            cksum_vals = list()
            for d in dataset_obj:
                cksum_val = Utils.get_cksum(str(d['id']), length=7)
                body = d.get('body', '')
                question = d.get('question', '')
                ids.append(d['id'])
                cksum_vals.append(cksum_val)
                input_texts.append(f"{body} {question}")
            # end for
            embeddings = SentEmbedding.get_embedding(
                bert_emb_model, 
                input_texts
            )
            res = {
                'cksum_val': cksum_vals, # List[str]
                'input_text': input_texts, # List[str]
                'embedding': embeddings.tolist() # List[List[float]]
            }
            with open(str(res_file), 'wb') as fp:
                pickle.dump(res, fp)
            # end with
        # end if
        return res

    @classmethod
    def cluster(cls, embeddings: List[List[float]]):
        _embeddings = KMeansClustering.list_to_np_arr(embeddings)
        return KMeansClustering.cluster(_embeddings)

    @classmethod
    def get_answer_n_rationale(
        cls, 
        cksum_val: str,
        input_text: str, 
        model_name: str,
        dataset_name: str
    ) -> Dict:
        eval_dir = Macros.result_dir / 'nl2nl' / dataset_name / 'evaluate_consistency' / model_name
        eval_file = eval_dir / f"eval-{cksum_val}.json"
        if os.path.exists(str(eval_file)):
            eval_res = Utils.read_json(eval_file)
            return eval_res
        # end if
        return

    @classmethod
    def check_if_resp_meets_criteria(cls, resp_dict: Dict):
        # Criteria
        # set the selected demonstration d(i) as d(i)j if 
        # 1. it has a question q(i)j with no more than 60 tokens 
        # 2. and a rationale r(i)j with no more than 5 reasoning steps.
        # for second criteria, our existing response is hard to parse it to know how many reasoning steps are used.
        # therefore, for now I only implement first criteria
        question = resp_dict['orig']['question']
        tokens = Utils.tokenize(question)
        if len(tokens)<61:
            return True
        else:
            return False
        # end if

    @classmethod
    def construct_demo(
        cls, 
        input_text: str,
        embedding_res: str,
        cluster_res,
        model_name: str,
        dataset_name: str,
        prompt: str=Macros.openai_cot_prompt
    ) -> List[str]:  
        max_num_demos = 8
        demos = list()
        for cluster_label in cluster_res.keys():
            for emb_i, emb_dist_from_centroid in cluster_res[cluster_label]:
                # find cksum_vals
                cksum_val = embedding_res['cksum_val'][emb_i]
                _input_text = embedding_res['input_text'][emb_i]
                if input_text!=_input_text:
                    resp_res = cls.get_answer_n_rationale(
                        cksum_val,
                        _input_text, 
                        model_name,
                        dataset_name
                    )
                    if cls.check_if_resp_meets_criteria(resp_res):
                        question = resp_res['orig']['question']
                        cot_resp = resp_res['orig']['cot_response']
                        demos.append(f"Q: {question} {prompt}\nA: {cot_resp}\n")
                        break
                    # end if
                # end if
            # end for
        # end for
        return demos

    @classmethod
    def get_response(
        cls, 
        model_name: str,
        input_text: str,
        prompt: str,
        prompt_append: bool
    ) -> List[str]:
        response = None
        if model_name=='gpt3.5':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif model_name=='gpt4':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        elif model_name=='palm':
            response = PaLMModel.predict(
                input_text,
                prompt=prompt,
                prompt_append=prompt_append,
                model_name=Macros.palm_engine_name
            )
        elif model_name=='gpt4omini':
            response = OpenAiModel.predict(
                input_text, 
                prompt=prompt, 
                prompt_append=prompt_append
            )
        # end if
        return response

    @classmethod
    def main(
        cls,
        model_name: str,
        dataset_name: str,
        emb_model_name: str='all-MiniLM-L6-v2'
    ):
        embedding_res = cls.get_embeddings(
            dataset_name=dataset_name,
            emb_model_name=emb_model_name
        )
        cluster_res = cls.cluster(embedding_res['embedding'])

        res_dir = Macros.result_dir / 'autocot' / 'evaluate' / dataset_name / model_name
        res_dir.mkdir(parents=True, exist_ok=True)
        i = 0
        dataset_obj = cls.read_dataset(dataset_name=dataset_name)
        num_data = len(dataset_obj)
        for d in dataset_obj:
            cksum_val = Utils.get_cksum(str(d['id']), length=7)
            if not os.path.exists(str(res_dir / f"eval-{cksum_val}.json")):
                print(f"AUTOCOT::DATASET_{dataset_name}::{i} out of {num_data}::{cksum_val}")
                body = d.get('body', '')
                question = d.get('question', '')
                answer = d.get('answer', '')
                input_text = f"{body} {question}"

                demos = cls.construct_demo( 
                    input_text,
                    embedding_res,
                    cluster_res,
                    model_name,
                    dataset_name,
                    prompt=Macros.openai_cot_prompt
                )
                demo_prompt = ''
                for d_i in range(len(demos)):
                    demo_prompt += demos[d_i]
                # end for

                inp = f"{demo_prompt}Q: {input_text} {Macros.openai_cot_prompt}\nA: "
                resp_with_demo = cls.get_response(
                    model_name,
                    inp,
                    prompt='',
                    prompt_append=True
                )
                demo_res = {
                    'demos': demos,
                    'input_text': input_text,
                    'answer': answer,
                    'response': resp_with_demo,
                }

                Utils.write_json(
                    demo_res,
                    res_dir / f"eval-{cksum_val}.json",
                    pretty_format=True
                )
            # end if
            i += 1
        # end for
        return