
import os
import re
import math
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import *
from pathlib import Path
from numpy.linalg import norm

from ..utils.utils import Utils
from ..utils.macros import Macros
from ..utils.logger import Logger


from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer


class EmbeddingConsistency:

    @classmethod
    def load_model(
        cls, 
        emb_model_name: str='all-MiniLM-L6-v2'
    ) -> SentenceTransformer:
        model = SentenceTransformer(emb_model_name)
        return model

    @classmethod
    def get_embedding(
        cls, 
        sentences: List[str], 
        emb_model: SentenceTransformer
    ) -> Dict:
        #Sentences are encoded by calling model.encode()
        embeddings = emb_model.encode(sentences)
        emb_dict = dict()
        for s_i, s in enumerate(sentences):
            emb_dict[s] = embeddings[s_i,:]
        # end for
        return emb_dict

    @classmethod
    def gm_model_fit(
        cls, 
        embedding_dict: Dict, 
        need_tsne_model = False,
        gm_n_components: int=1
    ) -> GaussianMixture:
        embeddings = list(embedding_dict.values())
        embeddings = np.asarray(embeddings)
        tsne_model = None
        if need_tsne_model:
            tsne_model = TSNE(n_components=2, random_state=0, perplexity=3)
            embeddings = tsne_model.fit_transform(embeddings)
        # end if
        if embeddings.shape[0]>1:
            gm = GaussianMixture(
                n_components=gm_n_components, 
                random_state=0
            ).fit(embeddings)
            return gm, tsne_model
        # end if
        return embeddings, tsne_model

    @classmethod
    def cosine_sim(
        cls, 
        vec_a: np.array, 
        vec_b: np.array
    ):
        return np.dot(vec_a, vec_b)/(norm(vec_a)*norm(vec_b))

    @classmethod
    def vec_dist(
        cls, 
        vec_a: np.array, 
        vec_b: np.array
    ):
        return norm(vec_a-vec_b)

    @classmethod
    def fit_dist_model_by_mutations(
        cls,
        sentences: List[str],
        emb_model_name: str='all-MiniLM-L6-v2',
        gm_n_components: int=1
    ) -> GaussianMixture:
        model = cls.load_model(emb_model_name)
        embedding_dict = cls.get_embedding(sentences, model)
        gm, tsne_model = cls.gm_model_fit(
            embedding_dict, 
            gm_n_components=gm_n_components
        )
        return gm, tsne_model

    @classmethod
    def get_scores(
        cls, 
        orig_embedding_dict: Dict,
        mut_embedding_dict: Dict, 
        gm_model: GaussianMixture,
        tsne_model: TSNE=None
    ) -> Dict[str, float]:
        # embeddings = list(embedding_dict.values())
        # embeddings = np.asarray(embeddings)
        # if tsne_model is not None:
        #     embeddings = tsne_model.fit_transform(embeddings)
        # # end if

        # approach 1. ranking by context vector similarity between target and mutation
        orig_sent = list(orig_embedding_dict.keys())[0]
        orig_embed = orig_embedding_dict[orig_sent]
        score_dict_cossim = dict()
        for mut_sent, mut_embed in mut_embedding_dict.items():
            sim = cls.cosine_sim(orig_embed, mut_embed)
            score_dict_cossim[mut_sent] = sim
        # end for

        # approach 2. ranking by context vector distance between target and mutation
        orig_sent = list(orig_embedding_dict.keys())[0]
        orig_embed = orig_embedding_dict[orig_sent]
        score_dict_dist = dict()
        for mut_sent, mut_embed in mut_embedding_dict.items():
            dist = cls.vec_dist(orig_embed, mut_embed)
            sim = math.exp(-1.*dist)
            score_dict_dist[mut_sent] = sim
        # end for

        # approach 3. ranking by average distance of context vectors over mutations
        orig_sent = list(orig_embedding_dict.keys())[0]
        orig_embed = orig_embedding_dict[orig_sent]
        score_dict_avg_dist_among_muts = dict()
        for mut_sent, mut_embed in mut_embedding_dict.items():
            dist_list = list()
            for _mut_sent, _mut_embed in mut_embedding_dict.items():
                if mut_sent!=_mut_sent:
                    dist_list.append(
                        cls.vec_dist(_mut_embed, mut_embed)
                    )
                # end if
            # end for
            score_dict_avg_dist_among_muts[mut_sent] = 0.
            if any(dist_list):
                avg_dist = Utils.avg(dist_list, decimal=5)
                score_dict_avg_dist_among_muts[mut_sent] = math.exp(-1.*avg_dist)
            # end if
        # end for
        return {
            'cos_sim': score_dict_cossim,
            'dist': score_dict_dist,
            'avg_dist_among_muts': score_dict_avg_dist_among_muts
        }

    @classmethod
    def score_sentences_with_gm(
        cls,
        orig_sentence: str,
        mut_sentences: List[str],
        gm: GaussianMixture,
        tsne_model: TSNE=None,
        emb_model_name: str='all-MiniLM-L6-v2'
    ) -> Dict:
        model = cls.load_model(emb_model_name)
        orig_embed_dict = cls.get_embedding([orig_sentence], model)
        mut_embed_dict = cls.get_embedding(mut_sentences, model)
        score_dict = cls.get_scores(
            orig_embed_dict,
            mut_embed_dict, 
            gm, 
            tsne_model=tsne_model
        )
        return score_dict, \
            orig_embed_dict, \
            mut_embed_dict

    @classmethod
    def get_consistency(
        cls,
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python'
    ) -> None:
        # if dataset_name=='svamp':
        #     res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        # else:
        #     res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        # # end if
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        llm_response_files = sorted([
            f for f in os.listdir(str(eval_dir))
            if f.endswith('.json') and f.startswith('eval-') and \
            (not f.startswith('eval-results-w-'))
        ])
        # modal_consistency_res = Utils.read_json(eval_dir / 'consistency-results.json')
        cons_res = dict()
        for f_i, resp_file in enumerate(llm_response_files):
            resp_dict = Utils.read_json(
                eval_dir / resp_file
            )
            cksum_val = re.search(r'eval\-(.*)\.json', resp_file).group(1)
            # orig_modal_cons = modal_consistency_res[cksum_val]['orig']['consistency']
            orig_resp = resp_dict['orig']
            mut_resps = resp_dict['mutation']

            mut_cot_resps = [
                mut_resp['cot_response']
                for mut_resp in mut_resps
            ]
            # mut_modal_cons = [
            #     m['consistency']
            #     for m in modal_consistency_res[cksum_val]['mutation']
            # ]
            print(f"{f_i} out of {len(llm_response_files)-1}", cksum_val)

            gm, tsne_model = cls.fit_dist_model_by_mutations(mut_cot_resps)
            
            scores, \
            orig_embed_dict, \
            mut_embed_dict = cls.score_sentences_with_gm(
                orig_resp['cot_response'], 
                mut_cot_resps, 
                gm, 
                tsne_model=tsne_model
            )
            
            cons_res[cksum_val] = {
                key: {
                    m: scores[key][m] for m in mut_cot_resps
                }
                for key in scores.keys()
            }
            
            # # EXP: plot reduced embeddings
            # df = {
            #     'x': [
            #         embeddings[orig_resp['cot_response']][0]
            #     ] + [
            #         embeddings[m][0]
            #         for m in cons_res[cksum_val]['mutation'].keys()
            #     ] + [gm.means_[0,0]],
            #     'y': [
            #         embeddings[orig_resp['cot_response']][1]
            #     ] + [
            #         embeddings[m][1]
            #         for m in cons_res[cksum_val]['mutation'].keys()
            #     ] + [gm.means_[0,1]],
            #     'group': [
            #         'orig'
            #     ] + [
            #         'mut_con' if mut_modal_cons[m_i] else 'mut_incon'
            #         for m_i, _ in enumerate(cons_res[cksum_val]['mutation'].keys())
            #     ] + ['mean']
            # }
            # fig: plt.Figure = plt.figure()
            # ax: plt.Axes = fig.subplots()
            # ax = sns.scatterplot(
            #     data=df,
            #     x="x",
            #     y="y",
            #     hue='group',
            #     ax=ax
            # )
            # fig.tight_layout()
            # fig.savefig(str(eval_dir / f"tsne-emb-{cksum_val}.pdf"))
        # end for
        Utils.write_json(
            cons_res, 
            eval_dir / 'emb-consistency-results.json',
            pretty_format=True
        )
        return


class ModalConsistency:

    @classmethod
    def get_answer_from_ground_truth(cls, str_answer: str) -> str:
        return Utils.get_answer_from_ground_truth(str_answer)

    @classmethod
    def get_answer_from_cot_resp(cls, cot_resp: str) -> str:
        return Utils.get_answer_from_cot_resp(cot_resp)

    @classmethod
    def get_answer_from_code_resp(cls, code_resp: str, dataset_name: str) -> str:
        return Utils.get_answer_from_code_resp(code_resp, dataset_name)

    @classmethod
    def get_answer_from_eqn_resp(cls, eqn_resp: str) -> str:
        return Utils.get_answer_from_eqn_resp(eqn_resp)

    @classmethod
    def get_answer(cls, resp: str, mod_name: str, dataset_name: str) -> str:
        if mod_name=='cot_response':
            answer = cls.get_answer_from_cot_resp(resp)
        elif mod_name=='code_response':
            answer = cls.get_answer_from_code_resp(resp, dataset_name)
        elif mod_name=='eqn_response':
            answer = cls.get_answer_from_eqn_resp(resp)
        # end if
        return answer

    @classmethod
    def is_consistent(
        cls, 
        resp_dict: Dict,
        answer_gt: Any,
        dataset_name: str
    ) -> bool:
        is_cons = False

        answer_dict = dict()

        for mod_name in Macros.prompts_over_modals.keys():
            # prompt, is_prompt_append = Macros.prompts_over_modals[mod_name]
            ans = cls.get_answer(resp_dict[mod_name], mod_name, dataset_name) 
            try:
                answer_dict[mod_name] = eval(ans)
            except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                answer_dict[mod_name] = ans
                pass
            # end try
        # end for

        # answer_from_cot = cls.get_answer_from_cot_resp(cot_resp)
        # answer_from_code = cls.get_answer_from_code_resp(code_resp)
        # answer_from_cot = eval(answer_from_cot) if answer_from_cot is not None else answer_from_cot
        # answer_from_code = eval(answer_from_code) if answer_from_code is not None else answer_from_code
        answer_gt = cls.get_answer_from_ground_truth(answer_gt)
        if answer_gt is not None:
            if type(answer_gt)==str and answer_gt!='<N/A>':
                try:
                    answer_gt = eval(answer_gt)
                except (ValueError, SyntaxError, NameError, ZeroDivisionError, TypeError) as e:
                    pass
                # end try
            # end if

            answer_values = list(answer_dict.values())
            mod_cons = False
            if answer_values.count(answer_values[0])==len(answer_values):
                mod_cons = True
            # end if

            return {
                'consistency': mod_cons,
                'answer': answer_gt,
                'correctness': {
                    'cot': (answer_dict['cot_response'], answer_dict['cot_response']==answer_gt),
                    'code': (answer_dict['code_response'], answer_dict['code_response']==answer_gt),
                    'eqn': (answer_dict['eqn_response'], answer_dict['eqn_response']==answer_gt)
                }
            }
        # end if
        return

    @classmethod
    def get_consistency(
        cls,
        model_name: str, 
        dataset_name: str, 
        pl_type: str='python'
    ) -> None:
        # if dataset_name=='svamp':
        #     res_dir = Macros.result_dir / 'eq2code' / dataset_name / pl_type
        # else:
        #     res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        # # end if
        res_dir = Macros.result_dir / 'nl2nl' / dataset_name
        eval_dir = res_dir / 'evaluate_consistency' / model_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        llm_response_files = sorted([
            f for f in os.listdir(str(eval_dir))
            if f.endswith('.json') and f.startswith('eval-') and \
                (not f.startswith('eval-results-w-'))
        ])
        cons_res = dict()
        if os.path.exists(str(eval_dir / 'modal-consistency-results.json')):
            cons_res = Utils.read_json(eval_dir / 'modal-consistency-results.json')
        # end if

        for r_i, resp_file in enumerate(llm_response_files):
            resp_dict = Utils.read_json(
                eval_dir / resp_file
            )
            cksum_val = re.search(r'eval\-(.*)\.json', resp_file).group(1)

            if (cksum_val not in cons_res.keys()) or \
                (cksum_val in cons_res.keys() and len(cons_res[cksum_val]['orig']['correctness'].keys())!=len(Macros.prompts_over_modals.keys())):
                print(f"{r_i} OUT OF {len(llm_response_files)}::{cksum_val}")
                orig_resp = resp_dict['orig']
                orig_consist = cls.is_consistent(
                    orig_resp,
                    orig_resp['answer'],
                    dataset_name
                )
                print(orig_consist)
                if orig_consist is not None:
                    # for asdiv, question with the type of comparison has the answer with no numbers.
                    # in this case, we exclude questions of such type.
                    mut_consists = [
                        cls.is_consistent(
                            mut_resp,
                            mut_resp['answer'],
                            dataset_name
                        )
                        for mut_resp in resp_dict['mutation']
                    ]
                    cons_res[cksum_val] = {
                        'orig': orig_consist,
                        'mutation': mut_consists
                    }
                    Utils.write_json(
                        cons_res, 
                        eval_dir / 'modal-consistency-results.json',
                        pretty_format=True
                    )
                # end if
            # end if
        # end for
        Utils.write_json(
            cons_res, 
            eval_dir / 'modal-consistency-results.json',
            pretty_format=True
        )
        print(f"consistency checked for {len(cons_res.keys())} files out of {len(llm_response_files)}")
        return