
import os
import openai
import numpy as np
# import pandas as pd

from typing import *
from pathlib import Path
# from datasets import load_dataset
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

from ..utils.macros import Macros
from ..utils.utils import Utils
from ..utils.logger import Logger


class Embedding:

    def __init__(
        self, 
        embedding_model_name: str='openai',
        openai_embedding_model_name: str='text-embedding-ada-002',
        bert_emb_model_name: str='all-MiniLM-L6-v2'
    ):
        self.embedding_model_name = embedding_model_name
        self.openai_embedding_model_name = None
        self.bert_emb_model_name = None
        self.bert_emb_model = None
        if self.embedding_model_name=='openai':
            self.openai_embedding_model_name = openai_embedding_model_name
        elif self.embedding_model_name=='bert':
            self.bert_emb_model_name = bert_emb_model_name
            self.bert_emb_model = self.load_bert_embedding_model(bert_emb_model_name)
        # end if

    def get_openai_embedding(self, input_text: str) -> List[float]:
        text = input_text.replace("\n", " ")
        return openai.Embedding.create(
            input = [text], 
            model=self.openai_embedding_model_name
        )['data'][0]['embedding']

    def load_bert_embedding_model(
        self, 
        emb_model_name: str
    ) -> SentenceTransformer:
        return SentenceTransformer(emb_model_name)

    def get_bert_embedding(
        self,
        input_text: str, 
    ):
        #Sentences are encoded by calling model.encode()
        embeddings = self.bert_emb_model.encode([input_text])
        return embeddings[0,:]

    def get_embedding(self, input_text: str) -> List[float]:
        if self.embedding_model_name=='openai':
            return self.get_openai_embedding(input_text)
        elif self.embedding_model_name=='bert':
            return self.get_bert_embedding(input_text)
        # end if
        