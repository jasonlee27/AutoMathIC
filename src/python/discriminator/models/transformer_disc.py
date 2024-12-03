
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from ...utils.macros import Macros
from ...utils.utils import Utils


# class TransformerDiscriminator(nn.Module):

#     def __init__(
#         self, 
#         base_model: AutoModel,
#         dropout_p: float=0.03,
#         max_len: int=200,
#         base_model_finetuning: bool=False
#     ):
#         super(TransformerEncoder, self).__init__()
#         # self.base_query_model = AutoModel.from_pretrained(base_model_name)
#         self.base_model = base_model
#         self.base_model_finetuning = base_model_finetuning
#         if self.base_model_finetuning:
#             self.base_model.train()
#         else:
#             self.base_model.eval()
#         # end if
#         base_hidden_size = base_model.config.hidden_size
#         self.dropout = nn.Dropout(p=dropout_p)
#         self.linear = nn.Linear(base_hidden_size, 1)

#     def forward(self, query):
#         # run model with tokenized code
#         base_output = self.base_model(**query)
#         x = self.dropout(base_output.pooler_output)
#         feat_query = self.linear(x)
#         return feat_query

def load_pretrained_model(
    query_model_name: str,
    code_model_name: str,
    out_dim: int,
    pretrained_query_model_path: Path,
    pretrained_code_model_path: Path
):
    tokenizer_query = AutoTokenizer.from_pretrained(query_model_name)
    base_model_query = AutoModel.from_pretrained(query_model_name)
    model_query = TransformerQueryEnc(
        base_model=base_model_query,
        out_dim=out_dim,
        base_model_finetuning=False
    )
    tokenizer_code = AutoTokenizer.from_pretrained(code_model_name)
    base_model_code = AutoModel.from_pretrained(code_model_name)
    model_code = TransformerCodeEnc(
        base_model=base_model_code,
        out_dim=out_dim,
        base_model_finetuning=False
    )

    query_checkpoint = torch.load(
        str(pretrained_query_model_path)
    )
    model_query.load_state_dict(
        query_checkpoint['state_dict']
    )
    model_query.eval()

    code_checkpoint = torch.load(
        str(pretrained_code_model_path)
    )
    model_code.load_state_dict(
        code_checkpoint['state_dict']
    )
    model_code.eval()

    return tokenizer_query, \
        tokenizer_code,\
        model_query, \
        model_code

def get_correctness_score(
    tokenizer_query: AutoTokenizer, 
    tokenizer_code: AutoTokenizer,
    model_query: AutoModel, 
    model_code: AutoModel,
    query_text: str,
    code_text: str,
    device: Any
) -> float:
    # tokenize query
    tokenized_query = tokenizer_query(
        query_text,
        padding='max_length',
        truncation=True,
        max_length=200,
        return_tensors='pt'
    ).to(device)
    tokenized_code = tokenizer_code(
        code_text,
        padding='max_length',
        truncation=True,
        max_length=150,
        return_tensors='pt'
    ).to(device)

    # encode query
    feat_query = model_query(tokenized_query) # batch_query: (#batch, feat_dim)

    # encode positive code
    feat_code = model_code(tokenized_code) # batch_query: (#batch, feat_dim)

    feat_query = F.normalize(feat_query, dim=1)
    feat_code = F.normalize(feat_code, dim=1)

    score = torch.matmul(feat_query, feat_code.T)
    return score[0][0].item()


class TransformerQueryEnc(nn.Module):

    def __init__(
        self, 
        base_model: str,
        out_dim: int,
        dropout_p: float=0.03,
        max_len: int=200,
        base_model_finetuning: bool=False
    ):
        super(TransformerQueryEnc, self).__init__()
        # self.base_query_model = AutoModel.from_pretrained(base_model_name)
        self.base_model = base_model
        self.base_model_finetuning = base_model_finetuning
        if self.base_model_finetuning:
            self.base_model.train()
        else:
            self.base_model.eval()
        # end if
        base_hidden_size = base_model.config.hidden_size
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(base_hidden_size, out_dim)

    def forward(self, query):
        # run model with tokenized code
        base_output = self.base_model(**query)
        x = self.dropout(base_output.pooler_output)
        feat_query = self.linear(x)
        return feat_query


class TransformerCodeEnc(nn.Module):

    def __init__(
        self, 
        base_model: str,
        out_dim: int,
        dropout_p: float=0.03,
        max_len: int=100,
        base_model_finetuning: bool=False
    ):
        super(TransformerCodeEnc, self).__init__()
        # self.base_code_model = AutoModel.from_pretrained(base_model_name)
        self.base_model = base_model
        self.base_model_finetuning = base_model_finetuning
        if self.base_model_finetuning:
            self.base_model.train()
        else:
            self.base_model.eval()
        # end if
        base_hidden_size = base_model.config.hidden_size
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(base_hidden_size, out_dim)

    def forward(self, code):
        # run model with tokenized code
        base_output = self.base_model(**code)
        x = self.dropout(base_output.pooler_output)
        feat_code = self.linear(x)
        return feat_code