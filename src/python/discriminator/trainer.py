
import os
import re
import sys
import torch
import logging

from tqdm import tqdm
from typing import *
from pathlib import Path

import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from .utils import save_config_file, accuracy, save_checkpoint

from ..utils.macros import Macros
from ..utils.utils import Utils


torch.manual_seed(0)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TrainDisc:

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        # self.tokenizer = kwargs['tokenizer']
        # self.model = kwargs['model'].to(self.args.device)
        self.tokenizer_query = kwargs['query_tokenizer']
        self.tokenizer_code = kwargs['code_tokenizer']
        self.model_query = kwargs['query_model'].to(self.args.device)
        self.model_code = kwargs['code_model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.res_dir = Macros.result_dir / 'discriminator'
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=self.res_dir / 'logs'
        )
        logging_file = self.writer.log_dir / 'discriminator_training.log'
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.device = self.args.device
        # if torch.cuda.is_available():
        #     self.model_query = torch.compile(self.model_query).to(self.device)
        #     self.model_code = torch.compile(self.model_code).to(self.device)
        # # end if
        logging.basicConfig(
            filename=str(logging_file), 
            level=logging.DEBUG
        )
        print(f"LOG_FILE: {logging_file}")

    def get_neg_code(self, batch):
        pass

    def info_nce_loss(self, features_query, features_code):
        # labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = torch.arange(self.args.batch_size)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features_query = F.normalize(features_query, dim=1) # features_query: (#batch, feat_dim)
        features_code = F.normalize(features_code, dim=1) # features_code: (#batch, feat_dim)
        similarity_matrix = torch.matmul(features_query, features_code.T)

        # similarity_matrix = torch.matmul(features, features.T)

        # # discard the main diagonal from both: labels and similarities matrix
        # mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        # labels = labels[~mask].view(labels.shape[0], -1)
        # similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start Discriminator training for {self.args.epochs} epochs.")

        for epoch_counter in range(self.args.epochs):
            for batch in tqdm(train_loader):

                batch_query = batch['query']
                batch_pos_code = batch['code']

                # batch_neg_codes = batch['neg_codes'] # batch_neg_codes: (#batch, #exs, dim)
                # print(type(batch_neg_codes))
                
                with autocast(enabled=self.args.fp16_precision):
                    # # encode query
                    # tok_batch_query = self.tokenizer(
                    #     batch_query,
                    #     padding='max_length',
                    #     truncation=True,
                    #     max_length=200,
                    #     return_tensors='pt'
                    # ).to(self.device)
                    # feat_query = self.model(tok_batch_query) # batch_query: (#batch, feat_dim)

                    # # encode positive code
                    # tok_batch_code = self.tokenizer(
                    #     batch_pos_code,
                    #     padding='max_length',
                    #     truncation=True,
                    #     max_length=200,
                    #     return_tensors='pt'
                    # ).to(self.device)
                    # feat_pos_code = self.model(tok_batch_code) # batch_query: (#batch, feat_dim)

                    # tokenize query
                    tok_batch_query = self.tokenizer_query(
                        batch_query,
                        padding='max_length',
                        truncation=True,
                        max_length=200,
                        return_tensors='pt'
                    ).to(self.device)

                    # tokenize code
                    tok_batch_code = self.tokenizer_code(
                        batch_pos_code,
                        padding='max_length',
                        truncation=True,
                        max_length=150,
                        return_tensors='pt'
                    ).to(self.device)

                    # encode query
                    feat_query = self.model_query(tok_batch_query) # batch_query: (#batch, feat_dim)

                    # encode positive code
                    feat_pos_code = self.model_code(tok_batch_code) # batch_query: (#batch, feat_dim)
                    
                    # # encode negative code
                    # # ----------
                    # feat_neg_codes = list()
                    # for b_i, batch_neg_code in enumerate(batch_neg_codes):
                    #     feat_neg_code = self.model_code(batch_neg_code) # feat_neg_code: (#batch, feat_dim)
                    #     feat_neg_codes.append(feat_neg_code)
                    # # end for
                    # feat_neg_codes = torch.stack(feat_neg_codes, dim=1) # feat_neg_code: (#batch, #exs, feat_dim)
                    # logits, labels = self.info_nce_loss(feat_query, feat_pos_code, feat_neg_codes)
                    # # ----------

                    logits, labels = self.info_nce_loss(feat_query, feat_pos_code)
                    loss = self.criterion(logits, labels)
                # end with

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                # end if
                n_iter += 1

            # Warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            # end if
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
        # end for

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = f"checkpoint_{self.args.epochs}.pth.tar"
        query_model_dir = self.res_dir / f"query_model_trained_from_{self.args.dataset_name}"
        code_model_dir = self.res_dir / f"code_model_trained_from_{self.args.dataset_name}"
        query_model_dir.mkdir(parents=True, exist_ok=True)
        code_model_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.query_model_name,
                'state_dict': self.model_query.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, 
            is_best=False, 
            filename=str(query_model_dir / checkpoint_name)
        )
        save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.code_model_name,
                'state_dict': self.model_code.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, 
            is_best=False, 
            filename=str(code_model_dir / checkpoint_name)
        )
        logging.info(f"Model checkpoint and metadata has been saved at\n\t{str(query_model_dir)}\n\t{str(code_model_dir)}.")
        return