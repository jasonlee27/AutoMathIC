
import torch
import argparse
import torch.backends.cudnn as cudnn

from typing import *
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from .models.transformer_disc import TransformerQueryEnc, TransformerCodeEnc
# from .models.transformer_disc import TransformerEncoder
from .trainer import TrainDisc
from .dataset import get_data_loader

from ..utils.macros import Macros
from ..utils.utils import Utils


parser = argparse.ArgumentParser(description='Discriminator')

# parser.add_argument('-data', metavar='DIR', default='./datasets',
#                     help='path to dataset')
parser.add_argument('--dataset-name', 
                    default='svamp',
                    help='dataset name', 
                    choices=['svamp', 'asdiv'])
parser.add_argument('--query_model_name', 
                    default='bert-base-uncased',
                    help='query model names for encoders')
parser.add_argument('--code_model_name',
                    default='bert-base-uncased',
                    help='code model names for encoders')
parser.add_argument('--epochs', 
                    default=100, 
                    type=int, 
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', 
                    default=12, 
                    type=int, 
                    metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', 
                    default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', 
                    default=0.00005,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', 
                    default=1e-4,
                    type=float,
                    metavar='W', 
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=Macros.RAND_SEED, type=int,
                    help='seed for initializing training. ')
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
parser.add_argument('--fp16-precision', 
                    action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', 
                    default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--log-every-n-steps', 
                    default=10, 
                    type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', 
                    default=0.07, 
                    type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', 
                    default=2, 
                    type=int, 
                    metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', 
                    default=0, 
                    type=int, 
                    help='Gpu index.')


def main():
    args = parser.parse_args()
    # assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    # end if

    # load dataset
    train_loader = get_data_loader(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size, 
        num_workers=args.workers,
        num_neg_codes=args.batch_size
    )

    # load models
    # tokenizer = AutoTokenizer.from_pretrained(args.query_model_name)
    # base_model = AutoModel.from_pretrained(args.query_model_name)
    # model = TransformerEncoder(
    #     base_model=base_model,
    #     out_dim=args.out_dim,
    #     base_model_finetuning=True
    # )
    tokenizer_query = AutoTokenizer.from_pretrained(args.query_model_name)
    base_model_query = AutoModel.from_pretrained(args.query_model_name)
    model_query = TransformerQueryEnc(
        base_model=base_model_query,
        out_dim=args.out_dim,
        base_model_finetuning=True
    )
    tokenizer_code = AutoTokenizer.from_pretrained(args.code_model_name)
    base_model_code = AutoModel.from_pretrained(args.code_model_name)
    model_code = TransformerCodeEnc(
        base_model=base_model_code,
        out_dim=args.out_dim,
        base_model_finetuning=True
    )
    params = list(model_query.parameters()) + list(model_code.parameters())

    optimizer = torch.optim.Adam(
        params, 
        args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader), 
        eta_min=0,
        last_epoch=-1
    )

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        # trainer = TrainDisc(
        #     tokenizer=tokenizer,
        #     model=model,
        #     optimizer=optimizer, 
        #     scheduler=scheduler, 
        #     args=args
        # )
        trainer = TrainDisc(
            query_tokenizer=tokenizer_query,
            code_tokenizer=tokenizer_code,
            query_model=model_query,
            code_model=model_code,
            optimizer=optimizer, 
            scheduler=scheduler, 
            args=args
        )
        trainer.train(train_loader)
    # end with
    return


if __name__ == "__main__":
    main()