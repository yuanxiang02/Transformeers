import os
import sys
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tools import *
from module import *
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "帮助: 从超参数中构建一个模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # 这里很重要
    # 用Glorot/fan_avg初始化参数。
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def inference_test():
    print("beegin test")
    test_model = make_model(11, 11, 2)
    test_model.eval()
    # 这里用LongTensor是因为怕单词长度过长。
    # torch.LongTensor 是一种整数类型的张量，它可以存储长整型数据。这种类型的张量通常用于存储索引数据，例如词汇表中的单词索引。
    # torch.Tensor 是一种浮点类型的张量，它可以存储单精度或双精度浮点数据。这种类型的张量通常用于存储神经网络中的权重参数或输入/输出数据。
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # shape=(1,1,10)，元素全部为1的mask
    src_mask = torch.ones(1, 1, 10)
    # 把encode输出的中间输出保存到memory，encode一步完成，不需要循环
    memory = test_model.encode(src, src_mask)
    # 创建一个和输入的张量类型一致的目标结果ys张量，一开始只有0起始符，所以shape=(1,1)，随着预测，shape=(1,n)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        # 传入memory,源mask,ys
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # 生成概率矩阵
        prob = test_model.generator(out[:, -1])
        # 取值和索引，我们只需要索引
        _, next_word = torch.max(prob, dim=1)
        # 取出单个词
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


if __name__ == "__main__":
    show_example(run_tests)
