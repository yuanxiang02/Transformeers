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
import parser
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


def mask(size):
    "屏蔽后面的位置"
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return mask == 0

def data_gen(V, batch_size, nbatches):
    "为src-tgt复制任务生成一组随机的数据"
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        # 迭代器语法，yield
        yield Batch(src, tgt, 0)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """一个训练epoch
    data_iter: 可迭代对象，一次返回一个Batch对象或者加上索引
    model:训练的模型，这里就是Transformer
    loss_compute: SimpleLossCompute对象，用于计算损失
    optimizer: 优化器。这里是Adam优化器。验证时，optimizer是DummyOptimizer。DummyOptimizer不会真的更新模型参数，主要用于不同优化器效果的对比。
    scheduler：执行控制器。scheduler是一种用于调整优化器学习率的工具。 它可以帮助我们在训练过程中根据指定的策略调整学习率，
      以提高模型的性能这里是LambdaLR对象，用于调整Adam的学习率，实现WarmUp；验证时，scheduler是DummyScheduler。
    accum_iter: 每迭代n个batch更新一次模型的参数。这里默认n=1，就是每次batch都更新参数。
    train_state: TrainState对象，用于保存前训练的情况
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        # 注意这里的out是decoder输出的结果，这会还没有经过最后一层linear+softmax
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        # 这里才传入out和训练目标tgt_y计算了loss和loss_node。loss_node返回的是正则化的损失；
        # loss用来计算损失，loss_node用来梯度下降更新参数
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        # 只有在train或者train+log的模式才开启参数更新
        if mode == "train" or mode == "train+log":
            # 先通过backward计算出来梯度
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                # 调用依次梯度下降
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            # 我们在备注里提到过,scheduler的作用就是用来优化学习，控制学习率等超参数。这里调用step就是更新学习率相关的参数
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        # 下面是每40个epoch打印下相关日志
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            # 学习率
            lr = optimizer.param_groups[0]["lr"]
            # 40个epoch花费的时间
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def run_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else False)
    criterion = LabelSmoothing(size=args.number_classes, padding_idx=0, smoothing=args.smoothing)
    if os.path.exists("./weights") is False:
            os.mkdir("./weights")
    tb_writer = SummaryWriter()

    number_classes = args.number_classes
    batch_size = args.batch_size

    model = make_model(args.number_classes, args.number_classes, N=2).to(device)

    #optimizer - SGD Adam(AdaGrad/AdaDelta) 本质上是学习率的优化
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    for epoch in range(args.num_epochs):
        model.train()
        run_epoch(
            data_gen(number_classes, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        # Transformer作者开启了下面的验证代码，译者实在没看懂他想验证什么，所以默认取消了，如果想要保持和原作者一样可以取消下面代码注释
        # #####################取消下面注释##################### #
        # model.eval()
        # run_epoch(
        #     data_gen(V, batch_size, 5),
        #     model,
        #     SimpleLossCompute(model.generator, criterion),
        #     DummyOptimizer(),
        #     DummyScheduler(),
        #     mode="eval",
        # )[0]
        # #####################取消上面注释##################### #

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


# 训练比较耗时，这里就不默认启动了；启动把下面注释去掉即可


if __name__ == "__main__":
    path = r"D:\PycharmProjects"
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_classes", type=int, default=11)
    parser.add_argument("--epochs",type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate",type=float, default=0.001)
    parser.add_argument("--data-path", type=str, default=path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smoothing", type=float,default=0.0)
    opt = parser.parse_args()
    run_model(opt)

    run_model()
    #example_learning_schedule()
