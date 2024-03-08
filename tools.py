import os
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
from module import *

RUN_EXAMPLES = True

def is_interactive_notebook():
    return __name__ == "__main__"

"__name__ == __main__ and"
def show_example(fn, args=[]):
    if  RUN_EXAMPLES:
        return fn(*args)
    else:
        print(RUN_EXAMPLES)
        print(__name__)

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class Batch:
    """训练期间用于保存一批带掩码的数据的对象"""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        # 关于下面这一段：
        # (torch.tensor([[ 0, 2, 4, 5, 1, 0, 2 ]]) != 2).unsqueeze(-2)
        # print：tensor([[[ True, False,  True,  True,  True,  True, False]]])
        # 实际是把2元素打上掩码。
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # 下面两个分别去掉句子的开始符和结束符，这两个符合不参与运算
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            # 把输入指定位置的下三角掩码
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建一个掩码来隐藏并填充未来的word"
        # 和src一样，需要把<blank>符合盖住
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # 取&操作
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


class TrainState:
    """用来跟踪当前训练的情况，包括步数，梯度步数，样本使用数和已经处理的tokens数量"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def rate(step, model_size, factor, warmup):
    """
    我们必须将LambdaLR函数的最小步数默认为1。以避免零点导致的负数学习率。
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

class LabelSmoothing(nn.Module):
    "实现标签平滑.初始的smoothing=0.0即不进行平滑处理"

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # 定义一个KL散度loss网络，损失的计算方式是sum，求和
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        # 克隆一份作为真实分布
        true_dist = x.data.to("cuda:0")
        # 用smoothing/(self.size - 2) 填充
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 用confidence填充指定位置的数据，scatter_用法参考[8]
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # detach，把变量从计算图分离，可参考 https://zhuanlan.zhihu.com/p/389738863
        return self.criterion(x, true_dist.to("cuda:0").detach())


class SimpleLossCompute:
    "一个简单的损失计算和训练函数"

    def __init__(self, generator, criterion):
        """
        generator: Generator对象，用于根据Decoder的输出预测token
        criterion: LabelSmoothing对象，用于对Label进行平滑处理和损失计算
        """
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        # 这里顺便用最简单的例子展示了smoothing的用法
        # 先把decoder的x输入generator，得到预测的x
        # 然后把x和预测的y传入，criterion会对y做平滑处理，需要注意的是:
        # 这里传入的y展开成了一个一阶张量，即向量，因为在criterion内部会对它打包，会为每个单词生成一个概率向量
        x = self.generator(x)
        sloss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )

        # 这里又在搞事情，相当于第一个没有norm,第二个sloss是norm版本的，除以的是一个常量,batch.ntokens
        return sloss.data * norm, sloss
