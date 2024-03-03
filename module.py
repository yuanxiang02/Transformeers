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

from tools import *

def attention(query, key, value, mask=None, dropout=None):
    "计算 '缩放点积注意力'"
    # 返回Query最后一个轴的长度，即d_k
    d_k = query.size(-1)
    # key.transpose实际上做的就是转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果存在掩码，使用掩码计算得分
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 按照最后一个轴计算softmax,即按照每行内进行softmax
    p_attn = scores.softmax(dim=-1)
    # 如果dropout存在则进行dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后和V相乘返回V的得分
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "接收多头的个数和维度进行初始化"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 假设d_v总是等于d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "实现图2"
        if mask is not None:
            # 对每一个头都用相同的掩码
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 这一段原文的投影应该指的就是图里面第一个块后面那些阴影，其实就是多头
        # 1) 批量计算线性投影从 d_model => h x d_k
        # 这一段很鸡贼，给了四个前馈线性层网络，这段打包只用了前三个，最后一个前馈网络什么时候用呢？最后return用。
        # 分别把query,key,value传给这三个线性层，然后reshape(这里用的view)成多头。然后得到了query,key,value
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) 计算attention
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) 在最后的线性层使用一个视图进行"Concat"，相当于把之前的多头变成单头
        # 关于调用contiguous原因 https://blog.csdn.net/weixin_43332715/article/details/124749348：
        # 1 transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，所以需要contiguous来返回一个contiguous copy；
        # 2 维度变换后的变量是之前变量的浅拷贝，指向同一区域，即view操作会连带原来的变量一同变形，这是不合法的，所以也会报错；---- 这个解释有部分道理，也即contiguous返回了tensor的深拷贝contiguous copy数据；

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value

        # 这里，用到了最后一个linear
        return self.linears[-1](x)


class EncoderDecoder(nn.Module):
    """
    一个标准的解码器，编码器模型
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "接收处理屏蔽的src和目标序列"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    "encoder - self src src_mask"
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    "decoder - self tgt memory src_mask tgt_mask"
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    定义一个标准的线性+softmax生成步骤。
    说人话，这个是用来接受最后的decode的结果，并且返回词典中每个词的概率
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        "为什么用log_softmax，主要是梯度和计算速度的考虑，可以百度一下，资料很多"
        return log_softmax(self.proj(x), dim=-1)

    "克隆N层"


class LayerNorm(nn.Module):
    "构造一个'层归一化'模块"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # -1 表示计算沿着张量的最后一维度的平均值。keepdim 是一个布尔值，用于指定是否保留维度。
        # 如果将 keepdim 设置为 True，则输出张量的形状将与输入张量的形状相同，只是最后一维的大小为 1。
        # 如果将 keepdim 设置为 False，则输出张量的形状将不包括最后一维。
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """每一个小的编码器的核心结构,由传入层(及其个数，后文使用的层个数是2)和层归一化组成
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "做前向传播需要依次传入每一个层，并且带上掩码"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    紧跟在层归一化模块后的残差链接
    注意，为了简化代码，先用norm而不是最后才使用(注意图1是sublayer计算后才norm)。
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "将残差层应用在所有大小相同的层"
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder 由自注意力层和前向网络层构成"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        "sublayer is Add & Norm -- x + drop(Sublayer(x))"
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "同图1左边的链接所示"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "一个带掩码的N层解码器通用结构"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "解码器是的自注意力模块是由编码器和解码器的att共同构成，再加上前馈网络"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 和编码器一样，共享SublayerConnection的结构，这个结构包括正则化和dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "图1右边所示的解码器结构即下面的代码"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 解码器第二个attn的k,v是编码器提供的输出，用编码器的x去查解码器的attn输出。
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "屏蔽后面的位置"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def example_mask():
    # 第一眼看这个嵌套循环给看懵了。其实就是用两个for循环生成了一个二维坐标，每一个都是一个df对象
    # 看下面这个就好理解了
    # 其实:=[(x,y) for y in range(20) for x in range(20)]

    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )
    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )

class PositionwiseFeedForward(nn.Module):
    "实现一个FFN模型"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "实现PE(位置编码)函数"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 在对数空间中计算位置编码。
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # requires_grad_(False)：禁用梯度下降
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
