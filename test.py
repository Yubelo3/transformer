from json import encoder
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device=torch.device("cuda")

epochs=100

# ================================================================================================

# encoder输入是需要翻译的句子，decoder输出是目标翻译，decoder
# P用来填充，保证矩阵维度相同。S表示Start，E表示End
sentences=[
    #encoder_input            decoder_input       decoder_output
    ["ich mochte ein bier P","S i want a beer .","i want a beer . E"],
    ["ich mochte ein cola P","S i want a coke .","i want a coke . E"]
]

# 正向索引 word:index
src_vocab={"P":0,"ich":1,"mochte":2,"ein":3,"bier":4,"cola":5}
# 反向索引 index:word
src_idx2word={i:w for i,w in enumerate(src_vocab)}
src_vocab_size=len(src_vocab)

tgt_vocab={"P":0,"i":1,"want":2,"a":3,"beer":4,"coke":5,"S":6,"E":7,".":8}
tgt_idx2word={i:w for i,w in enumerate(tgt_vocab)}

# 源句子长度
src_len=5
# 目标句子长度
tgt_len=6


# =================================================================================================

d_model=512 # Embedding size
d_ff=2048 # Feed-forward 512->2048->512
d_k=d_v=64 # K，Q，V维度, d_k*h=d_model
n_layers=6 # number of encoder and decoder layers
h=8 # number of heads


# =================================================================================================

def make_data(sentences:list):
    encoder_input,decoder_input,decoder_output=[],[],[]
    for i in range(len(sentences)): # i是样例序号
        encoder_input=[[src_vocab[n] for n in sentences[i][0].split()]] # 每个单词转为词表中的下标
        decoder_output=[[tgt_vocab[n] for n in sentences[i][1].split()]]
        decoder_output=[[tgt_vocab[n] for n in sentences[i][2].split()]]
        print(encoder_input)

make_data(sentences)