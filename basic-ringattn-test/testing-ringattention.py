#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import math
from ring_attention_pytorch import RingAttention
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import math
torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = torch.utils._import_utils.dill_available()

def read_wikitext(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

train_text = read_wikitext('wiki.train.tokens')
valid_text = read_wikitext('wiki.valid.tokens')
test_text = read_wikitext('wiki.test.tokens')


# Initialize the tokenizer
tokenizer = get_tokenizer("basic_english")

# Load the WikiText2 dataset
train_iter, val_iter, test_iter = train_text, valid_text, test_text

# Define a function to yield tokens from the dataset
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Build the vocabulary from the training dataset
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Define a function to process the raw text and convert it to tensors
def data_process(raw_text_iter):
    data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# Process the train, validation, and test datasets
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

# Define a function to batchify the data
def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

# Set the batch sizes and batchify the data
batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

attn = RingAttention(
    dim = 512,
    dim_head = 64,
    heads = 8,
    causal = False,
    auto_shard_seq = True,
    ring_attn = True,
    ring_seq_size = 512
)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayerCustom(nn.Module):
    def __init__(self, d_model, nhead, d_hid, dropout=0.5):
        super(TransformerEncoderLayerCustom, self).__init__()
        self.self_attn = RingAttention(
            dim=d_model,
            dim_head=d_model // nhead,
            heads=nhead,
            causal=True,
            auto_shard_seq=True,
            ring_attn=True,
            ring_seq_size=d_model
        )
        self.linear1 = nn.Linear(d_model, d_hid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hid, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayerCustom(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        print("reached forward")
#         if self.src_mask is None or self.src_mask.size(0) != len(src):
#             device = src.device
#             mask = self._generate_square_subsequent_mask(len(src)).to(device)
#             self.src_mask = mask
        print("here1")
        src = self.encoder(src) * math.sqrt(self.d_model)
        print("here2")
        src = self.pos_encoder(src)
        print("here3")
        output = self.transformer_encoder(src, self.src_mask)
        print("here4")
        output = self.decoder(output)
        print("here5")
        return output

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train(model, train_data, criterion, optimizer, batch_size, bptt=35):
    model.train()  # Set the model to training mode
    total_loss = 0.  # Initialize the total loss to 0
    ntokens = len(vocab)  # Get the number of tokens in the vocabulary

    # Iterate through the mini-batches of data
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        print(1)
        data, targets = get_batch(train_data, i, bptt)  
        print(2)
        data, targets = data.to('cuda'), targets.to('cuda')  
        print(3)
        optimizer.zero_grad()
        print(4)
        output = model(data) 
        print(5)
        loss = criterion(output.view(-1, ntokens), targets) 
        print(6)
        loss.backward()  
        print(7)
        optimizer.step()  
        print(8)
        total_loss += loss.item() 
        print(9)

    return total_loss / (batch + 1)

def evaluate(model, data_source, criterion, batch_size, bptt=35):
    model.eval()  
    total_loss = 0.
    ntokens = len(vocab) 

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)  
            data, targets = data.to('cuda'), targets.to('cuda')  
            output = model(data)  
            loss = criterion(output.view(-1, ntokens), targets)  
            total_loss += loss.item()  

    return total_loss / (i + 1) 

device = torch.device("cuda")
ntokens = len(vocab) 
emsize = 512 
nhead = 8
nhid = 2048
nlayers = 6
dropout = 0.5 
batch_size = 32  
eval_batch_size = 10  

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)

model = nn.DataParallel(model)

model = model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 10
best_val_loss = float("inf")

for epoch in range(1, epochs + 1):
    print("Training...")
    train_loss = train(model, train_data, criterion, optimizer, batch_size)
    print("Evaluating...")
    val_loss = evaluate(model, val_data, criterion, eval_batch_size)
    print(f"Epoch: {epoch}, Train loss: {train_loss:.10f}, Validation loss: {val_loss:.10f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "transformer_wikitext2.pth")

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
model.load_state_dict(torch.load('transformer_wikitext2.pth'))
model.eval()
model = nn.DataParallel(model).to('cuda')
