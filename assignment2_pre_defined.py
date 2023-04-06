import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from pathlib import Path
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence, pad_packed_sequence
from collections import Counter

class SentimentModel(nn.Module):
  def __init__(self,vocab_size, hidden_size=128, num_layers=3, ):
    super().__init__()
    self.word_embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, dropout=0.3, batch_first=True)
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.final_layer = nn.Linear(hidden_size*2, 1)
    
  def forward(self, x):
    x = PackedSequence(self.word_embedding(x.data), batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
    x, hidden = self.gru(x)
    pad_x, lens = pad_packed_sequence(x, batch_first=True)

    max_x = torch.stack([torch.max(pad_x[i, :lens[i]], dim=0)[0] for i in range(len(lens))], dim=0)

    # max_x = torch.max(pad_x, dim=0)[0]
    pred_logit = self.final_layer(max_x)[:,0]
    return torch.sigmoid(pred_logit)

def get_train_txt_paths_in_split(dir_path='aclImdb/train', seed=0):
  dir_path = Path(dir_path)
  train_set, valid_set = [], []
  random.seed(seed) # manually seed random so that you can get the same random result whenever you run the code for reproducibility
  for typ in ('pos', 'neg'):
    paths_of_typ = list( (dir_path / typ).glob('*.txt'))
    num_examples = len(paths_of_typ)
    num_train_sample = num_examples * 4 // 5
    
    paths_of_typ = sorted(paths_of_typ)
    random.shuffle(paths_of_typ) # shuffle the dataset
    train_set += paths_of_typ[:num_train_sample] # assign first num_train_sample samples for train set
    valid_set += paths_of_typ[num_train_sample:] # assign the remaining samples for validation set
    
  random.shuffle(train_set)
  random.shuffle(valid_set)
  
  return train_set, valid_set

def read_txt(txt_path):
  with open(txt_path, 'r') as f:
    txt_string = f.readline()
  return txt_string

def make_vocab_from_txt_fns(txt_fns_list, tokenizer):
  '''
  This function takes a list of txt file paths and returns a list of all the words in the txt files
  '''
  vocab = Counter()
  for txt_fn in txt_fns_list:
    with open(txt_fn, 'r') as f:
      for line in f:
        vocab.update(tokenizer(line))
  return vocab
