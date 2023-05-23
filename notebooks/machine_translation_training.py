import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import transformers, tokenizers
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from tqdm.auto import tqdm


class Dataset:
  def __init__(self, df, src_tokenizer, tgt_tokenizer):
    self.data = df
    self.src_tokenizer = src_tokenizer
    self.tgt_tokenizer = tgt_tokenizer
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    selected_row = self.data.iloc[idx]
    source = selected_row['원문']
    target = selected_row['번역문']

    source_enc = self.src_tokenizer(source)['input_ids']
    target_enc = self.tgt_tokenizer(target)['input_ids']

    return torch.tensor(source_enc, dtype=torch.long), torch.tensor(target_enc[:-1], dtype=torch.long), torch.tensor(target_enc[1:], dtype=torch.long)


class Seq2seq(nn.Module):
  def __init__(self, enc_vocab, dec_vocab, hidden_size, num_layers=2):
    super().__init__()
    self.encoder = Encoder(enc_vocab, hidden_size, num_layers=num_layers)
    self.decoder = Decoder(dec_vocab, hidden_size, num_layers=num_layers)

  def forward(self, src, tgt):
    enc_out = self.encoder(src)
    if isinstance(src, PackedSequence) and isinstance(tgt, PackedSequence):
      enc_out = enc_out[:, src.unsorted_indices][:, tgt.sorted_indices ]
    dec_out = self.decoder(tgt, enc_out)
    return dec_out 

class Encoder(nn.Module):
  def __init__(self, num_vocab, hidden_size, num_layers=2):
    super().__init__()
    self.emb = nn.Embedding(num_vocab, hidden_size)
    self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
    # batch_first True: it takes (Num_samples_in_batch, num_timesteps, num_dim)
    # batch_first False: it takes (num_timesteps, Num_samples_in_batch, num_dim)

  def forward(self, x):
    if isinstance(x, PackedSequence):
      emb = PackedSequence(self.emb(x.data), batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
    else:
      emb = self.emb(x)
    out, last_hidden = self.rnn(emb)
    return last_hidden


class Decoder(nn.Module):
  def __init__(self, num_vocab, hidden_size, num_layers=2):
    super().__init__()
    self.emb = nn.Embedding(num_vocab, hidden_size)
    self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
    self.proj = nn.Linear(hidden_size, num_vocab)


  def forward(self, x, enc_output):
    if isinstance(x, PackedSequence):
      emb = PackedSequence(self.emb(x.data), batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
      out, last_hidden = self.rnn(emb, enc_output)
      logit = PackedSequence(self.proj(out.data), batch_sizes=out.batch_sizes, sorted_indices=out.sorted_indices, unsorted_indices=out.unsorted_indices)
    else:
      emb = self.emb(x)
      out, last_hidden = self.rnn(emb, enc_output)
      logit = self.proj(out)
    return logit


class Trainer:
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    
    self.model.to(device)
    
    self.best_valid_accuracy = 0
    self.device = device
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []

  def save_model(self, path='kor_eng_translator_model.pt'):
    torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, path)
    
  def train_by_num_epoch(self, num_epochs):
    for epoch in tqdm(range(num_epochs)):
      self.model.train()
      with tqdm(self.train_loader, leave=False) as pbar:
        for batch in pbar:
          loss_value = self._train_by_single_batch(batch)
          self.training_loss.append(loss_value)
          pbar.set_description(f"Epoch {epoch+1}, Loss {loss_value:.4f}")
      self.model.eval()
      validation_loss, validation_acc = self.validate()
      self.validation_loss.append(validation_loss)
      self.validation_acc.append(validation_acc)
      
      if validation_acc > self.best_valid_accuracy:
        print(f"Saving the model with best validation accuracy: Epoch {epoch+1}, Acc: {validation_acc:.4f} ")
        self.save_model('kor_eng_translator_model_best.pt')
      else:
        self.save_model('kor_eng_translator_model_last.pt')
      self.best_valid_accuracy = max(validation_acc, self.best_valid_accuracy)

      
  def _train_by_single_batch(self, batch):
    '''
    This method updates self.model's parameter with a given batch
    
    batch (tuple): (batch_of_input_text, batch_of_label)
    
    You have to use variables below:
    
    self.model (SentimentModel/torch.nn.Module): A neural network model
    self.optimizer (torch.optim.adam.Adam): Adam optimizer that optimizes model's parameter
    self.loss_fn (function): function for calculating BCE loss for a given prediction and target
    self.device (str): 'cuda' or 'cpu'

    output: loss (float): Mean binary cross entropy value for every sample in the training batch
    The model's parameters, optimizer's steps has to be updated inside this method

    TODO: Complete this method 
    '''
    src, tgt, shifted_tgt = batch
    src = src.to(self.device)
    tgt = tgt.to(self.device)
    shifted_tgt = shifted_tgt.to(self.device)

    logit = self.model(src, tgt)

    if isinstance(logit, PackedSequence):
      prob = logit.data.softmax(dim=-1)
      loss = self.loss_fn(prob, shifted_tgt.data)
    else:
      loss = self.loss_fn(prob, shifted_tgt)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    return loss.item()

    
  def validate(self, external_loader=None):
    '''
    This method calculates accuracy and loss for given data loader.
    It can be used for validation step, or to get test set result
    
    input:
      data_loader: If there is no data_loader given, use self.valid_loader as default.
      
    
    output: 
      validation_loss (float): Mean Binary Cross Entropy value for every sample in validation set
      validation_accuracy (float): Mean Accuracy value for every sample in validation set
      
    TODO: Complete this method 

    '''
    
    ### Don't change this part
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader
      
    self.model.eval()
    
    '''
    Write your code from here, using loader, self.model, self.loss_fn.
    '''

    validation_loss = 0
    num_correct_guess = 0
    num_data = 0
    with torch.inference_mode():
      for batch in tqdm(loader, leave=False):
        src, tgt, shifted_tgt = batch
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        shifted_tgt = shifted_tgt.to(self.device)

        logit = self.model(src, tgt)

        if isinstance(logit, PackedSequence):
          prob = logit.data.softmax(dim=-1)
          loss = self.loss_fn(prob, shifted_tgt.data)
        else:
          loss = self.loss_fn(prob, shifted_tgt)
        
        validation_loss += loss.item() * len(prob.data)
        num_correct_guess += (prob.argmax(dim=-1) == shifted_tgt.data).sum().item()
        num_data += len(prob.data)
    return validation_loss / num_data, num_correct_guess / num_data


def load_data_in_df(dataset_dir: Path):
  data_list = sorted(list(dataset_dir.glob('*.xlsx')))
  dfs = [pd.read_excel(path) for path in data_list]
  return pd.concat(dfs, axis=0)

def split_dataset(dataset):
  num_data = len(dataset)
  num_train = int(num_data * 0.8)
  num_valid = int(num_data * 0.1)
  num_test = num_data - (num_train + num_valid)

  train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_valid, num_test], generator=torch.Generator().manual_seed(42))
  return train_set, valid_set, test_set

def nll_loss(pred, target, eps=1e-8):
  if pred.ndim == 3:
    pred = pred.flatten(0, 1)
  if target.ndim == 2:
    target = target.flatten(0, 1)
  assert pred.ndim == 2
  assert target.ndim == 1  
  return -torch.log(pred[torch.arange(len(target)), target] + eps).mean()

def pack_collate(raw_batch):
  source, target, shifted_target = zip(*raw_batch)
  return pack_sequence(source, enforce_sorted=False), pack_sequence(target, enforce_sorted=False), pack_sequence(shifted_target, enforce_sorted=False)



def main():
  dataset_dir = Path('nia_korean_english')
  df = load_data_in_df(dataset_dir)
  # df = pd.read_csv('../nia_korean_english.csv')

  tokenizer_src = BertTokenizerFast.from_pretrained('hugging_kor_32000',
                                                       strip_accents=False,
                                                       lowercase=False) 
  tokenizer_tgt = BertTokenizerFast.from_pretrained('hugging_eng_32000',
                                                        strip_accents=False,
                                                        lowercase=False) 
  
  dataset = Dataset(df, tokenizer_src, tokenizer_tgt)
  train_set, valid_set, test_set = split_dataset(dataset)
  hidden_size = 256
  model = Seq2seq(tokenizer_src.vocab_size, tokenizer_tgt.vocab_size, hidden_size, num_layers=3)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=pack_collate, num_workers=8)
  valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=pack_collate, num_workers=8)
  device = 'cuda'

  trainer = Trainer(model, optimizer, nll_loss, train_loader, valid_loader, device)
  trainer.train_by_num_epoch(5)

  return

if __name__ == "__main__":
  main()