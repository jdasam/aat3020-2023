import torch
import torch.nn as nn
from tqdm.auto import tqdm

from torch.nn.utils.rnn import PackedSequence, pad_sequence, pack_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader

class TranslationSet:
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

    return torch.LongTensor(source_enc), torch.LongTensor(target_enc[:-1]), torch.LongTensor(target_enc[1:])
  

def pack_collate(raw_batch):
  source, target, shifted_target = zip(*raw_batch)
  return pack_sequence(source, enforce_sorted=False), pack_sequence(target, enforce_sorted=False), pack_sequence(shifted_target, enforce_sorted=False)



class TranslatorBi(nn.Module):
  def __init__(self, src_tokenizer, tgt_tokenizer, hidden_size=256, num_layers=3):
    super().__init__()
    self.src_tokenizer = src_tokenizer
    self.tgt_tokenizer = tgt_tokenizer
    
    self.src_vocab_size = self.src_tokenizer.vocab_size
    self.tgt_vocab_size = self.tgt_tokenizer.vocab_size
    
    self.src_embedder = nn.Embedding(self.src_vocab_size, hidden_size)
    self.tgt_embedder = nn.Embedding(self.tgt_vocab_size, hidden_size)
    
    self.encoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
    self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    
    self.decoder_proj = nn.Linear(hidden_size, self.tgt_vocab_size)
    
  def run_encoder(self, x):
    if isinstance(x, PackedSequence):
      emb_x = PackedSequence(self.src_embedder(x.data), batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
    else:
      emb_x = self.src_embedder(x)
      
    enc_hidden_state_by_t, last_hidden = self.encoder(emb_x)
    
    # Because we use bi-directional GRU, there are (num_layers * 2) last hidden states
    # Here, we make it to (num_layers) last hidden states by taking mean of [left-to-right-GRU] and [right-to-left-GRU]
    last_hidden_sum = last_hidden.reshape(self.encoder.num_layers, 2, last_hidden.shape[1], -1).mean(dim=1)
    if isinstance(x, PackedSequence):
      hidden_mean = enc_hidden_state_by_t.data.reshape(-1, 2, last_hidden_sum.shape[-1]).mean(1)
      enc_hidden_state_by_t = PackedSequence(hidden_mean, x[1], x[2], x[3])
    else:
      enc_hidden_state_by_t = enc_hidden_state_by_t.reshape(x.shape[0], x.shape[1], 2, -1).mean(dim=2)
      
    
    return enc_hidden_state_by_t, last_hidden_sum 

  def run_decoder(self, y, last_hidden_state):
    if isinstance(y, PackedSequence):
      emb_y = PackedSequence(self.tgt_embedder(y.data), batch_sizes=y.batch_sizes, sorted_indices=y.sorted_indices, unsorted_indices=y.unsorted_indices)
    else:
      emb_y = self.tgt_embedder(y)
    out, decoder_last_hidden = self.decoder(emb_y, last_hidden_state)
    return out, decoder_last_hidden

  def forward(self, x, y):
    '''
    x (torch.Tensor or PackedSequence): Batch of source sentences
    y (torch.Tensor or PackedSequence): Batch of target sentences
    '''
    
    enc_hidden_state_by_t, last_hidden_sum = self.run_encoder(x)
    out, decoder_last_hidden = self.run_decoder(y, last_hidden_sum)
    
    if isinstance(out, PackedSequence):
      logits = self.decoder_proj(out.data)
      probs = torch.softmax(logits, dim=-1)
      probs = PackedSequence(probs, batch_sizes=y.batch_sizes, sorted_indices=y.sorted_indices, unsorted_indices=y.unsorted_indices)
    else:
      logits = self.decoder_proj(out)
      probs = torch.softmax(logits, dim=-1)
    return probs
  

'''
Pre-defined class
'''




class Trainer:
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device, model_name='nmt_model'):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    
    self.model.to(device)
    
    self.grad_clip = 1.0
    self.best_valid_accuracy = 0
    self.device = device
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []
    self.model_name = model_name

  def save_model(self, path):
    torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, path)
    
  def train_by_num_epoch(self, num_epochs):
    for epoch in tqdm(range(num_epochs)):
      self.model.train()
      for batch in tqdm(self.train_loader, leave=False):
        loss_value = self._train_by_single_batch(batch)
        self.training_loss.append(loss_value)
      self.model.eval()
      validation_loss, validation_acc = self.validate()
      self.validation_loss.append(validation_loss)
      self.validation_acc.append(validation_acc)
      
      if validation_acc > self.best_valid_accuracy:
        print(f"Saving the model with best validation accuracy: Epoch {epoch+1}, Acc: {validation_acc:.4f} ")
        self.save_model(f'{self.model_name}_best.pt')
      else:
        self.save_model(f'{self.model_name}_last.pt')
      self.best_valid_accuracy = max(validation_acc, self.best_valid_accuracy)

      
  def _train_by_single_batch(self, batch):
    '''
    This method updates self.model's parameter with a given batch
    
    batch (tuple): (batch_of_input_text, batch_of_label)
    
    You have to use variables below:
    
    self.model (Translator/torch.nn.Module): A neural network model
    self.optimizer (torch.optim.adam.Adam): Adam optimizer that optimizes model's parameter
    self.loss_fn (function): function for calculating BCE loss for a given prediction and target
    self.device (str): 'cuda' or 'cpu'

    output: loss (float): Mean binary cross entropy value for every sample in the training batch
    The model's parameters, optimizer's steps has to be updated inside this method
    '''
    
    src, tgt_i, tgt_o = batch
    pred = self.model(src.to(self.device), tgt_i.to(self.device))
    loss = self.loss_fn(pred.data, tgt_o.data)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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
    validation_acc = 0
    num_total_tokens = 0
    with torch.no_grad():
      for batch in tqdm(loader, leave=False):
        
        src, tgt_i, tgt_o = batch
        pred = self.model(src.to(self.device), tgt_i.to(self.device))
        loss = self.loss_fn(pred.data, tgt_o.data)
        num_tokens = tgt_i.data.shape[0]
        validation_loss += loss.item() * num_tokens
        num_total_tokens += num_tokens
        
        acc = torch.sum(torch.argmax(pred.data, dim=-1) == tgt_o.to(self.device).data)
        validation_acc += acc.item()
        
    return validation_loss / num_total_tokens, validation_acc / num_total_tokens

def get_nll_loss(predicted_prob_distribution, indices_of_correct_token, eps=1e-10):
  '''
  for PackedSequence, the input is 2D tensor
  
  predicted_prob_distribution has a shape of [num_entire_tokens_in_the_batch x vocab_size]
  indices_of_correct_token has a shape of [num_entire_tokens_in_the_batch]
  '''
  prob_of_correct_next_word = predicted_prob_distribution[torch.arange(len(predicted_prob_distribution)), indices_of_correct_token]
  loss = -torch.log(prob_of_correct_next_word+eps)
  return loss.mean()


class TransformerTrainer(Trainer):
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device):
    super().__init__(model, optimizer, loss_fn, train_loader, valid_loader, device)
    self.num_iter = 0
    self._adjust_optim()

  def _adjust_optim(self):
    self.num_iter += 1 
    self.optimizer.param_groups[0]['lr'] = 512 ** (-0.5) * min(self.num_iter**(-0.5), self.num_iter*4000**(-1.5))

  def _train_by_single_batch(self, batch):
    '''
    This method updates self.model's parameter with a given batch
    
    batch (tuple): (batch_of_input_text, batch_of_label)
    
    You have to use variables below:
    
    self.model (Translator/torch.nn.Module): A neural network model
    self.optimizer (torch.optim.adam.Adam): Adam optimizer that optimizes model's parameter
    self.loss_fn (function): function for calculating BCE loss for a given prediction and target
    self.device (str): 'cuda' or 'cpu'

    output: loss (float): Mean binary cross entropy value for every sample in the training batch
    The model's parameters, optimizer's steps has to be updated inside this method
    '''
    
    src, tgt_i, tgt_o = batch
    pred = self.model(src.to(self.device), tgt_i.to(self.device))
    pred = pack_padded_sequence(pred, pad_packed_sequence(tgt_o)[1], batch_first=True, enforce_sorted=False)
    loss = self.loss_fn(pred.data, tgt_o.data)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    self._adjust_optim()
    
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
        src, tgt_i, tgt_o = batch
        tgt_o = tgt_o.to(self.device)
        pred = self.model(src.to(self.device), tgt_i.to(self.device))
        pred = pack_padded_sequence(pred, pad_packed_sequence(tgt_o)[1], batch_first=True, enforce_sorted=False)
        loss = self.loss_fn(pred.data, tgt_o.data)

        if isinstance(pred, PackedSequence):
          loss = self.loss_fn(pred.data, tgt_o.data)
        else:
          loss = self.loss_fn(pred, tgt_o)
        
        validation_loss += loss.item() * len(pred.data)
        if isinstance(pred, PackedSequence):
          num_correct_guess += (pred.data.argmax(dim=-1) == tgt_o.data).sum().item()
        else:
          num_correct_guess += (pred.argmax(dim=-1) == tgt_o.data).sum().item()
        num_data += len(pred.data)
    return validation_loss / num_data, num_correct_guess / num_data
  
def pad_collate(raw_batch):
  srcs = [x[0] for x in raw_batch]
  tgts_i = [x[1][:-1] for x in raw_batch]
  tgts_o = [x[1][1:] for x in raw_batch]
  
  srcs = pad_sequence(srcs, batch_first=True)
  tgts_i = pad_sequence(tgts_i, batch_first=True)
  tgts_o = pack_sequence(tgts_o, enforce_sorted=False)
  return srcs, tgts_i, tgts_o


class MLP(nn.Module):
  def __init__(self, in_size, hidden_size):
    super().__init__()
    self.input_size = in_size
    self.layer = nn.Sequential(nn.Linear(in_size, hidden_size),
                              nn.ReLU(),
                              nn.Linear(hidden_size, in_size))
  def forward(self, x):
    return self.layer(x)

class PosEncoding(nn.Module):
  def __init__(self, size, max_t):
    super().__init__()
    self.size = size
    self.max_t = max_t
    self.register_buffer('encoding', self._prepare_emb())
    
  def _prepare_emb(self):
    dim_axis = 10000**(torch.arange(self.size//2) * 2 / self.size)
    timesteps = torch.arange(self.max_t)
    pos_enc_in = timesteps.unsqueeze(1) / dim_axis.unsqueeze(0)
    pos_enc_sin = torch.sin(pos_enc_in)
    pos_enc_cos = torch.cos(pos_enc_in)

    pos_enc = torch.stack([pos_enc_sin, pos_enc_cos], dim=-1).reshape([self.max_t, 512])
    return pos_enc
    
  def forward(self, x):
    return self.encoding[x]

class ResidualLayerNormModule(nn.Module):
  def __init__(self, submodule):
    super().__init__()
    self.submodule = submodule
    self.layer_norm = nn.LayerNorm(self.submodule.input_size)

  def forward(self, x, mask=None, y=None):
    if y is not None:
      res_x = self.submodule(x, y, mask)
    elif mask is not None:
      res_x = self.submodule(x, mask)
    else:
      res_x = self.submodule(x)
    x =  x + res_x
    return self.layer_norm(x)

