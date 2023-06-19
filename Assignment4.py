import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

from torch.nn.utils.rnn import PackedSequence, pad_sequence, pack_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from assignment4_pre_defined import TranslatorBi, TranslationSet, pack_collate, Trainer, ResidualLayerNormModule, MLP, PosEncoding


def get_attention_score_for_a_single_query(keys, query):
  '''
  This function returns an attention score for each vector in keys for a given query.
  You can regard 'keys' as hidden states over timestep of Encoder, while query is a hidden state of specific time step of Decoder
  Name 'keys' are used because it is used for calculating attention score (match rate between given vector and query).
  
  For every C-dimensional vector key, the attention score is a dot product between the key and the query vector.
  
  Arguments:
    keys (torch.Tensor): Has a shape of [T, C]. These are vectors that a query wants attend to
    query (torch.Tensor): Has a shape of [C]. This is a vector that attends to other set of vectors (keys and values)
  
  Output:
    attention_score (torch.Tensor): The attention score in real number that represent how much does query have to attend to each vector in keys
                                    Has a shape of [T]
                                    
    attention_score[i] has to be a dot product value between keys[i] and query                                 


  TODO: Complete this sentence using torch.mm (matrix multiplication)
  Hint: You can use atensor.unsqueeze(dim) to expand a dimension (with a diemsion of length 1) without changing item value of the tensor.
  '''
  
  return

def get_attention_weight_from_score(attention_score):
  '''
  This function converts attention score to attention weight.
  
  Argument:
    attention_score (torch.Tensor): Tensor of real number. Has a shape of [T]

  Output:
    attention_weight (torch.Tensor): Tensor of real number between 0 and 1. Sum of attention_weight is 1. Has a shape of [T]
  
  TODO: Complete this function
  '''
  assert attention_score.ndim == 1
  
  return

def get_weighted_sum(values, attention_weight):
  '''
  This function converts attention score to attention weight
  
  Argument:
    values (torch.Tensor): Has a shape of [T, C]. These are vectors that are used to form attention vector
    attention_weight: Has a shape of [T], which represents the weight for each vector to compose the attention vector

  Output:
    attention_vector (torch.Tensor): Weighted sum of values using the attention weight. Has a shape of [C]
  
  TODO: Complete this function using torch.mm
  '''
  return


def get_attention_score_for_a_batch_query(keys, query):
  '''
  This function returns a batch of attention score for each vector in (multi-batch) keys for a given (single-batch) query.
  You can regard 'keys' as hidden states over timestep of Encoder, while query is a hidden state of specific time step of Decoder
  Name 'keys' are used because it is used for calculating attention score (match rate between given vector and query).
  
  For every C-dimensional vector key, the attention score is a dot product between the key and the query vector.
  
  Arguments:
    keys (torch.Tensor): Has a shape of [N, T, C]. These are vectors that a query wants attend to
    query (torch.Tensor): Has a shape of [N, C]. This is a vector that attends to other set of vectors (keys and values)
  
  Output:
    attention_score (torch.Tensor): The attention score in real number that represent how much does query have to attend to each vector in keys
                                    Has a shape of [N, T]
                                    
    attention_score[n, i] has to be a dot product value between keys[n, i] and query[n]                     
    
  TODO: Complete this function without using for loop
  Hint: Use torch.bmm or torch.matmul after make two input tensors as 3-dim tensors.

  '''
  return 

def get_attention_score_for_a_batch_multiple_query(keys, queries):
  '''
  Now you have to implement the attention score for not only single query, but multiple queries.
  
  This function returns a batch of attention score for each vector in keys for given queries.
  You can regard 'keys' as hidden states over timestep of Encoder, while querys are hidden states over timestep of Decoder
  Name 'keys' are used because it is used for calculating attention score (match rate between given vector and query).
  
  For every C-dimensional vector key, the attention score is a dot product between the key and the query vector.
  
  Arguments:
    keys (torch.Tensor): Has a shape of [N, Ts, C]. These are vectors that a query wants attend to
    queries (torch.Tensor): Has a shape of [N, Tt, C]. This is a vector that attends to other set of vectors (keys and values)
  
  Output:
    attention_score (torch.Tensor): The attention score in real number that represent how much does query have to attend to each vector in keys
                                    Has a shape of [N, Ts, Tt]
                                    
    attention_score[n, i, t] has to be a dot product value between keys[n, i] and query[n, t] 
    
  TODO: Complete this function without using for loop
  HINT: Use torch.bmm() with proper transpose (permutation) of given tensors. (You can use atensor.permute())
        Think about which dimension (axis) of tensors has to be multiplied together and resolved (disappear) after matrix multiplication,
        and how the result tensor has to look like (shape)
  '''
  return


def get_masked_softmax(attention_score, mask, mask_value=-1e10):
  '''
  During the batch computation, each sequence in the batch can have different length.
  To group them as in a single tensor, we usually pad values
    
  Arguments:
    attention_score (torch.Tensor): The attention score in real number that represent how much does query have to attend to each vector in keys
                                    Has a shape of [N, Ts, Tt]
    mask (torch.Tensor): Boolean tensor with a shape of [N, Ts] that represents whether the corresponding is valid or not.
                         mask[n, t] == 1 if and only if input_batch[n,t] is not a padded value.
                         If input_batch[n,t] is a padded value, then mask[n,t] == 0
  
  Output:
    attention_weight (torch.Tensor): The attention weight in real number between 0 and 1. The sum of attention_weight along keys timestep dimension is 1.
                                    Has a shape of [N, Ts, Tt]
                                    
    attention_weight[n, i, t] has to be an attention weight of values[n, i] for queries[n, t] 
    
  TODO: Complete this function without using for loop
  Hint: You can give -infinity value by -float("inf")

  '''

  return

def get_batch_weighted_sum(values, attention_weight):
  '''
  This function converts attention score to attention weight
  
  Argument:
    values (torch.Tensor): Has a shape of [N, Ts, C]. These are vectors that are used to form attention vector
    attention_weight: Has a shape of [N, Ts, Tt], which represents the weight for each vector to compose the attention vector
                      attention_weight[n, s, t] represents weight for value[n, s] that corresponds to a given query, queries[n, t]

  Output:
    attention_vector (torch.Tensor): Weighted sum of values using the attention weight. 
                                     Has a shape of [N, Tt, C]
  
  TODO: Complete this function using torch.bmm
  '''
  
  return



class TranslatorAtt(TranslatorBi):
  def __init__(self, src_tokenizer, tgt_tokenizer, hidden_size=512, num_layers=3):
    super().__init__(src_tokenizer, tgt_tokenizer, hidden_size, num_layers)
    
    # TODO: define new self.decoder_proj
    self.decoder_proj = nn.Linear(hidden_size * 2, self.tgt_vocab_size)
    
  def get_attention_vector(self, encoder_hidden_states, decoder_hidden_states, mask):
    '''
    Arguments:
      encoder_hidden_states (torch.Tensor or PackedSequence): Hidden states of encoder GRU. Shape: [N, Ts, C]
      decoder_hidden_states (torch.Tensor or PackedSequence): Hidden states of decoder GRU. Shape: [N, Tt, C]
      mask (torch.Tensor): Masking tensor. If the mask value is 0, the attention weight has to be zero. Shape: [N, Tt, Ts]

    Outputs:
      attention_vectors (torch.Tensor or PackedSequence): Attention vectors that has the same shape as decoder_hidden_states
      attention_weights (torch.Tensor): Zero-padded attention weights.
                                You don't need to return it during the training, but it will help you to implement later problem
    
    TODO: Complete this function using following functions
      get_attention_score_for_a_batch_multiple_query
      get_masked_softmax
      get_batch_weighted_sum
    If the inputs are PackedSequence, the output has to be a PackedSequence
    Use torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=True) to convert PackedSequence to Tensor
    Use torch.nn.utils.rnn.pack_padded_sequence(tensor, batch_lens, batch_first=True) to convert Tensor to PackedSequence
    '''
    is_packed = isinstance(encoder_hidden_states, PackedSequence)
    if is_packed:
      encoder_hidden_states, source_lens = pad_packed_sequence(encoder_hidden_states, batch_first=True)
      decoder_hidden_states, target_lens = pad_packed_sequence(decoder_hidden_states, batch_first=True)
    
    # Write your code from here

    # 1. Calculate attention score using encoder_hidden_states and decoder_hidden_states
    # 2. Mask the attention score using mask and apply softmax to get attention weight
    # 3. Calculate attention vector using attention weight and encoder_hidden_states

    
    # 


    return 
  
  def forward(self, x, y):
    '''
    Arguments:
      x (torch.Tensor or PackedSequence): Batch of source sentences
      y (torch.Tensor or PackedSequence): Batch of target sentences
    Output:
      prob_dist (torch.Tensor or PackedSequence): Batch of probability distribution of word for target sentence
    
    TODO: Complete this function
    '''

    is_packed = isinstance(x, PackedSequence)
    enc_hidden_state_by_t, last_hidden_sum = self.run_encoder(x)
    dec_hidden_state_by_t, decoder_last_hidden = self.run_decoder(y, last_hidden_sum)
    
    if is_packed:
      mask = pad_packed_sequence(x, batch_first=True)[0] != 0
    else:
      mask = torch.ones(x.shape[0], x.shape[1])

    attention_vec = self.get_attention_vector(enc_hidden_state_by_t, dec_hidden_state_by_t, mask)

    # TODO: Write your code from here
    # CAUTION: 
    #   For the concatenation, you have to concat [dec_hidden_state_by_t; attention_vec], not [attention_vec; dec_hidden_state_by_t]
    return

def translate(model, source_sentence):
  '''
  
  Arguments:
    model (TranslatorAtt): Translator model with attention
    source_sentence (str): Sentence to translate

  Returns:
    input_tokens (list): Source sentence in a list of token in token_id
    predicted_tokens (list): Translated sentence in a list of token in token_id
    decoded_string (str): Translated sentence in string
    attention_map (torch.Tensor): Attention weight between each token of source sentence and target sentence. Has a shape of [Ts, Tt]
    
  '''
  
  input_tokens = model.src_tokenizer.encode(source_sentence)
  input_tensor = torch.LongTensor(input_tokens).unsqueeze(0)
  mask = torch.ones_like(input_tensor)
  enc_hidden_state_by_t, last_hidden_sum = model.run_encoder(input_tensor)
  
  # Setup for 0th step
  current_hidden = last_hidden_sum
  current_decoder_token = torch.LongTensor([[2]]) # start of sentence token
  total_output = []
  total_attetion_weights = []

  for i in range(100): # You can chage it to while True:
    emb = model.tgt_embedder(current_decoder_token)
    '''
    TODO: Complete the code here
    
    You have to 
      1) run decoder rnn for a single step
      2) get attention weight (variable name: att_weight) and attention vector.
         att_weight.shape == torch.Size([1, len(tokenized_sentence), 1])
      3) concat decoder out and attention vector
      4) calculate probabilty logit (variable name: logit)
    '''


    # You don't have to change the codes below.
    # Declare logit and last_hidden properly so that the code below can run without error
    selected_token = torch.argmax(logit, dim=-1)
    current_decoder_token = selected_token
    current_hidden = last_hidden
    if current_decoder_token == 3: ## end of sentence token
      break
    total_output.append(selected_token[0])
    total_attetion_weights.append(att_weight[0,:,0])
  predicted_tokens = torch.cat(total_output, dim=0).tolist()
  attention_map = torch.stack(total_attetion_weights, dim=1)
  
  return  input_tokens, predicted_tokens, model.tgt_tokenizer.decode(predicted_tokens), attention_map

def get_query_key_value(input_tensor, qkv_layer):
  '''
  This function returns key, query, and value that is calculated by input tensor and nn_layer.

  Arguments:
    input_tensor (torch.Tensor): Has a shape of [N, T, C]
    kqv_layer (torch.nn.Linear): Linear layer with in_features=C and out_features=Cn * 3
    
  Outputs:
    queries (torch.Tensor): Has a shape of [N, T, Cn]
    keys (torch.Tensor): Has a shape of [N, T, Cn]
    values (torch.Tensor): Has a shape of [N, T, Cn]

  TODO: Complete this function without using for loop
  Hint: Use torch.chunk() to split a tensor into given number of chunks
  '''
  return

def get_3d_masked_softmax(attention_score, mask):
  '''
  During the batch computation, each sequence in the batch can have different length.
  To group them as in a single tensor, we usually pad values
    
  Arguments:
    attention_score (torch.Tensor): The attention score in real number that represent how much does query have to attend to each vector in keys
                                    Has a shape of [N, Tk, Tq]
    mask (torch.Tensor): Boolean tensor with a shape of [N, Tk, Tq] that represents whether the corresponding is valid or not.
                         mask[n, tk, tq] == 1 if and only if input_batch[n,tk] is not a padded value.
                         If input_batch[n,tk] is a padded value, then mask[n,tk, tq] == 0
  
  Output:
    attention_weight (torch.Tensor): The attention weight in real number between 0 and 1. The sum of attention_weight along keys timestep dimension is 1.
                                    Has a shape of [N, Tk, Tq]
                                    
    attention_weight[n, i, t] has to be an attention weight of values[n, i] for queries[n, t] 
    
  TODO: Complete this function without using for loop

  '''
  assert attention_score.ndim == 3 and mask.ndim == 3

  return

def get_self_attention(input_tensor, qkv_layer, mask):
  '''
  This function returns output of self-attention for a given input tensor using with a given kqv_layer
  
  Arguments:
    input_tensor (torch.Tensor): Has a shape of [N, T, C]
    kqv_layer (torch.nn.Linear): Linear layer with in_features=C and out_features=Cn * 3
    mask (torch.Tensor): 
    
  Outputs:
    output (torch.Tensor): Has a shape of [N, T, Cn]

  TODO: Complete this function using your completed functions of below:
        get_query_key_value()
        get_attention_score_for_a_batch_multiple_query()
        get_3d_masked_softmax()
        get_batch_weighted_sum()
  '''
  return

def get_multihead_split(x, num_head):
  '''
  This function returns split tensor for multi-head attention

  Arguments:
    x (torch.Tensor): Has a shape of [N, T, C]
    num_head (int): Number of heads

  Output:
    x (torch.Tensor): Has a shape of [N * num_head, T, C // num_head]
    The order of N * num_head is [Batch1_head1, Batch1_head2, ..., Batch1_headN, Batch2_head1, Batch2_head2, ..., Batch2_headN, ...]
  '''
  assert x.shape[-1] % num_head == 0

  # TODO: Complete this function
  return

def get_multihead_concat(x, num_head):
  '''
  This function returns concat tensor for multi-head attention

  Arguments:
    x (torch.Tensor): Has a shape of [N * num_head, T, C // num_head]
    num_head (int): Number of heads
  Outputs:
    x (torch.Tensor): Has a shape of [N, T, C]
  '''
  # TODO: Complete this function
  assert x.shape[0] % num_head == 0

  return


def get_multi_head_self_attention(input_tensor, qkv_layer, output_proj_layer, mask, num_head=8):
  '''
  This function returns output of multi-headed self-attention for a given input tensor using with a given kqv_layer
  
  Arguments:
    input_tensor (torch.Tensor): Has a shape of [N, T, C]
    qkv_layer (torch.nn.Linear): Linear layer with in_features=C and out_features=Cn * 3
    output_proj_layer (torch.nn.Linear): Linear layer with in_features=Cn and out_features=C
    mask (torch.Tensor): Boolean tensor with a shape of [N, Ts] that represents whether the corresponding is valid or not.
                         mask[n, t] == 1 if and only if input_batch[n,t] is not a padded value.
                         If input_batch[n,t] is a padded value, then mask[n,t] == 0
    num_head (int): Number of heads
    
  Outputs:
    output (torch.Tensor): Has a shape of [N, T, Cn]

  TODO: Complete this function using your completed functions of below:
        get_query_key_value(): Get QKV from input_tensor and kqv_layer

        get_multihead_split(): Split QKV into multiple heads

        get_attention_score_for_a_batch_multiple_query(): Get attention score for a batch of multiple queries
          CAUTION: You have to scale the attention score by dividing by sqrt(Cn // num_head)
          HINT: att_score /= keys.shape[-1] ** 0.5

        get_3d_masked_softmax(): Get masked softmax/
          CAUTION: You have to repeat mask for num_head times to use it for multi-head attention
          USE head_repeated_mask
        get_batch_weighted_sum(): Get batch weighted sum

        get_multihead_concat(): Concatenate multiple heads into a single tensor

        Additionally, use output_proj_layer to project concatenated tensor at the final step
  '''
  head_repeated_mask = mask.unsqueeze(1).repeat(1, num_head, 1, 1).reshape(-1, mask.shape[1], mask.shape[2])

  return

class SelfAttention(nn.Module):
  def __init__(self, input_size, hidden_size, num_head, mask_value=0):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.qkv = nn.Linear(self.input_size, self.hidden_size * 3)
    self.out_proj = nn.Linear(self.hidden_size, self.input_size)
    self.mask_value = mask_value
    self.num_head = num_head 
    assert self.hidden_size % self.num_head == 0
    self.dim_per_head = self.hidden_size // self.num_head
    
  '''
  TODO: Implement this function as functions you implemented above
  '''
  def _get_qkv(self, x):
    return
  
  def _get_multihead_split(self, x):
    return 
  
  def _get_multiheaded_att_score(self, keys, queries):
    return 
  
  def _get_masked_softmax(self, score, masks):
    return 
  
  def _get_weighted_sum(self, values, weights):
    return 
  
  def forward(self, x, mask=None):
    '''
    TODO: Implement this function using the functions you implemented above
    '''
    if mask is None:
      mask = torch.ones([x.shape[0], x.shape[1], x.shape[1]])
    
    return
  
class CrossAttention(SelfAttention):
  def __init__(self, input_size, hidden_size, num_head, mask_value=0):
    super().__init__(input_size, hidden_size, num_head, mask_value)
    
  def forward(self, x, y, mask=None):
    '''
    Arguments:
      x (torch.Tensor): Sequence to be used for query
      y (torch.Tensor): Sequence to be used for key and value
      mask (torch.Tensor): Masking tensor. If the mask value is 0, the attention weight has to be zero. Shape: [N, Ty, Tx]

    Outs:
      output (torch.Tensor): Output of cross attention. Shape: [N, Tx, C]

    TODO: Complete this function using your completed functions of below:
    '''
    if mask is None:
      mask = torch.ones([x.shape[0], y.shape[1], x.shape[1]])

    return 

class EncoderLayer(nn.Module):
  def __init__(self, in_size, emb_size, mlp_size, num_head):
    super().__init__()
    self.att_block = ResidualLayerNormModule(SelfAttention(in_size, emb_size, num_head))
    self.mlp_block = ResidualLayerNormModule(MLP(emb_size, mlp_size))
  
  def forward(self, x):
    out = self.mlp_block(self.att_block(x['input'], x['mask']))
    return {'input':out, 'mask':x['mask']}


class DecoderLayer(nn.Module):
  def __init__(self, in_size, emb_size, mlp_size, num_head):
    super().__init__()
    self.att_block = ResidualLayerNormModule(SelfAttention(in_size, emb_size, num_head))
    self.cross_att_block = ResidualLayerNormModule(CrossAttention(in_size, emb_size, num_head))
    self.mlp_block = ResidualLayerNormModule(MLP(emb_size, mlp_size))
  
  def forward(self, x):
    out = self.att_block(x['input'], x['decoder_mask'])
    out = self.cross_att_block(out,  x['encoder_mask'], x['encoder_out'])
    out = self.mlp_block(out)
    return {'input':out, 'decoder_mask':x['decoder_mask'], 'encoder_out':x['encoder_out'], 'encoder_mask':x['encoder_mask']}

class Encoder(nn.Module):
  def __init__(self, in_size, emb_size, mlp_size, num_head, num_layers, vocab_size):
    super().__init__()
    self.layers = nn.Sequential()
    for i in range(num_layers):
      self.layers.append(EncoderLayer(in_size,emb_size,mlp_size,num_head))
    self.pos_enc = PosEncoding(emb_size, 10000)
    self.token_emb = nn.Embedding(vocab_size, emb_size)
    
  def forward(self, x):
    mask = torch.ones([x.shape[0], x.shape[1], x.shape[1]])
    mask[x==0] = 0
    temp = torch.ones_like(x)
    result = torch.arange(x.shape[-1]).to(x.device) * temp
    x = self.token_emb(x) + self.pos_enc(result)
    return self.layers({'input':x, 'mask':mask})
  
class Decoder(nn.Module):
  def __init__(self, in_size, emb_size, mlp_size, num_head, num_layers, vocab_size):
    super().__init__()
    self.layers = nn.Sequential()
    for i in range(num_layers):
      self.layers.append(DecoderLayer(in_size,emb_size,mlp_size,num_head))
    self.pos_enc = PosEncoding(emb_size, 10000)
    self.token_emb = nn.Embedding(vocab_size, emb_size)
    
  def forward(self, x, y):
    mask = torch.triu(torch.ones(x.shape[0], x.shape[1], x.shape[1]))
    cross_attention_mask = torch.ones(x.shape[0], y['input'].shape[1], x.shape[1]) # N, Tk, Tq
    cross_attention_mask[y['mask'][:,:, 0]==0] = 0

    temp = torch.ones_like(x)
    result = torch.arange(x.shape[-1]).to(x.device) * temp
    x = self.token_emb(x) + self.pos_enc(result)
    return self.layers({'input':x, 'decoder_mask':mask, 'encoder_out':y['input'], 'encoder_mask':cross_attention_mask})

class TransformerTranslator(nn.Module):
  def __init__(self, in_size, emb_size, mlp_size, num_head, num_enc_layers, num_dec_layers, enc_vocab_size, dec_vocab_size):
    super().__init__()
    self.encoder = Encoder(in_size, emb_size, mlp_size, num_head, num_enc_layers, enc_vocab_size)
    self.decoder = Decoder(in_size, emb_size, mlp_size, num_head, num_dec_layers, dec_vocab_size)
    self.final_proj = nn.Linear(emb_size, dec_vocab_size)
  
  def forward(self, x:torch.Tensor, y:torch.Tensor):
    '''
    Arguments:
    '''
    enc_out = self.encoder(x)
    dec_out = self.decoder(y, enc_out)
    return self.final_proj(dec_out['input']).softmax(dim=-1)


def main():
  torch.manual_seed(0)
  num_t = 23
  h_size = 16

  keys = torch.randn(num_t, h_size)
  query = torch.randn(h_size)

  att_score = get_attention_score_for_a_single_query(keys, query)
  assert att_score.ndim == 1 and len(att_score) == num_t, "Error: Check output shape"
  answer = torch.Tensor([-3.0786,  2.1729,  1.7950, -5.0503,  3.3254,  0.2828, -0.9800, -1.8868,
          0.2550,  2.9389, -0.1799, -1.0586,  0.1465, -0.9441,  0.8888, -3.8108,
          -2.5662, -1.1660, -2.2327,  2.7087, -0.5800,  8.7984,  4.3816])
  assert torch.allclose(att_score, answer, atol=1e-4) , "Error: The output value is different"
  att_weight = get_attention_weight_from_score(att_score)
  answer = torch.Tensor([0.0000,     0.0013,     0.0009,     0.0000,     0.0041,     0.0002,
            0.0001,     0.0000,     0.0002,     0.0028,     0.0001,     0.0001,
            0.0002,     0.0001,     0.0004,     0.0000,     0.0000,     0.0000,
            0.0000,     0.0022,     0.0001,     0.9756,     0.0118])
  assert att_weight.shape == att_score.shape, 'Shape has to be remained the same'
  assert att_weight.sum() == 1, "Sum of attention weight has to be 1"
  assert torch.allclose(att_weight, answer, atol=1e-4) , "Error: The output value is different"
  att_vec = get_weighted_sum(keys, att_weight) # In simple dot-product-attention, key and value are the same
  answer = torch.Tensor([ 0.6280,  3.8540, -0.1042,  0.3148,  0.3711, -0.5095, -0.9663,  1.3295,
         1.9003, -1.2611, -2.2939, -2.0338,  0.8757, -0.6726,  1.9071, -1.0711])
  assert att_vec.shape == query.shape, 'Shape has to be remained the same'
  assert torch.allclose(att_vec, answer, atol=1e-4) , "Error: The output value is different"

  print("Problem 1 Passed!")

  torch.manual_seed(0)
  num_b = 6
  num_t = 23
  h_size = 16

  keys = torch.randn(num_b,num_t, h_size)
  query = torch.randn(num_b, h_size)
  out = get_attention_score_for_a_batch_query(keys, query)
  answer = torch.tensor([ -2.7744,   1.3793,   5.0969,   0.7559,   2.5898,  -0.9475,  -1.1960,
            5.4975,   0.4018,   5.9949,  -5.9428,  -0.4441,   0.6729,  -0.8326,
            3.7091,   1.4913,   2.2062,  -0.2244,  -4.0612,   2.9037,  10.6111,
            4.1383,  -4.6549])

  assert out.ndim == 2 and out.shape == torch.Size([num_b, num_t]), "Error: Check output shape"
  assert torch.allclose(out[2], answer, atol=1e-4), "Error: The output value is different"

  torch.manual_seed(0)
  num_b = 6
  num_ts = 23
  num_tt = 14
  h_size = 16

  keys = torch.randn(num_b, num_ts, h_size)
  queries = torch.randn(num_b, num_tt, h_size)
  att_score = get_attention_score_for_a_batch_multiple_query(keys, queries)

  answer = torch.Tensor([ 4.9620, -9.6091, -4.9472,  1.4543, -5.6273,  9.1436,  1.4172,  0.0464,
          -5.7033,  4.5473,  7.7498,  1.3405, -3.1877,  2.8759])
  answer2 = torch.Tensor([[ 2.5171,  0.6216,  3.7929,  2.6163,  5.3290,  0.3592,  2.3067, -0.1099,
          1.8963,  0.4175, -1.4283,  1.4388, -2.7825, -1.3690, -1.9615, -1.9514,
          -6.4635,  1.9574,  0.1868,  8.5354,  4.6053,  2.8786, -2.1453]])
  assert att_score.ndim == 3 and att_score.shape == torch.Size([num_b, num_ts, num_tt]), 'Check the output shape'
  assert torch.allclose(att_score[2,4], answer, atol=1e-4), 'Calculated result is wrong'
  assert torch.allclose(att_score[3,:,2], answer2, atol=1e-4),  'Calculated result is wrong'
  mask = torch.ones_like(att_score)[..., 0]
  mask[4, 15:] = 0
  mask[5, 17:] = 0

  attention_weight = get_masked_softmax(att_score, mask)
  attention_weight

  answer = torch.Tensor([0.0120,     0.0002,     0.0901,     0.0003,     0.0259,     0.0036,
            0.5617,     0.0108,     0.2508,     0.0054,     0.0001,     0.0010,
            0.0000,     0.0005,     0.0375,     0.0000,     0.0000,     0.0000,
              0.0000,     0.0000,     0.0000,     0.0000,     0.0000])
  assert torch.allclose(attention_weight[4,:,3], answer, atol=1e-4), 'Calculated result is wrong'
  assert torch.allclose(attention_weight.sum(1),  torch.tensor([1.0]) , atol=1e-6 ), 'Sum of attention weight has to be 1'

  att_out = get_batch_weighted_sum(keys, attention_weight)

  answer = torch.Tensor([-0.9348, -1.2628, -0.9189, -0.3434, -1.6476,  0.1031, -0.6963, -0.7462,
         0.1484,  0.6810,  0.7950,  1.0277, -1.5988,  0.4232, -1.5540,  0.1801])
  answer2 = torch.Tensor([-0.9204, -0.9710,  0.3062, -1.0122,  1.1933,  0.1302, -1.0280,  0.0095,
          0.6124,  0.0615, -1.2312, -0.6714, -0.1764, -0.1254])
  assert att_out.ndim == 3 and att_out.shape == torch.Size([num_b, num_tt, h_size]), 'Check the output shape'
  assert torch.max(torch.abs(att_out[2, 5] - answer)) < 1e-4, 'Calculated result is wrong'
  assert torch.max(torch.abs(att_out[3,:,2] - answer2)) < 1e-4,  'Calculated result is wrong'

  print("Problem 2 Passed!")


  src_tokenizer = BertTokenizerFast.from_pretrained('hugging_kor_32000',
                                                        strip_accents=False,
                                                        lowercase=False) 
  tgt_tokenizer = BertTokenizerFast.from_pretrained('hugging_eng_32000',
                                                        strip_accents=False,
                                                        lowercase=False) 



  model = TranslatorAtt(src_tokenizer, tgt_tokenizer, hidden_size=32, num_layers=2)
  # model(batch[0], batch[1])

  model = TranslatorAtt(src_tokenizer, tgt_tokenizer, 512)
  state_dict = torch.load('kor_eng_translator_attention_model_best.pt', map_location='cpu')['model']
  model.eval()
  model.load_state_dict(state_dict)

  # Load the pre-calculated example and result
  prob3_values = torch.load('assignment_4_values.pt')
  single_batch_example, packed_batch_example, correct_single_out, correct_packed_out = prob3_values['single_test_batch'], prob3_values['packed_test_batch'], prob3_values['correct_single_out'],  prob3_values['correct_packed_out'] 
  single_out = model(single_batch_example[0], single_batch_example[1])

  assert isinstance(single_out, torch.Tensor), "The output of model for Tensor has to be Tensor"
  assert torch.allclose(single_out, correct_single_out, atol=1e-4), "The output value is different from the expected"
  packed_out = model(packed_batch_example[0], packed_batch_example[1])

  assert isinstance(packed_out, PackedSequence), "The output of model for PackedSequence has to be PackedSequence"
  assert (packed_out.batch_sizes == correct_packed_out.batch_sizes).all(), "Output's batch_sizes is wrong"
  assert (packed_out.sorted_indices == correct_packed_out.sorted_indices).all(), "Output's sorted_indices is wrong"

  assert torch.allclose(packed_out.data, correct_packed_out.data, atol=1e-4),  "The output value is different from the expected"

  test_sentence = '이 알고리즘을 사용하면 한국어 단어와 영어 단어가 어떻게 연결되는지를 알 수 있습니다.'
  input_tokens, pred_tokens, translated_string, att_weights  = translate(model, test_sentence)

  correct_output = 'using this algorithm, you can see how korean words and english words are connected.'
  answer = torch.tensor([1.6955e-07, 1.1118e-07, 6.0198e-10, 9.1661e-01, 7.8413e-18, 1.2012e-17,
          7.6764e-29, 1.2549e-20, 5.7431e-19, 3.3563e-31, 1.1225e-21, 1.3192e-27,
          3.6772e-26, 2.2244e-23, 3.6098e-20, 1.5309e-21, 7.5093e-05])
  answer2 = torch.tensor([1.1643e-11, 3.9906e-30, 1.2813e-33, 2.7519e-13, 2.3483e-15, 4.2758e-12,
          1.9385e-18, 6.8541e-16, 4.9662e-18, 5.0304e-33, 7.2299e-26, 4.4580e-25,
          3.7096e-23, 7.5614e-22, 4.5226e-22, 2.3576e-25, 1.7577e-12])
  answer3 = torch.tensor([1.2012e-17, 1.7528e-21, 1.1316e-18, 2.7204e-17, 8.2384e-10, 3.4510e-11,
          1.8289e-09, 1.1806e-12, 3.9218e-19, 2.8321e-16, 1.3933e-12, 3.6876e-10,
          4.1782e-07, 1.0905e-02, 9.8673e-01, 2.3651e-03, 1.7207e-08, 4.2758e-12])


  assert translated_string == correct_output, 'Translated sentence is wrong'
  assert att_weights.shape == torch.Size([18, 17]), 'Attention weight has wrong shape'
  assert torch.allclose(att_weights[0], answer, rtol=1e-4), 'Calculated result is wrong'
  assert torch.allclose(att_weights[-1], answer2, rtol=1e-4), 'Calculated result is wrong'
  assert torch.allclose(att_weights[:,5], answer3, rtol=1e-4), 'Calculated result is wrong'

  print("Problem 3 Passed!")


  torch.manual_seed(0)
  test = torch.randn(4, 17, 8)
  linear = nn.Linear(8, 16 * 3)
  queries, keys, values = get_query_key_value(test, linear)
  answer = torch.Tensor([ 0.5393,  0.0587,  0.6597, -1.1150, -0.7343,  0.3282,  0.0551,  0.0178,
          0.4408, -0.3078,  0.3289, -0.4874,  0.2256, -0.1007, -0.4304, -0.2109])
  answer2 = torch.Tensor([ 0.8704, -0.2256,  0.6611,  0.0332, -0.5233, -0.1159,  0.1805,  0.7238,
          0.5590,  0.7260,  1.3096,  0.2465,  1.1961,  0.1751, -0.9674,  0.6297])
  assert keys.ndim == queries.ndim == values.ndim == 3
  assert keys.shape == queries.shape == values.shape == torch.Size([4, 17, 16])
  assert not (keys==queries).any() and not (keys==values).any() and not (values==queries).any()
  assert torch.allclose(queries[2, 13], answer, atol=1e-4)
  assert torch.allclose(values[0, 3], answer2, atol=1e-4)

  
  torch.manual_seed(0)
  mask = torch.ones([3, 9, 9])
  mask[1, 2:] = 0
  mask[2, 7:] = 0
  att_score = torch.randn([3, 9, 9])
  att_score_modified = att_score.clone()
  att_score_modified[1, 2:] = 0 
  attention_weight = get_3d_masked_softmax(att_score, mask)
  attention_weight_for_modified = get_3d_masked_softmax(att_score_modified, mask)

  answer = torch.tensor([0.1348, 0.1429, 0.2938, 0.0369, 0.0748, 0.0577, 0.2591, 0.0000, 0.0000])

  assert attention_weight.ndim == 3
  # assert torch.allclose(attention_weight[2,:, 0], answer, atol=1e-4)
  assert not torch.isnan(attention_weight).any(), "Error: There is a nan value in attention_weight"
  assert torch.allclose(attention_weight, attention_weight_for_modified, atol=1e-4), "Error: The attention_weight are different even though only masked item is different"

  torch.manual_seed(0)
  test = torch.randn(5, 17, 8)
  linear = nn.Linear(8, 16 * 3)
  mask = torch.ones([5, 17, 17])
  mask[2, 4:] = 0
  mask[4, 14:] = 0

  att_vecs = get_self_attention(test, linear, mask)
  modified_test = test.clone()
  modified_test[2, 4:] = 0
  modified_test[4, 14:] = 0
  modified_att_vecs = get_self_attention(modified_test, linear, mask)

  answer = torch.Tensor([-0.3925, -0.0043,  0.0343, -0.6713,  0.2388, -0.4703, -0.2195, -0.1550,
          -0.0830, -0.4170, -0.1829,  0.3884,  0.2899,  0.1284,  0.0225, -0.5960])
  answer2 = torch.Tensor([-0.4078,  0.0173,  0.2670, -0.7959, -0.0314, -0.3455,  0.5751, -0.5806,
          -0.3328, -0.2571, -0.4913, -0.1833,  0.6236, -0.5167,  0.3256, -0.9818])
  assert torch.allclose(att_vecs[3, 2], answer, atol=1e-4)
  assert torch.allclose(att_vecs[0, 11], answer2, atol=1e-4)
  assert torch.allclose(att_vecs[4, :14], modified_att_vecs[4, :14], atol=1e-6)

  torch.manual_seed(0)
  dummy_input = torch.randn(4, 17, 32)
  head_split_output = get_multihead_split(dummy_input, 8)
  head_split_output.shape

  assert head_split_output.shape == torch.Size([32, 17, 4])
  assert (dummy_input[0, :, 4:8] == head_split_output[1]).all()
  assert (dummy_input[0, 3, 8:12] == head_split_output[2, 3, :]).all()
  assert (dummy_input[2, 10, 16:20] == head_split_output[20, 10, :]).all()
  print('Passed all the cases!')
  head_cat_output = get_multihead_concat(head_split_output, 8)
  print(f"Output shape: {head_cat_output.shape}")
  assert head_cat_output.shape == torch.Size([4, 17, 32])
  assert (dummy_input == head_cat_output).all()


  torch.manual_seed(0)
  test = torch.randn(5, 17, 16)
  linear = nn.Linear(16, 16 * 3)
  out_proj = nn.Linear(16, 16)

  mask = torch.ones([5, 17, 17])
  mask[2, 4:] = 0
  mask[4, 14:] = 0

  att_vecs = get_multi_head_self_attention(test, linear, out_proj, mask, num_head=4)
  official_attention = torch.nn.MultiheadAttention(16, num_heads=4, batch_first=True)
  official_attention.in_proj_weight.data = linear.weight.data
  official_attention.in_proj_bias.data = linear.bias.data
  official_attention.out_proj.weight.data = out_proj.weight.data
  official_attention.out_proj.bias.data = out_proj.bias.data

  head_repeated_mask = mask.unsqueeze(1).repeat(1, 4, 1, 1).reshape(-1, mask.shape[1], mask.shape[2]).transpose(1,2)
  official_attention_output, attention_weights = official_attention(test, test, test, attn_mask=head_repeated_mask==0)

  assert torch.allclose(att_vecs, official_attention_output, atol=1e-4), "Your output is different from the official output"
  torch.manual_seed(0)
  attention_module = SelfAttention(512, 512, 8)
  test = torch.randn(5, 17, 512)
  mask = torch.ones([5, 17, 17])
  mask[2, 4:] = 0
  mask[4, 14:] = 0

  out = attention_module(test, mask)
  official_attention = torch.nn.MultiheadAttention(512, num_heads=8, batch_first=True)
  official_attention.in_proj_weight.data = attention_module.qkv.weight.data
  official_attention.in_proj_bias.data = attention_module.qkv.bias.data
  official_attention.out_proj.weight.data = attention_module.out_proj.weight.data
  official_attention.out_proj.bias.data = attention_module.out_proj.bias.data

  head_repeated_mask = mask.unsqueeze(1).repeat(1, 8, 1, 1).reshape(-1, mask.shape[1], mask.shape[2]).transpose(1,2)
  official_attention_output, attention_weights = official_attention(test, test, test, attn_mask=head_repeated_mask==0)

  assert torch.allclose(out, official_attention_output, atol=1e-4), "Your output is different from the official output"
  torch.manual_seed(0)
  encoder_layer = EncoderLayer(512, 512, 2048, 8)
  test = torch.randn(5, 17, 512)
  mask = torch.ones([5, 17, 17])
  mask[2, 4:] = 0
  mask[4, 14:] = 0

  out = encoder_layer({'input':test, 'mask':mask})

  official_encoder_layer = nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True, dropout=0)
  official_encoder_layer.self_attn.in_proj_weight.data = encoder_layer.att_block.submodule.qkv.weight.data
  official_encoder_layer.self_attn.in_proj_bias.data = encoder_layer.att_block.submodule.qkv.bias.data
  official_encoder_layer.self_attn.out_proj.weight.data = encoder_layer.att_block.submodule.out_proj.weight.data
  official_encoder_layer.self_attn.out_proj.bias.data = encoder_layer.att_block.submodule.out_proj.bias.data
  official_encoder_layer.linear1.weight.data = encoder_layer.mlp_block.submodule.layer[0].weight.data
  official_encoder_layer.linear1.bias.data = encoder_layer.mlp_block.submodule.layer[0].bias.data
  official_encoder_layer.linear2.weight.data = encoder_layer.mlp_block.submodule.layer[2].weight.data
  official_encoder_layer.linear2.bias.data = encoder_layer.mlp_block.submodule.layer[2].bias.data
  official_encoder_layer.norm1.weight.data = encoder_layer.att_block.layer_norm.weight.data
  official_encoder_layer.norm1.bias.data = encoder_layer.att_block.layer_norm.bias.data
  official_encoder_layer.norm2.weight.data = encoder_layer.mlp_block.layer_norm.weight.data
  official_encoder_layer.norm2.bias.data = encoder_layer.mlp_block.layer_norm.bias.data

  head_repeated_mask = mask.unsqueeze(1).repeat(1, 8, 1, 1).reshape(-1, mask.shape[1], mask.shape[2]).transpose(1,2)
  official_encoder_output = official_encoder_layer(test, src_mask=head_repeated_mask==0)

  assert torch.allclose(official_encoder_output, out['input'], atol=1e-4), "Your output is different from the official output"


  torch.manual_seed(0)
  decoder_layer = DecoderLayer(512, 512, 2048, 8)
  test_src = torch.randn(5, 17, 512)
  test_tgt = torch.randn(5, 19, 512)
  mask_src = torch.ones([5, 17, 19])
  mask_src[2, 4:] = 0
  mask_src[4, 14:] = 0
  mask_tgt = torch.tril(torch.ones(test_tgt.shape[0], test_tgt.shape[1], test_tgt.shape[1]))

  out = decoder_layer({'input':test_tgt, 'decoder_mask':mask_tgt, 'encoder_out':test_src, 'encoder_mask':mask_src})
  official_decoder_layer = nn.TransformerDecoderLayer(512, 8, 2048, batch_first=True, dropout=0)
  official_decoder_layer.self_attn.in_proj_weight.data = decoder_layer.att_block.submodule.qkv.weight.data
  official_decoder_layer.self_attn.in_proj_bias.data = decoder_layer.att_block.submodule.qkv.bias.data
  official_decoder_layer.self_attn.out_proj.weight.data = decoder_layer.att_block.submodule.out_proj.weight.data
  official_decoder_layer.self_attn.out_proj.bias.data = decoder_layer.att_block.submodule.out_proj.bias.data
  official_decoder_layer.multihead_attn.in_proj_weight.data = decoder_layer.cross_att_block.submodule.qkv.weight.data
  official_decoder_layer.multihead_attn.in_proj_bias.data = decoder_layer.cross_att_block.submodule.qkv.bias.data
  official_decoder_layer.multihead_attn.out_proj.weight.data = decoder_layer.cross_att_block.submodule.out_proj.weight.data
  official_decoder_layer.multihead_attn.out_proj.bias.data = decoder_layer.cross_att_block.submodule.out_proj.bias.data
  official_decoder_layer.linear1.weight.data = decoder_layer.mlp_block.submodule.layer[0].weight.data
  official_decoder_layer.linear1.bias.data = decoder_layer.mlp_block.submodule.layer[0].bias.data
  official_decoder_layer.linear2.weight.data = decoder_layer.mlp_block.submodule.layer[2].weight.data
  official_decoder_layer.linear2.bias.data = decoder_layer.mlp_block.submodule.layer[2].bias.data
  official_decoder_layer.norm1.weight.data = decoder_layer.att_block.layer_norm.weight.data
  official_decoder_layer.norm1.bias.data = decoder_layer.att_block.layer_norm.bias.data
  official_decoder_layer.norm2.weight.data = decoder_layer.cross_att_block.layer_norm.weight.data
  official_decoder_layer.norm2.bias.data = decoder_layer.cross_att_block.layer_norm.bias.data
  official_decoder_layer.norm3.weight.data = decoder_layer.mlp_block.layer_norm.weight.data
  official_decoder_layer.norm3.bias.data = decoder_layer.mlp_block.layer_norm.bias.data

  head_repeated_mask_src = mask_src.unsqueeze(1).repeat(1, 8, 1, 1).reshape(-1, mask_src.shape[1], mask_src.shape[2]).transpose(1,2)
  head_repeated_mask_tgt = mask_tgt.unsqueeze(1).repeat(1,8,1,1).reshape(-1, mask_tgt.shape[1], mask_tgt.shape[2]).transpose(1,2)
  official_decoder_output = official_decoder_layer(test_tgt, test_src, tgt_mask=head_repeated_mask_tgt==0, memory_mask=head_repeated_mask_src==0)

  assert torch.allclose(official_decoder_output, out['input'], atol=1e-4), "Your output is different from the official output"



  print("Problem 4 Passed!")



if __name__ == '__main__':
  main()