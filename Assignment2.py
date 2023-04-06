import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt
import warnings
import gensim.downloader
from pathlib import Path
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_sequence
import torch
from tqdm.auto import tqdm

from assignment2_pre_defined import get_train_txt_paths_in_split, SentimentModel

class IMDbData:
  def __init__(self, path_list):
    self.paths = path_list
    self.tokenizer = get_tokenizer('basic_english')
  
  def __len__(self):
    """
    __len__ is a special method that returns length of the instance when called with len(class_instance)
    e.g.
      dataset = IMDbData()
      length_of_dataset = len(dataset)
      
    TODO: Complete this function 
    """
    return 

  def __getitem__(self, idx):
    """
    __getitem__ is a special method that returns an item for a given index when called with class_instance[index]
    e.g.
      trainset = IMDbData(train_pths)
      trainset[6] == trainset.__getitem__(6)
      
    output: sequence_of_token, label
      sequence_of_token (list): a list of string (word token). Use self.tokenizer to make string into a list of word token
      label (int): 0 if the sentence is negative, 1 if the sentence is positive    
      
    HINT: use str(pth) to convert Path into String.
          You can find the label of the sample in its file directory path

          
    TODO: Complete this function using self.paths, self.tokenizer, and read_txt()
    """

    return 

class Str2Idx2Str:
  def __init__(self, vocab):
    '''
    TODO: Complete the class
    
    1. Declare self.idx2str
    - self.idx2str is a list of strings, which contains a string of word that corresponds to the index of the list
    - e.g. self.idx2str[your_index] returns a string value of your_index of the vocabulary
    - Use the input argument vocab to initialize self.idx2str
    
    2. Declare self.str2idx
    - self.str2idx is a dictionary, where its keys are the words in strings and its values are the corresponding index of each word
    - e.g. self.str2idx[your_word] returns an integer value of the index of your_word in the vocabulary

    '''
    self.idx2str = []
    self.str2idx = {}
    
    
    '''
    You have to add these lines,
    And explain in your report what is the function of these two lines 
    '''
    self.unknown_idx = len(self.str2idx)
    self.idx2str.append("UNKNOWN")
  
    
  
  def __call__(self, alist):
    '''
    This function converts list of word string to its index and vice versa.
    For example, if it takes ['if', 'anyone', 'who', 'loves', 'laurel', 'and', 'hardy', 'can', 'watch', 'this', 'movie', 'and', 'feel', 'good', 'about', 'it', ',', 'you'] as an input,
    it will return [83, 1544, 38, 6741, 15722, 5, 10801, 86, 1716, 37, 1005, 5, 998, 219, 59, 20, 1, 81].
    
    If it takes [83, 1544, 38, 6741, 15722, 5, 10801, 86, 1716, 37, 1005, 5, 998, 219, 59, 20, 1, 81],
    it will return ['if', 'anyone', 'who', 'loves', 'laurel', 'and', 'hardy', 'can', 'watch', 'this', 'movie', 'and', 'feel', 'good', 'about', 'it', ',', 'you']
    
    If it takes a list of list of string, such as [['after', 'watching', 'about', 'half', 'of'], ['reading', 'all', 'of', 'the', 'comments'], ['why', 'has', 'this', 'not', 'been'], ['this', 'is', 'a', 'really', 'strange']],
    it will return a list of list of integer, [[49, 2641, 59, 343, 3], [2185, 64, 3, 0, 1939], [738, 31, 37, 36, 51], [37, 14, 7, 588, 5186]]
    
    Vice versa, if it takes [[49, 2641, 59, 343, 3], [2185, 64, 3, 0, 1939], [738, 31, 37, 36, 51], [37, 14, 7, 588, 5186]] as an input,
    it will return [['after', 'watching', 'about', 'half', 'of'], ['reading', 'all', 'of', 'the', 'comments'], ['why', 'has', 'this', 'not', 'been'], ['this', 'is', 'a', 'really', 'strange']],
    
    Input: alist of strings, or a list of integers, or a list of lists
      e.g. alist = ['if', 'anyone', 'who', 'loves', 'laurel', 'and', 'hardy', 'can', 'watch', 'this', 'movie', 'and', 'feel', 'good', 'about', 'it', ',', 'you']
        or alist = [83, 1544, 38, 6741, 15722, 5, 10801, 86, 1716, 37, 1005, 5, 998, 219, 59, 20, 1, 81]
        or alist = [['after', 'watching', 'about', 'half', 'of'], ['reading', 'all', 'of', 'the', 'comments'], ['why', 'has', 'this', 'not', 'been'], ['this', 'is', 'a', 'really', 'strange']]
        or alist = [[49, 2641, 59, 343, 3], [2185, 64, 3, 0, 1939], [738, 31, 37, 36, 51], [37, 14, 7, 588, 5186]]
    
    IMPORTANT: If a word in the input list is not in the vocabulary of Str2Idx2Str, then it has to convert it into UNKNOWN token.
    
    
    Output: a list of integer
    
    TODO: Complete this function, using self.idx2str and self.str2idx
    
    Hint: You can figure out the type of input by using the function isinstance. It will return boolean.
        isinstance(an_item, list)
        isinstance(an_item, str)
        isinstance(an_item, int)
    '''
    
    # Write your code from here
    
    
    return


def pack_collate(batch):
  '''
  TODO: Declare variables txts_in_idxs and label_tensor, following the description below
  
  word_sentences_in_idxs: A list of torch.LongTensor. Each element in a list is a sequence of integer, and the each integer represents a vocabulary index of word in a sentence.
                i-th element of word_sentences_in_idxs corresponds to the i-th data sample in the batch  
  labels: torch.FloatTensor with a shape of [len(batch)]. i-th value of the tensor represents the label of the i-th data sample in the batch (either 0.0 or 1.0)
  '''
  
  # Write your code from here
  word_sentences_in_idxs = []
  label_tensor = torch.Tensor([])
  
  '''
  Leave the code below as it is
  '''
  assert isinstance(word_sentences_in_idxs, list), f"txts_in_idxs has to be a list, not {type(word_sentences_in_idxs)}"
  assert isinstance(word_sentences_in_idxs[0], torch.LongTensor), f"An elmenet of txts_in_idxs has to be a torch.LongTensor, not {type(word_sentences_in_idxs[0])}"
  assert isinstance(label_tensor, torch.FloatTensor), f"labels has to be a torch.FloatTensor, not {type(label_tensor)}"
  assert label_tensor[-1] == batch[-1][1], "i-th element of labels has to be "
  
  packed_sequence = pack_sequence(word_sentences_in_idxs, enforce_sorted=False)

  return packed_sequence, label_tensor


def get_binary_cross_entropy_loss(pred, target, eps=1e-8):
  '''
  pred (torch.FloatTensor): Prediction value for N samples 
                            Each element in the tensor is the output of torch.sigmoid, and has a value between 0 and 1
  target (torch.FloatTensor): Corresponding target value for N samples. 
                              Each element in the tensor has value of either 0 or 1
  eps (float): A small value to avoid log(0) error
  
  output: Mean of binary cross entropy of N samples
  
  TODO: Complete this function
  
  '''
  return


class Trainer:
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = torch.nn.BCELoss()
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    
    self.model.to(device)
    
    self.best_valid_accuracy = 0
    self.device = device
    
    self.training_loss = []
    self.training_acc = []
    self.validation_loss = []
    self.validation_acc = []

  def save_model(self, path='imdb_sentiment_model.pt'):
    torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, path)
    
  def train_by_num_epoch(self, num_epochs):
    for epoch in range(num_epochs):
      self.model.train()
      for batch in tqdm(self.train_loader, leave=False):
        loss_value, acc = self._train_by_single_batch(batch)
        self.training_loss.append(loss_value)
        self.training_acc.append(acc)

      self.model.eval()
      validation_loss, validation_acc = self.validate()
      self.validation_loss.append(validation_loss)
      self.validation_acc.append(validation_acc)
      
      print(f"Epoch {epoch+1}, Training Loss: {loss_value:.4f}, Training Acc: {acc:.4f}, Validation Loss: {validation_loss:.4f}, Validation Acc: {validation_acc:.4f}")
      if validation_acc > self.best_valid_accuracy:
        print(f"Saving the model with best validation accuracy: Epoch {epoch+1}, Acc: {validation_acc:.4f} ")
        self.save_model('imdb_sentiment_model_best.pt')
      else:
        self.save_model('imdb_sentiment_model_last.pt')
      self.best_valid_accuracy = max(validation_acc, self.best_valid_accuracy)

  def _get_accuracy(self, pred, target):
    '''
    This method calculates accuracy for given prediction and target
    
    input:
      pred (torch.Tensor): Prediction value for a given batch
      target (torch.Tensor): Target value for a given batch
      
    output: 
      accuracy (float): Mean Accuracy value for every sample in a given batch
    

    TODO: Complete this method
    '''

    return

  def _get_loss_and_acc_from_single_batch(self, batch):
    '''
    This method calculates loss value for a given batch

    batch (tuple): (batch_of_input_text, batch_of_label)

    You have to use variables below:
    self.model (SentimentModel/torch.nn.Module): A neural network model
    self.loss_fn (function): function for calculating BCE loss for a given prediction and target
    self.device (str): 'cuda' or 'cpu'
    self._get_accuracy (function): function for calculating accuracy for a given prediction and target

    output: 
      loss (torch.Tensor): Mean binary cross entropy value for every sample in the training batch
      acc (float): Accuracy for the given batch
    # CAUTION! The output loss has to be torch.Tensor that is backwardable, not a float value or numpy array

    TODO: Complete this method
    '''

    return
      
  def _train_by_single_batch(self, batch):
    '''
    This method updates self.model's parameter with a given batch
    
    batch (tuple): (batch_of_input_text, batch_of_label)
    
    You have to use methods and variables below:

    self._get_loss_and_acc_from_single_batch (function): function for calculating loss value for a given batch    
    self.optimizer (torch.optim.adam.Adam): Adam optimizer that optimizes model's parameter

    output: 
      loss (float): Mean binary cross entropy value for every sample in the training batch
      acc (float): Accuracy for the given batch
    The model's parameters, optimizer's steps has to be updated inside this method

    TODO: Complete this method 
    '''
    return


    
  def validate(self, external_loader=None):
    '''
    This method calculates accuracy and loss for given data loader.
    It can be used for validation step, or to get test set result
    
    input:
      data_loader: If there is no data_loader given, use self.valid_loader as default.
    
    output: 
      validation_loss (float): Mean Binary Cross Entropy value for every sample in validation set
      validation_accuracy (float): Mean Accuracy value for every sample in validation set
      
    Use these methods:
      self._get_loss_from_single_batch (function): function for calculating loss value for a given batch
      self._get_accuracy (function): function for calculating accuracy for a given batch
    
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

    return 


def main():
  train_path = Path('aclImdb/train')
  test_path = Path('aclImdb/test')
  train_pths, valid_pths = get_train_txt_paths_in_split('aclImdb/train')
  test_pths = list(test_path.rglob("*.txt"))

  trainset = IMDbData(train_pths)
  
  validset = IMDbData(valid_pths)
  short_validset = IMDbData(valid_pths[:100])
  testset = IMDbData(test_pths)

  trainset = IMDbData(train_pths)
  assert len(trainset) == 20000 and len(validset) ==5000 and len(short_validset)==100
  assert len(trainset[0]) == 2
  assert trainset[154][0][10:15] == ['ends', 'right', 'after', 'this', 'little'], "Error in the trainset __getitem__ output"
  assert trainset[594][1] == 0 and trainset[523][1] == 1 and trainset[1523][1] == 0, "Error in the trainset __getitem__ output"

  print("Passed all the test cases!")

  wrd2vec_model = gensim.downloader.load("glove-twitter-25")   
  converter = Str2Idx2Str(wrd2vec_model, vocab_size=30000)
  input_sentence = trainset[0][0][:20]
  print(f"Input sentence: {input_sentence}")
  print(f"Converted sentence: {converter(input_sentence)}")
  print(f"Re-converted sentence: {converter(converter(input_sentence))}")
  print(f"Result for a list of sentences/ input_list: {[trainset[i][0][:5]for i in range(1,5)]}, output_list: {converter([trainset[i][0][:5]for i in range(1,5)])}")


  train_loader = DataLoader(trainset, batch_size=32, collate_fn=pack_collate, shuffle=True)


  model = SentimentModel(wrd2vec_model, hidden_size=32, num_layers=1)
  batch = next(iter(train_loader))
  model(batch[0])

  trainset.paths = trainset.paths[:100]
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  train_loader = DataLoader(trainset, batch_size=32, collate_fn=pack_collate, shuffle=True, drop_last=True)
  valid_loader = DataLoader(validset, batch_size=128, collate_fn=pack_collate, shuffle=False)
  test_loader = DataLoader(testset, batch_size=128, collate_fn=pack_collate, shuffle=False)
  short_valid_loader = DataLoader(short_validset, batch_size=128, collate_fn=pack_collate, shuffle=False)

  trainer =  Trainer(model, optimizer, get_binary_cross_entropy_loss, train_loader, short_valid_loader, device='cuda')

  trainer.train_by_num_epoch(1)
  print(f"Last 10 Training loss: {trainer.training_loss[-10:]}")
  print(f"Training loss: {trainer.training_loss[-10:]}")



if __name__ == "__main__":
  main()