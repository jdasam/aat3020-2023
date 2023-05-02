import torch
from torchtext.data.utils import get_tokenizer
from pathlib import Path
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_sequence
import torch
from tqdm.auto import tqdm

from assignment2_pre_defined import get_train_txt_paths_in_split, SentimentModel, read_txt, make_vocab_from_txt_fns

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


class PackCollateWithConverter:
  def __init__(self, converter):
    self.converter = converter
  
  def __call__(self, batch):
    '''
    TODO: Declare variables txts_in_idxs and label_tensor, following the description below
    Use self.converter to convert txts to txts_in_idxs

    word_sentences_in_idxs: A list of torch.LongTensor. Each element in a list is a sequence of integer, and the each integer represents a vocabulary index of word in a sentence.
                  i-th element of word_sentences_in_idxs corresponds to the i-th data sample in the batch  
    labels: torch.FloatTensor with a shape of [len(batch)]. i-th value of the tensor represents the label of the i-th data sample in the batch (either 0.0 or 1.0)
    '''
    
    # Write your code from here
    word_sentences_in_idxs = []
    label_tensor = torch.FloatTensor([0])

    
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
        self.save_model('imdb_sentiment_model_best_py_test.pt')
      else:
        self.save_model('imdb_sentiment_model_last_py_test.pt')
      self.best_valid_accuracy = max(validation_acc, self.best_valid_accuracy)

  def _get_accuracy(self, pred, target, threshold=0.5):
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
  train_pths, valid_pths = get_train_txt_paths_in_split(train_path)
  test_pths = list(test_path.rglob("*.txt"))

  print(f"Number of training data: {len(train_pths)}, validation data: {len(valid_pths)}, test data: {len(test_pths)}")

  tokenizer = get_tokenizer('basic_english')
  entire_vocab = make_vocab_from_txt_fns(train_pths, tokenizer)
  min_count = 5
  vocab = sorted([token for token, count in entire_vocab.items() if count >= min_count])

  print(f"Number of tokens in the entire vocabulary: {len(entire_vocab)}")
  print(f"Number of tokens in the vocabulary with min_count = {min_count}: {len(vocab)}")
  print(f"First 10 tokens in the vocabulary: {vocab[:10]}")
  print(f"Last 10 tokens in the vocabulary: {vocab[-10:]}")
  print(f"Middle 10 tokens in the vocabulary: {vocab[len(vocab)//2-5:len(vocab)//2+5]}")

  trainset = IMDbData(train_pths)
  validset = IMDbData(valid_pths)
  short_validset = IMDbData(valid_pths[:100])
  testset = IMDbData(test_pths)

  assert len(trainset) == 20000 and len(validset) ==5000 and len(short_validset)==100
  assert len(trainset[0]) == 2
  assert trainset[154][0][10:15] == ['ends', 'right', 'after', 'this', 'little'], "Error in the trainset __getitem__ output"
  assert trainset[594][1] == 0 and trainset[523][1] == 1 and trainset[1523][1] == 0, "Error in the trainset __getitem__ output"

  print("Passed all the test cases!")

  
  converter = Str2Idx2Str(vocab)
  input_sentence = trainset[0][0][:20] #0th sample, text (instead of label), first 20 words
  print(f"Input sentence: {input_sentence}")
  print(f"Converted sentence: {converter(input_sentence)}")
  print(f"Re-converted sentence: {converter(converter(input_sentence))}")
  print(f"Result for a list of sentences/ input_list: {[trainset[i][0][:5]for i in range(1,5)]}, output_list: {converter([trainset[i][0][:5]for i in range(1,5)])}")

  pack_collate = PackCollateWithConverter(converter)
  train_loader = DataLoader(trainset, batch_size=32, collate_fn=pack_collate, shuffle=True)
  batch = next(iter(train_loader))
  print('A batch looks like this: ', batch)

  train_loader = DataLoader(trainset, batch_size=2, collate_fn=pack_collate, shuffle=True)
  model = SentimentModel(len(converter.idx2str), hidden_size=32, num_layers=1)
  batch = next(iter(train_loader))
  print(f"Model output: {model(batch[0])}")

  test_pred_case = torch.Tensor([9.9894e-01, 2.2645e-03, 1.8131e-01, 8.0153e-03, 9.9972e-01, 1.0378e-03,
        9.9949e-01, 9.9967e-01, 6.4150e-03, 9.9912e-01, 9.9896e-01, 1.4350e-01,
        9.9896e-01, 2.1979e-02, 9.9976e-01, 4.5389e-03, 9.9906e-01, 1.0633e-02,
        9.9749e-01, 5.5501e-04, 7.0052e-04, 2.9509e-04, 3.2752e-04, 9.9940e-01,
        4.5912e-04, 9.9969e-01, 6.0225e-03, 9.9974e-01, 9.9907e-01, 9.9942e-01,
        4.0911e-01, 2.8850e-01])
  test_target_case = torch.Tensor([1., 0., 0., 0., 1., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 0.,
          1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0.])

  your_result = get_binary_cross_entropy_loss(test_pred_case, test_target_case)
  print(f"Your BCE result: {your_result}")
  
  trainset.paths = trainset.paths[:100]
  testset.paths = testset.paths[:100]
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  train_loader = DataLoader(trainset, batch_size=16, collate_fn=pack_collate, shuffle=True, drop_last=True)
  test_loader = DataLoader(testset, batch_size=16, collate_fn=pack_collate)
  short_valid_loader = DataLoader(short_validset, batch_size=50, collate_fn=pack_collate)

  trainer =  Trainer(model, optimizer, get_binary_cross_entropy_loss, train_loader, short_valid_loader, device='cuda')

  trainer.model.train()
  train_batch = next(iter(trainer.train_loader)) # get a batch from train_loader

  loss_track = []
  for _ in range(10):
    loss_value, acc = trainer._train_by_single_batch(train_batch) # test the trainer
    loss_track.append(loss_value)

  assert isinstance(loss_value, float) and loss_value > 0,  "The return of trainer._train_by_single_batch has to be a single float value that is larger than 0"
  print(f"Loss value for 10 repetition for the same training batch is  {[f'{loss:.4f}' for loss in loss_track]}")


  validation_loss, validation_acc = trainer.validate(short_valid_loader)
  assert isinstance(validation_loss, float) and isinstance(validation_acc, float), "Both return value of trainer.validate has to be float"
  assert validation_loss > 0, "Validation Loss has to be larger than 1"
  assert 0 <= validation_acc <= 1, "Validation Acc has to be between 0 and 1"

  print(f"Valid loss: {validation_loss}, Accuracy: {validation_acc}")


  trainer.train_by_num_epoch(1)
  print(f"Last 5 Training loss: {trainer.training_loss[-5:]}")


if __name__ == "__main__":
  main()