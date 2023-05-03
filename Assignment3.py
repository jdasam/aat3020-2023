import torch
import torch.nn as nn




class MyLSTM(nn.Module):
  def __init__(self, input_size: int, hidden_size: int):
    super().__init__()
    '''
    TODO: Define Weights and Bias of a single-layer, uni-directional LSTM
    
    Arguments
      input_size (int): Num dimension of input x. It is represented as `d` in the equations in the explanation part
      hidden_size (int): Num dimension of output h. It is represented as `h` in the equations in the explanation part
    
    Module Definition
      self.weight_ih (nn.Linear): Linear layer that combines Weight matrix [W_i | W_f | W_c | W_o] and bias b
      self.weight_hh (nn.Linear): Linear layer that combines Weight matrix [U_i | U_f | U_c | U_o] and bias b
    
    Implementation Condition: To compare your implementation with the torch's official implementation, please strictly follow the explanation above
    '''
    
    self.weight_ih = nn.Linear(in_features=1, out_features=1, bias=True) # TODO: complete this layer by selecting proper in_features and out_features
    self.weight_hh = nn.Linear(in_features=1, out_features=1, bias=True) # TODO: complete this layer by selecting proper in_features and out_features
    self.hidden_size = hidden_size

  def _cal_single_step(self, x_t:torch.Tensor, last_hidden:torch.Tensor, last_cell:torch.Tensor):
    '''
    Argument:
      x_t : input of timestep t. Has a shape of [Num_Batch, Num_input_dim]
      last_hidden: hidden state of timestep (t-1). Has a shape of [Num_Batch, Num_hidden_dim]
      last_cell: cell state of timestep (t-1). Has a shape of [Num_batch, Num_hidden_dim]
      
    Output:
      updated_hidden (torch.Tensor): hidden state of timestep t. Has a shape of [Num_Batch, Num_hidden_dim]
      updated_cell (torch.Tensor): cell state of timestep t. Has a shape of [Num_Batch, Num_hidden_dim]
      
    TODO: Complete this function using the input arguments and self.weight_ih, self.weight_hh
    
    '''

    
    return updated_hidden, updated_cell
  
  def forward(self, x:torch.Tensor, hidden_and_cell_state:tuple=None):
    '''
    Argument:
      x (torch.Tensor): Input sequence. Has a shape of [Num_Batch, Num_Timestep, Num_input_dim
      hidden_and_cell_state (optional, tuple): Hidden state and Cell state of last timestep.
      
    Return:
      output, (last_hidden, last_cell)
      Be carefule that you have to return two variables, where the second variable is a tuple of two tensors.
      
      output (torch.Tensor): Output of LSTM that has a shape of [Batch_Size, Num_Time_Steps, Hidden_State_Size]
                             It is the concatenation of output hidden states of every given time steps
      last_hidden (torch.Tensor): The hidden state of LSTM after calculating entire time steps. 
      last_cell (torch.Tensor): LSTM has two types of hidden states, and one is called "cell state". 
      
    
    TODO: Implement this using a for loop and `self._cal_single_step`.
    '''
    
    # Leave the code below as it is
    if hidden_and_cell_state is not None and isinstance(hidden_and_cell_state, tuple):
      last_hidden = hidden_and_cell_state[0]
      last_cell = hidden_and_cell_state[1]
    else:
      last_hidden = torch.zeros([x.shape[0], self.hidden_size])
      last_cell = torch.zeros([x.shape[0], self.hidden_size])
    
    '''
    Write your code from here
    '''
    
    return output, (last_hidden, last_cell)
  

def main():
  input_size = 16
  hidden_size = 32

  model = MyLSTM(input_size, hidden_size)

  dummy_batch_size = 8
  dummy_time_steps = 20
  dummy_input = torch.randn([dummy_batch_size, dummy_time_steps, input_size])

  output, (last_hidden_state, last_cell_state) = model(dummy_input)

  assert output.shape[0] == dummy_batch_size, "0th dimension of output has to be the batch size"
  assert output.shape[1] == dummy_time_steps, "1st dimension of output has to be the time steps"
  assert output.shape[2] == hidden_size, "2nd dimension of output has to be the hidden_size"

  total_output, (last_hidden_state, last_cell_state) = model(dummy_input)

  hidden_and_cell_state = (torch.zeros([dummy_batch_size, hidden_size]), torch.zeros([dummy_batch_size, hidden_size]))
  for i in range(dummy_time_steps):
    time_step_output, hidden_and_cell_state = model(dummy_input[:,i:i+1], hidden_and_cell_state)
    
  assert (total_output[:,-1:] == time_step_output).all(), 'The LSTM output has to be equal for sliced input using for-loop'

  lstm_pre_impl = nn.LSTM(input_size, hidden_size, batch_first=True)

  lstm_pre_impl.weight_hh_l0.data = model.weight_hh.weight.data
  lstm_pre_impl.bias_hh_l0.data = model.weight_hh.bias.data

  lstm_pre_impl.weight_ih_l0.data = model.weight_ih.weight.data
  lstm_pre_impl.bias_ih_l0.data = model.weight_ih.bias.data

  output, (last_hidden_state, last_cell_state) = model(dummy_input)
  output_compare, (last_hidden_state_compare, last_cell_state_compare) = lstm_pre_impl(dummy_input)

  assert torch.allclose(output, output_compare, atol=1e-6), "The output of LSTM is different"
  assert torch.allclose(last_hidden_state, last_hidden_state_compare, atol=1e-6), "The last hidden state of LSTM is different"
  assert torch.allclose(last_cell_state, last_cell_state_compare, atol=1e-6), "The last cell state of LSTM is different"

  print("Test passed! Your LSTM implementation returns the exactly same result for PyTorch's official implementation of single-layer uni-directiona LSTM")


if __name__ == '__main__':
  main()