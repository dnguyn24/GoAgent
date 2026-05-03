import torch.nn as nn
import torch.nn.functional as F
import torch

def save_model(path: str, model):
    """
    Save model to a file
    Input:
        path: path to save model to
        model: Pytorch model to save
    """
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)

def load_model(path: str, model):
    """
    Load model from file

    Note: you still need to provide a model (with the same architecture as the saved model))

    Input:
        path: path to load model from
        model: Pytorch model to load
    Output:
        model: Pytorch model loaded from file
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
      super(ValueNetwork, self).__init__()

      # TODO: What should the output size of a Value function be?
      output_size = 1

      # TODO: Add more layers, non-linear activation functions, etc.
      self.layer1 = nn.Linear(input_size, 60)
      self.layer2 = nn.Linear(60, 40)
      self.layer3 = nn.Linear(40, 20)
      self.layer4 = nn.Linear(20, output_size) # output layer
      self.sigmoid = nn.Sigmoid()
      self.relu = nn.ReLU()

    def forward(self, x):
      """
      Run forward pass of network

      Input:
        x: input to network
      Output:
        output of network
      """
      # TODO: Update as more layers are added
      z1 = self.layer1(x)
      a1 = self.relu(z1)
      z2 = self.layer2(a1)
      a2 = self.relu(z2)
      z3 = self.layer3(a2)
      a3 = self.relu(z3)
      z4 = self.layer4(a3)
      return self.sigmoid(z4)

