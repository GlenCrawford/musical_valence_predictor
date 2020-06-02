import torch

class RegressionModel(torch.nn.Module):
  def __init__(self):
    super(RegressionModel, self).__init__()

    # Define the transformations of the model (similar concept to layers).
    # Linear transformation arguments: Number of inputs, number of outputs.
    self.linear1 = torch.nn.Linear(53, 16)
    self.leaky_relu = torch.nn.LeakyReLU()
    self.linear2 = torch.nn.Linear(16, 1)

  def forward(self, data):
    data = data.float()

    data = torch.nn.functional.relu(self.linear1(data))
    data = self.leaky_relu(data)
    data = self.linear2(data)

    return data
