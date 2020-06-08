import torch

class RegressionModel(torch.nn.Module):
  def __init__(self):
    super(RegressionModel, self).__init__()

    # Define the transformations of the model (similar concept to layers).
    # Linear transformation arguments: Number of inputs, number of outputs.
    self.linear1 = torch.nn.Linear(53, 300)
    self.relu = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(300, 1)

  def forward(self, data):
    data = data.float()

    data = self.linear1(data)
    data = self.relu(data)
    data = self.linear2(data)

    return data
