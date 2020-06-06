import torch

class ClassificationModel(torch.nn.Module):
  def __init__(self):
    super(ClassificationModel, self).__init__()

    self.linear1 = torch.nn.Linear(53, 300)
    self.linear2 = torch.nn.Linear(300, 3)

  def forward(self, data):
    data = data.float()

    data = torch.nn.functional.relu(self.linear1(data))
    data = torch.nn.functional.relu(self.linear2(data)) # Softmax?

    return data
