import torch

class ClassificationModel(torch.nn.Module):
  def __init__(self):
    super(ClassificationModel, self).__init__()

    # Even though this is a multi-class classification problem, do not apply Softmax within the model.
    # The CrossEntropyLoss loss function applies Softmax internally, and thus should be given the raw logits in order to not apply it twice.
    # An alternative would be to use a LogSoftmax layer within the model, and feed the results into NLLLoss outside.
    self.linear1 = torch.nn.Linear(53, 300)
    self.relu1 = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(300, 3)

  def forward(self, data):
    data = data.float()

    data = self.linear1(data)
    data = self.relu1(data)
    data = self.linear2(data)

    return data
