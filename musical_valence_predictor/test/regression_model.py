import torch

def test(model, data_loader):
  print('Testing model...')

  # Loss function: Mean absolute error (MAE).
  criterion = torch.nn.L1Loss(reduction = 'mean')

  with torch.no_grad():
    for mini_batch in data_loader:
      inputs, labels = mini_batch

      outputs = model(inputs)
      loss = criterion(outputs, labels)

      print('Test run complete. Loss (MAE): %.3f' % loss.item())
