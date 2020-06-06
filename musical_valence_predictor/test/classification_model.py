import torch

def test(model, data_loader):
  print('Testing model...')

  correct_predictions = 0
  total_predictions = 0

  with torch.no_grad():
    for mini_batch in data_loader:
      # Reminder that the labels are the index of the class that the valence value was mapped to.
      inputs, labels = mini_batch

      # Outputs is a tensor where each sample in the batch has an array of N decimal values, where N is the number of neurons in the output layer.
      # The outputs are "energies" for each class. The higher the energy for a class, the more the network believes the sample is of that batch.
      outputs = model(inputs)

      # get the index of the highest energy:
      # Returns the value and index of the highest class value ("energy") for each sample in the tensor.
      predicted_class_value, predicted_class_index = torch.max(outputs, 1)

      print('Labels:')
      print(str(labels))
      print('\n\n==========\n\n')
      print('Outputs:')
      print(str(outputs))

      total_predictions += labels.size(0)

      correct_predictions += (predicted_class_index == labels).sum().item()

  print('Test run accuracy: %d %%' % (100 * correct_predictions / total_predictions))
