import torch

def test(model, data_loader):
  print('Testing model...')

  correct_predictions = 0
  total_predictions = 0

  with torch.no_grad():
    for mini_batch in data_loader:
      # Reminder: The labels are the index of the class that the valence value was mapped to.
      inputs, labels = mini_batch

      # Outputs is a tensor where each sample in the batch has an array of N decimal values, where N is the number of classes/neurons in the output layer.
      # The outputs can be thought of as probabilities for each class. The higher the probability for a class, the more the model believes the sample is of that class.
      outputs = model(inputs)

      # We don't apply Softmax within the model, as the CrossEntropyLoss loss function applies it. So it outputs raw logits and we apply Softmax as needed outside of the model.
      outputs = torch.nn.Softmax()(outputs)

      # Returns the value and index of the highest class value's probability for each sample in the tensor.
      predicted_class_value, predicted_class_index = torch.max(outputs, 1)

      total_predictions += labels.size(0)

      correct_predictions += (predicted_class_index == labels).sum().item()

  print('Test run accuracy: %d %%' % (100 * correct_predictions / total_predictions))
