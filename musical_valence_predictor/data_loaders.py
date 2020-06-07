import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch

class MusicDataSet(torch.utils.data.Dataset):
  VALENCE_CLASS_RANGES = {
    0: (0.0, 0.35), # Sad.
    1: (0.35, 0.65), # Neutral.
    2: (0.65, 1.0) # Happy.
  }

  def __init__(self, data_frame, model):
    self.data_frame = data_frame
    self.model = model

  def __len__(self):
    return len(self.data_frame)

  # Return a row from the Pandas data frame, plus the extracted label value.
  def __getitem__(self, index):
    row = self.data_frame.iloc[index]

    valence = row['Valence']
    del row['Valence']

    if self.model == 'regression':
      # Wrap in an array to mirror the dimensions of the output of the regression model.
      valence = np.array([valence])
    elif self.model == 'classification':
      valence = self.map_valence_to_class(valence)

    return (row.values, valence)

  # Use the ranges to map each numerical valence to the range that includes it.
  def map_valence_to_class(self, valence):
    for valence_class, valence_range in self.VALENCE_CLASS_RANGES.items():
      if valence_range[0] <= valence <= valence_range[1]:
        return valence_class

    raise ValueError('An out-of-range valence value was encountered that could not be mapped to a class: ' + str(valence))

# Returns two DataLoaders, one for training and one for testing.
# Handles sampling the datasets, shuffling, etc. DataLoaders stack data
# vertically (by column) instead of horizonally (by row), meaning that a
# returned mini-batch will be a list of tensors, each representing a feature
# and having $BatchSize values.
def build_data_loaders(data_frame, model, batch_size):
  # Split the data set into two, 1000 records for testing, and all the rest for training.
  train_data_frame, test_data_frame = sklearn.model_selection.train_test_split(
    data_frame,
    test_size = 1000,
    shuffle = True
  )

  train_data_set = MusicDataSet(train_data_frame, model)
  test_data_set = MusicDataSet(test_data_frame, model)

  train_data_loader = torch.utils.data.DataLoader(
    train_data_set,
    batch_size = batch_size,
    shuffle = True
  )

  test_data_loader = torch.utils.data.DataLoader(
    test_data_set,
    batch_size = batch_size,
    shuffle = True
  )

  return train_data_loader, test_data_loader

def print_sample_mini_batch(data_loader):
  data_iterator = iter(data_loader)
  inputs, labels = data_iterator.next()

  print('Mini-batch inputs:')
  print(inputs)

  print('----------')

  print('Mini-batch labels:')
  print(labels)
