import sklearn
from sklearn.model_selection import train_test_split
import torch

BATCH_SIZE = 1000

class MusicDataSet(torch.utils.data.Dataset):
  def __init__(self, data_frame):
    self.data_frame = data_frame

  def __len__(self):
    return len(self.data_frame)

  # Return a row from the Pandas data frame, plus the extracted label value.
  def __getitem__(self, index):
    row = self.data_frame.iloc[index]

    valence = row['Valence']
    del row['Valence']

    return row.values, valence

# Returns two DataLoaders, one for training and one for testing.
# Handles sampling the datasets, shuffling, etc. DataLoaders stack data
# vertically (by column) instead of horizonally (by row), meaning that a
# returned mini-batch will be a list of tensors, each representing a feature
# and having $BatchSize values.
def build_data_loaders(data_frame):
  # Split the data set into two, 1000 records for testing, and all the rest for training.
  train_data_frame, test_data_frame = sklearn.model_selection.train_test_split(
    data_frame,
    test_size = 1000,
    shuffle = True
  )

  train_data_set = MusicDataSet(train_data_frame)
  test_data_set = MusicDataSet(test_data_frame)

  train_data_loader = torch.utils.data.DataLoader(
    train_data_set,
    batch_size = BATCH_SIZE,
    shuffle = True
  )

  test_data_loader = torch.utils.data.DataLoader(
    test_data_set,
    batch_size = BATCH_SIZE,
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
