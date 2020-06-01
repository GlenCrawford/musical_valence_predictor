import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

INPUT_DATA_PATH = 'data/spotify_features.csv'
INPUT_DATA_COLUMN_NAMES = [
  'Genre', 'artist_name', 'track_name', 'track_id', 'popularity', 'Acousticness',
  'Danceability', 'duration_ms', 'Energy', 'Instrumentalness', 'Key', 'Liveness',
  'Loudness', 'Mode', 'Speechiness', 'Tempo', 'Time Signature', 'Valence'
]

# Quotes are from Spotify developer documentation of Tracks > Get Audio Features for a Track API endpoint.
INPUT_DATA_COLUMNS_TO_USE = [
  # Numerical columns.
  'Acousticness', # Float. Range: 0.0 to 1.0. Mean: 0.368560. Std: 0.354768. Range . "Confidence measure ... of whether the track is acoustic".
  'Danceability', # Float. Range: 0.0 to 1.0. Mean: 0.554364. Std: 0.185608.. Calculated from a "combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity".
  'Energy', # Float. Range: 0.0 to 1.0. Mean: 0.570958. Std: 0.263456.. Measure of "intensity and activity", calculated from "dynamic range, perceived loudness, timbre, onset rate, and general entropy".
  'Instrumentalness', # Float. Range: 0.0 to 1.0. Mean: 0.148301. Std: 0.302768. Whether a track contains no vocals.
  'Liveness', # Float. Range: 0.0 to 1.0. Mean: 0.215009. Std: 0.198273. Probability of the recording being a live performance, based on "presence of an audience in the recording".
  'Loudness', # Float. Range: -52.5 to 3.74. Mean: -9.569885. Std: 5.998204. "Overall [averaged] loudness of a track in decibels (dB)." Scale by z-score.
  'Speechiness', # Float. Range: 0.0 to 1.0. Mean: 0.120765. Std: 0.185518. "Presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0."
  'Tempo', # Float. Range: 30.3 to 242.9. Semantic range: 0 to 250. Mean: 117.666585. Std: 30.898907. "Overall estimated tempo of a track in beats per minute (BPM)."
  'Valence', # Float. Range: 0.0 to 1.0. Mean: 0.454917. Std: 0.260065. "Musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)."

  # Categorical columns. Listed last since Pandas will append the one-hot encoded columns to the end of the data frame anyway, so might as well document in that order.
  'Genre', # String. 27 unique values. See below for normalization of "Children's Music". One-hot encode.
  'Key', # String. 12 unique values: C (12%), G (11%), etc. "Estimated overall key of the track." Mapped from integers to pitches using Pitch Class notation (0 = C, etc.). One-hot encode.
  'Mode', # String. 2 unique values: Major (65%) and Minor (35%). "Modality (major or minor) of a track, the type of scale from which its melodic content is derived." One-hot encode.
  'Time Signature' # String. 5 unique values: 4/4, 5/4, etc. "Notational convention to specify how many beats are in each bar (or measure)". One-hot encode.
]

NUMERIC_COLUMNS_TO_SCALE = ['Loudness', 'Tempo']
CATEGORICAL_COLUMNS_TO_ONE_HOT_ENCODE = ['Genre', 'Key', 'Mode', 'Time Signature']

TRAINING_EPOCHS = 1
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

PRINT_TRAINING_PROGRESS_EVERY_N_MINI_BATCHES = 10

MODEL_SAVE_PATH = 'models/model.pth'

class MusicDataSet(torch.utils.data.Dataset):
  def __init__(self, data_frame):
    self.data_frame = data_frame

  def __len__(self):
    return len(self.data_frame)

  def __getitem__(self, index):
    row = self.data_frame.iloc[index]

    valence = row['Valence']
    del row['Valence']

    return row.values, valence

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.linear = torch.nn.Linear(
      53, # Number of inputs.
      1 # Number of outputs.
    )

  def forward(self, input):
    output = self.linear(input.float())
    return output

def main():
  # Prepare the data. Load, preprocess, split and build data loaders.
  data_frame = load_input_data()
  data_frame = preprocess_input_data(data_frame)
  train_data_loader, test_data_loader = build_data_loaders(data_frame)
  # print_sample_mini_batch(train_data_loader)

  model = Model()

  # Loss function: Mean absolute error (MAE).
  criterion = torch.nn.L1Loss(reduction = 'mean')

  # Adam optimization algorithm.
  optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay= 0)

  print('Training model...')
  train_model(model, train_data_loader, criterion, optimizer)
  print('Finished training.')

  save_model(model)

def load_input_data():
  data_frame = pd.read_csv(
    INPUT_DATA_PATH,
    header = 0,
    names = INPUT_DATA_COLUMN_NAMES,
    usecols = INPUT_DATA_COLUMNS_TO_USE
  )

  # Re-order the columns (column order is ignored by the usecols argument).
  data_frame = data_frame[INPUT_DATA_COLUMNS_TO_USE]

  return data_frame

def preprocess_input_data(data_frame):
  # Column "Genre" has both "Children's Music" and "Children’s Music" (different apostrophe character). Normalize them.
  data_frame.replace({'Genre': {'Children’s Music': 'Children\'s Music'}}, inplace=True)

  # Scale/normalize numeric columns by calculating the z-score of each value.
  z_score_scaler = sklearn.preprocessing.StandardScaler(copy = True)
  data_frame[NUMERIC_COLUMNS_TO_SCALE] = z_score_scaler.fit_transform(data_frame[NUMERIC_COLUMNS_TO_SCALE].to_numpy())

  # Apply one-hot encoding to categorical features.
  data_frame = pd.get_dummies(
    data_frame,
    columns = CATEGORICAL_COLUMNS_TO_ONE_HOT_ENCODE,
    sparse = False
  )

  return data_frame

# Returns two DataLoaders, one for training and one for testing.
# Handles sampling the datasets, shuffling, etc. DataLoaders stack data
# vertically (by column) instead of horizonally (by row), meaning that a
# returned mini-batch will be a list of tensors, each representing a feature
# and having $BatchSize values.
def build_data_loaders(data_frame):
  # Split the data set into two, 1000 record for testing, and all the rest for training.
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

  print('Mini-batch Labels:')
  print(labels)

def train_model(model, data_loader, criterion, optimizer):
  for epoch in range(TRAINING_EPOCHS):
    running_loss = 0.0

    for i, mini_batch in enumerate(data_loader, 0):
      inputs, labels = mini_batch

      # Zero the parameter gradients.
      optimizer.zero_grad()

      outputs = model(inputs)

      # Get loss for the predicted outputs.
      loss = criterion(outputs, labels)

      # Get gradients w.r.t to parameters.
      loss.backward()

      # Update parameters.
      optimizer.step()

      # Track statistics. Print every N mini-batches.
      running_loss += loss.item()
      if i % PRINT_TRAINING_PROGRESS_EVERY_N_MINI_BATCHES == (PRINT_TRAINING_PROGRESS_EVERY_N_MINI_BATCHES - 1):
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / PRINT_TRAINING_PROGRESS_EVERY_N_MINI_BATCHES))
        running_loss = 0.0

def save_model(model):
  torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == '__main__':
  main()
