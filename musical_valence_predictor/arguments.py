import argparse

def parse_arguments():
  parser = argparse.ArgumentParser(
    allow_abbrev = False,
    description = 'PyTorch machine learning model to predict valence of a song based on musical characteristics, e.g. tempo, key, etc.'
  )

  parser.add_argument(
    '--model',
    action = 'store',
    default = 'regression',
    choices = ['regression', 'classification'],
    dest = 'model',
    help = 'Which model to run (regression or classification). Defaults to regression.'
  )

  parser.add_argument(
    '--epochs',
    action = 'store',
    default = 3,
    type = int,
    dest = 'epochs',
    help = 'Number times that the training process will run through the training data set.'
  )

  parser.add_argument(
    '--batch-size',
    action = 'store',
    default = 1000,
    type = int,
    dest = 'batch_size',
    help = 'Number of examples from the training data set used per training iteration.'
  )

  parser.add_argument(
    '--skip-training',
    action = 'store_false',
    dest = 'train',
    help = 'Skip the training of the model and load a pre-trained one (model trains by default).'
  )

  parser.add_argument(
    '--print-batch',
    action = 'store_true',
    dest = 'print_sample_mini_batch',
    help = 'Print a sample mini-batch from the training data set.'
  )

  return parser.parse_args()
