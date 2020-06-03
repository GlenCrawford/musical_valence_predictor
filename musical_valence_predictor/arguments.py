import argparse

def parse_arguments():
  parser = argparse.ArgumentParser(
    allow_abbrev = False,
    description = 'PyTorch machine learning model to predict valence of a song based on musical characteristics, e.g. tempo, key, etc.'
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
