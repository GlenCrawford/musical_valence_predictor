import torch
import musical_valence_predictor.models as Models

# Relative from top-level directory.
MODEL_SAVE_PATH = 'models/'

def load_model(model):
  model.load_state_dict(torch.load(file_path_for_model(model)))
  return model

def save_model(model):
  torch.save(model.state_dict(), file_path_for_model(model))

def file_path_for_model(model):
  return MODEL_SAVE_PATH + type(model).__name__ + '.pth'
