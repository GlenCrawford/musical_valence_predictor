import torch
import musical_valence_predictor.models as Models

# Relative from top-level directory.
MODEL_SAVE_PATH = 'models/model.pth'

def load_model():
  model = Models.RegressionModel.RegressionModel()
  model.load_state_dict(torch.load(MODEL_SAVE_PATH))
  return model

def save_model(model):
  torch.save(model.state_dict(), MODEL_SAVE_PATH)
