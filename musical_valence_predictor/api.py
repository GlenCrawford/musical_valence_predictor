import torch
from flask import jsonify, request
import musical_valence_predictor.serialization as Serialization
import musical_valence_predictor.data_preprocessing as DataPreprocessing

# Load model outside of API endpoint(s), so it's loaded once, not upon each request.
model = Serialization.load_model()

# Switch to eval mode, since the model is only being used for inference.
model.eval()

data_frame = DataPreprocessing.load_input_data()
data_frame = DataPreprocessing.preprocess_input_data(data_frame)

# GET /predict
def predict():
  # Start with just the simplest thing that will work, to get up and running.
  # Pick a random song from the data set and run it through the model, returning
  # the valence prediction.
  track_row = data_frame.sample()

  # Remove the target ;)
  expected_valence = track_row['Valence'].values.item()
  del track_row['Valence']

  input = torch.tensor(track_row.values)
  output = model.forward(input)[0].detach()

  return jsonify({
    'expected': expected_valence,
    'prediction': output.numpy().item()}
  )
