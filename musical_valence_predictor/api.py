import torch
from flask import jsonify, request
import musical_valence_predictor.serialization as Serialization
import musical_valence_predictor.data_preprocessing as DataPreprocessing
import musical_valence_predictor.models as Models

data_frame = DataPreprocessing.load_input_data(include_identifying_columns = True)
data_frame = DataPreprocessing.preprocess_input_data(data_frame)

# GET /predict
def predict():
  # In the real-world, we would load the model outside of API endpoint(s), so that it's loaded once, not upon each request.
  model = Models.RegressionModel.RegressionModel()
  model = Serialization.load_model(model)

  # Switch to eval mode, since the model is only being used for inference.
  model.eval()

  track_row = find_track_from_params()

  if track_row.empty:
    return 'Track not found for the given params.', 422

  expected_valence = track_row['Valence'].values.item()

  # Remove the identifying columns. And don't forget to remove the target ;)
  track_row.drop(columns = ['Artist Name', 'Track Name', 'Valence'], inplace = True)

  input = torch.tensor(track_row.values)
  output = model.forward(input)[0].detach()

  return jsonify({
    'expected': expected_valence,
    'prediction': round(output.numpy().item(), 2)}
  )

def find_track_from_params():
  return data_frame.loc[(data_frame['Artist Name'] == request.args.get('artist_name')) & (data_frame['Track Name'] == request.args.get('track_name'))]
