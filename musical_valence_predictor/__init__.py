from flask import Flask
import musical_valence_predictor.api as API

# Flask API.
app = Flask(__name__)
app.add_url_rule(
  '/predict',
  view_func = API.predict,
  methods = ['GET']
)
