# Predicting musical valence of Spotify songs using PyTorch

This is a project to predict the valence of songs in a dataset of 232,725 Spotify songs. Primarily, the goal was to build a machine learning model using [PyTorch](https://pytorch.org/) and deploy it to production as an API using [Flask](https://flask.palletsprojects.com/en/1.1.x/); the dataset just provided an interesting problem to solve in the process!

> You might remember "valence" from high school chemistry. It has to do with how many electrons an atom will lose, gain, or share when it joins with another atom. Psychologists put a spin on that concept, using the word "valence" to describe whether something is likely to make someone feel happy (positive valence) or sad (negative valence).

_[The Echo Nest](https://web.archive.org/web/20170422195736/http://blog.echonest.com/post/66097438564/plotting-musics-emotional-valence-1950-2013)_

The dataset, in addition to identifying the songs, artists, etc, contains values measuring their key, tempo, time signature, and so on, and includes a roughly equal number of around 10,000 songs per genre. Here are some example songs from the dataset, without any preprocessing:

| genre      | artist_name   | track_name   | track_id               | popularity | acousticness | danceability | duration_ms | energy | instrumentalness | key | liveness | loudness | mode  | speechiness | tempo   | time_signature | valence |
|:----------:|:-------------:|:------------:|:----------------------:|:----------:|:------------:|:------------:|:-----------:|:------:|:----------------:|:---:|:--------:|:--------:|:-----:|:-----------:|:-------:|:--------------:|:-------:|
| Electronic | BT            | Flaming June | 3zdsnLKSspBgqoBKowL6cJ | 29         | 0.0596       | 0.454        | 258333      | 0.814  | 0.0447           | F#  | 0.109    | -4.099   | Minor | 0.0546      | 137.964 | 4/4            | 0.168   |
| Rock       | Pink Floyd    | Eclipse      | 1tDWVeCR9oWGX8d5J9rswk | 62         | 0.0591       | 0.359        | 130429      | 0.579  | 0.746            | A#  | 0.0686   | -10.765  | Major | 0.0406      | 68.102  | 4/4            | 0.135   |
| Pop        | Justin Bieber | Baby         | 6epn3r7S14KUqlReYr77hA | 74         | 0.0544       | 0.656        | 214240      | 0.841  | 0                | F   | 0.122    | -5.183   | Minor | 0.232       | 65.024  | 4/4            | 0.522   |

After preprocessing the data (e.g. scaling and encoding the features as needed), it is fed into two models, one regression and one classification, both with the goal of learning to determine the valence of the given song(s).

## Results

Actually, before getting to the results, I should point out that the accuracy of the models was fairly predictable with only one line of code, simplified below:

```
$ print(data_frame.corr())

              Danceability    Energy  Loudness   Valence
Danceability      1.000000  0.325807  0.438668  0.547154
Energy            0.325807  1.000000  0.816088  0.436771
Loudness          0.438668  0.816088  1.000000  0.399901
Valence           0.547154  0.436771  0.399901  1.000000
```

And here's a more pretty visualisation using [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/), showing all features except for the categorical ones (Genre, Key, Mode and Time Signature):

![Numerical feature correlation heatmap.](docs/numerical_feature_correlation_heatmap.png?raw=true "Numerical feature correlation heatmap.")

As you can see, danceability, energy, and loudness are really the only three features that have a somewhat-significant positive linear correlation to valence, as shown by the pairwise correlation coefficients above. And even then the correlations aren't particularly strong. So even before starting on the models I wasn't all that optimistic of achieving great results.

Surprisingly, the regression model performed quite well: training after only one epoch results in an average Mean Squared Error of 0.036 and a Mean Absolute Error of 0.149. The latter number means that after training, when evaluating the test data set the model's predicted valence of a song has on average an absolute difference of 14.9% from the actual valence. Again, that's not incredible, but given the subjectivity of valence, and the fact that none of the features are highly correlated to it, it's pretty good.

## Requirements

Developed with Python version 3.8.2.

See dependencies.txt for packages and versions (and below to install).

## Setup

Clone the Git repo.

Install the dependencies:

```bash
$ pip install -r dependencies.txt
```

Download the [input data file](https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db) into the `data/spotify_features.csv` path.

## Run

At its simplest, the project can be run with sensible defaults by simply running:

```bash
$ python musical_valence_predictor.py
```

In addition, you can override the defaults by specifying the model type, number of training epochs, batch size, and so on. To view the supported arguments, run:

```bash
$ python musical_valence_predictor.py --help

usage: musical_valence_predictor.py [-h] [--model {regression,classification}] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--skip-training] [--print-batch]

PyTorch machine learning model to predict valence of a song based on musical characteristics, e.g. tempo, key, etc.

optional arguments:
  -h, --help            show this help message and exit
  --model {regression,classification}
                        Which model to run (regression or classification). Defaults to regression.
  --epochs EPOCHS       Number times that the training process will run through the training data set.
  --batch-size BATCH_SIZE
                        Number of examples from the training data set used per training iteration.
  --skip-training       Skip the training of the model and load a pre-trained one (model trains by default).
  --print-batch         Print a sample mini-batch from the training data set.
```

## API

You don't have to use the command line; the project wraps the regression model in a JSON API built with Flask, allowing for easy deployment. To start the server in development mode, run:

```bash
$ FLASK_ENV=development FLASK_APP=musical_valence_predictor flask run
```

Then query the API:

```bash
$ curl -v -H "Accept: application/json" "http://localhost:5000/predict?artist_name=BT&track_name=Yahweh"

< HTTP/1.0 200 OK
< Content-Type: application/json
< 
{
  "expected": 0.36, 
  "prediction": 0.34
}
```

And then open up [http://localhost:5000/predict](http://localhost:5000/predict).

## Dataset credits

The dataset used in this project was obtained from [Zaheen Hamidani](https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db), who in turn generated it using [this tool by Tomi Gelo](https://github.com/tgel0/spotify-data). The data itself was ultimately sourced from [Spotify's API](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/).
