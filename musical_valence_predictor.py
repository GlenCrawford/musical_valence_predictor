import musical_valence_predictor.data_preprocessing as DataPreprocessing
import musical_valence_predictor.data_loaders as DataLoaders
import musical_valence_predictor.serialization as Serialization
import musical_valence_predictor.models as Models
import musical_valence_predictor.train as Train
import musical_valence_predictor.test as Test

def main(train = False):
  # Prepare the data. Load, preprocess, split and build data loaders.
  data_frame = DataPreprocessing.load_input_data()
  data_frame = DataPreprocessing.preprocess_input_data(data_frame)

  train_data_loader, test_data_loader = DataLoaders.build_data_loaders(data_frame)

  # Uncomment below to view a mini-batch from the train data set.
  # DataLoaders.print_sample_mini_batch(train_data_loader)

  if train:
    model = Models.RegressionModel.RegressionModel()
    Train.RegressionModel.train(model, train_data_loader)
  else:
    model = Serialization.load_model()

  Test.RegressionModel.test(model, test_data_loader)

if __name__ == '__main__':
  main(train = True)
