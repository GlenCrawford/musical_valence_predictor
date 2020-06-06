import torch
import musical_valence_predictor.models as Models
import musical_valence_predictor.serialization as Serialization

LEARNING_RATE = 0.001
TRAINING_EPOCHS = 1
PRINT_TRAINING_PROGRESS_EVERY_N_MINI_BATCHES = 10

def train(model, data_loader):
  # Loss function.
  criterion = torch.nn.CrossEntropyLoss()

  # Optimization algorithm.
  optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9)

  print('Training model...')

  for epoch in range(TRAINING_EPOCHS):
    running_loss = 0.0

    for i, mini_batch in enumerate(data_loader, 0):
      inputs, labels = mini_batch

      # Zero the parameter gradients.
      optimizer.zero_grad()

      outputs = model(inputs)

      print('Labels:')
      print(str(labels))
      print('\n\n=========\n\n')
      print('Outputs:')
      print(str(outputs))

      # Get loss for the predicted outputs.
      loss = criterion(outputs, labels)

      print('\n\n==========\n\n')
      print('Loss:')
      print(str(loss))
      exit()

      # Get gradients with respect to parameters.
      loss.backward()

      # Update parameters.
      optimizer.step()

      # Track statistics. Print every N mini-batches.
      running_loss += loss.item()
      if i % PRINT_TRAINING_PROGRESS_EVERY_N_MINI_BATCHES == (PRINT_TRAINING_PROGRESS_EVERY_N_MINI_BATCHES - 1):
        print('[%d, %5d] Cross-entropy loss: %.3f' % (epoch + 1, i + 1, running_loss / PRINT_TRAINING_PROGRESS_EVERY_N_MINI_BATCHES))
        running_loss = 0.0

  print('Finished training.')

  Serialization.save_model(model)
  print('Saved model.')