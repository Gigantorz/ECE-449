# lab 3
# !pip install tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import pandas as pd

from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Layer
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from itertools import permutations, product
from statistics import mean
import random 

class Classifier(BaseEstimator, TransformerMixin):
  """
  Creates and trains classifier on the Stratified K Fold

  ...
  Methods
  ```````
  make_model:
    makes desired Sequential model
  evaluate:
    Evaluates model
  fit:
    To fit the dataset with model and to find the optimal model
  transform:
    returns model
  """

  # Will run hyperparameter exploration and choose the most optimal model
  def optimal(self, X):
    values = {}
    for (column_name, column_data) in X.iteritems():
      values[column_name] = column_data
    column_names = values.keys()

    num_filters = [16, 32]
    learning_rate = [0.001, 0.01]

    # The combination with the highest mean scores will be the optimal model
    # mean_result (key) : combo list (value)
    mean_results_dict = {}

    # Will merge all the collected values then add all the same keys
    # The key with the max value will be the ideal K value
    results_kfold_mean = {}
    # - might not even need anymore.

    # Permutation of all possibility [16, 32] can be in a length 3 list.
    iter = [16, 32]
    permutation = []
    for filter_combo in product(num_filters, repeat = 3):
      print(type(filter_combo))
      for rate in learning_rate:
        combo = [filter_combo, rate]
        model = self.make_model2(filter = filter_combo, lr = rate)
        mean_results = self.evaluate(model, X)
        mean_results_dict[mean_results] = combo
      permutation.append(filter_combo)

    print(permutation)
    optimal = 0
    for key in mean_results_dict:
      if key > optimal:
        optimal = key

    print(optimal)
    print("This is the", mean_results_dict)
    arr_optimal = mean_results_dict[optimal]
    ideal_model = self.make_model2(filter = arr_optimal[0], lr = arr_optimal[1])
    ideal_model.summary()
    ideal_model.save("temp_save.keras")
    return ideal_model

  # Will evaluate the model using 5-fold
  # --- This will take a while to make ---
  def evaluate(self, model, X):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    list_scores = []
    skf = StratifiedKFold(n_splits = 5)
    for train, test in skf.split(X_train, y_train):
        print("\nThis is the y value with train index, {} and y_train {}".format(train, y_train[train]))
        model.fit(X_train[train], y_train_encoded[train], epochs = 150, batch_size = 10, verbose = 0)
        score = model.evaluate(X_train[test], y_train_encoded[test], verbose = 0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1]) # we are only going to take into account the accuracy
        list_scores.append(score[1] * 100)

    print("These are the list of scores: ", list_scores)
    mean_results = mean(list_scores)
    # mean_results = random.randrange(80.0, 97.0)
    return mean_results

  # Will make a model of version 1
  def make_model1(self, filter = [], lr = 0.001):
    # From Code given
    model = models.Sequential()

    #Define filters and convolutional layers here
    model.add(layers.Conv2D(filters=filter[0], kernel_size=(3, 3),
    activation='relu', input_shape=(28, 28, 1)))

    #Add a maxpooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    #Flatten the output and give it to a fully connected layer
    model.add(layers.Flatten())

    #One hidden layer maps the flattened neurons to output
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

    # Will make a model of version 2
  def make_model2(self, filter = [], lr = 0.001):
    # From Diagram

    model = models.Sequential()
    #define filters and convolutional layers here
    # we have 16, 3x3 filters.
    # Each of the 16 filters produces a 26×26 feature map.
    # So the output shape of the Conv2D layer in this case would be 26 x 26 x 16.
    model.add(layers.Conv2D(filters=filter[0], kernel_size=(3, 3), activation='relu',
          input_shape=(28, 28, 1)))

    #  The input_shape parameter is only required for the first layer in the model,
    # as subsequent layers can automatically infer
    # the input shape from the output shape of the previous layer.
    model.add(layers.Conv2D(filters=filter[1], kernel_size=(3, 3), activation='relu'))

    #Add a maxpooling layer
    # The max pooling layer reduces the dimension by a factor of 2 in this case.
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=filter[2], kernel_size=(3, 3), activation='relu'))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    #Flatten the output and give it to a fully connected layer
    model.add(layers.Flatten())
    #one hidden layer maps the flattened neurons to output
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

  # This will be executed when fit_transform is called.
  def fit(self, X, y=None):
    model = Sequential()
    model = self.optimal(X)
    print("This is the most optimal model... Yay!!!", model.summary())
    return self

  def transform(self, X):
    return X
  
if __name__ == "__main__":
    print("hello world!")
    data = pd.DataFrame()
    neuron = Classifier()
    d1 = neuron.fit_transform(data)