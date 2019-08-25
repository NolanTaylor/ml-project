# load and evaluate a saved model
import numpy as np
import pandas
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.utils import np_utils
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_model():
    # loading model
    model = model_from_json(open('C:\PythonPrograms\mlproject\model_architecture.json').read())
    model.load_weights('C:\PythonPrograms\mlproject\model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

flowers = pandas.read_csv("C:\PythonPrograms\mlproject\iris.data", header=None)
values = flowers.values
X = values[:,0:4]

# load model
model = load_model()

predictions = model.predict_classes(X, verbose=0)
print(predictions)