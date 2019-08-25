# Train model and make predictions
import numpy
import pandas
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.utils import np_utils
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
iris = datasets.load_iris()
X, Y, labels = iris.data, iris.target, iris.target_names
X = preprocessing.scale(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_Y)

flowers = pandas.read_csv("C:\PythonPrograms\mlproject\iris.data", header=None)
values = flowers.values
X = values[:,0:4]

def build_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def save_model(model):
    # saving model
    json_model = model.to_json()
    open('C:\PythonPrograms\mlproject\model_architecture.json', 'w').write(json_model)
    # saving weights
    model.save_weights('C:\PythonPrograms\mlproject\model_weights.h5', overwrite=True)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# build
model = build_model()
model.fit(X_train, Y_train, nb_epoch=200, batch_size=5, verbose=0)

# save
save_model(model)