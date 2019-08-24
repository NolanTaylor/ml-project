# load and evaluate a saved model
import pandas
from numpy import loadtxt
from keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# load model
model = load_model('C:\PythonPrograms\mlproject\model.h5')
# summarize model.
model.summary()
# load dataset
dataframe = pandas.read_csv("C:\PythonPrograms\mlproject\iris.data", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
# evaluate the model
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))