import numpy as np
from builtins import print
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import pickle

df = pd.read_csv('data/breast-cancer-wisconsin.data.txt', error_bad_lines=False)
df.replace('?', 1, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

pickle_in = open('trained_classifier/trained_tumor_classifier.pickle', 'rb')
clf = pickle.load(pickle_in)

# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train, y_train)

# with open('trained_classifier/trained_tumor_classifier.pickle', 'wb') as f:
#     pickle.dump(clf, f)

confidence = clf.score(X_test, y_test)
print(confidence)

example_measures = np.array([[8, 7, 5, 10, 7, 9, 5, 5, 4]])
example_measures = example_measures.reshape(len(example_measures), -1)
forecast_set = clf.predict(example_measures)
print(forecast_set)
