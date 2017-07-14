import numpy as np
import pandas as pd
from sklearn import model_selection, svm
import pickle

df = pd.read_csv('data/breast-cancer-wisconsin.data.txt', error_bad_lines=False)
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

pickle_in = open('trained_classifier/trained_tumor_classifier.pickle', 'rb')
clf = pickle.load(pickle_in)

# clf = svm.SVC()
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test)
# print(confidence)
#
# with open('trained_classifier/trained_tumor_svc_classifier.pickle', 'wb') as f:
#     pickle.dump(clf, f)

example_measures = np.array([[4, 2, 1, 1,1 ,2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), - 1)

prediction = clf.predict(example_measures)
print(prediction)