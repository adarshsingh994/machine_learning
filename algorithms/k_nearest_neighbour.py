import random
import pandas as pd
import warnings
from math import sqrt
from collections import Counter


def k_nearest_neighbour(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = sqrt((features[0] - predict[0]) ** 2 + (features[1] - predict[1]) ** 2)

            # Using numpy
            # euclidean_distance = np.sqrt(np.sum(np.array(features) - np.array(predict)) ** 2)

            # Using norm
            # euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))

            distances.append([euclidean_distance, group])

            vote = [i[1] for i in sorted(distances)[:k]]
            vote_result = Counter(vote).most_common(1)[0][0]
    return vote_result

df = pd.read_csv('../data/breast-cancer-wisconsin.data.txt', error_bad_lines=False)
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbour(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total)