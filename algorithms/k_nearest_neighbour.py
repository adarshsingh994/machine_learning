import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')

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

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100)
result = k_nearest_neighbour(dataset, new_features)
plt.scatter(new_features[0], new_features[1], s=100, color=result)
print(result)
plt.show()
