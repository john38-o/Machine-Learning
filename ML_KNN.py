#K Nearest Neighbors

import numpy as np 
from sklearn import preprocessing, cross_validation, neighbors 
import pandas as pd 
from math import sqrt
from collections import Counter
import warnings, random

df = pd.read_csv('breastcancer.data.webarchive')
df.replace('?', -99999, inplace=True)
df.drop(['sample_code_number'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

exmp_measures = np.array([4,2,1,1,1,2,3,2,1])

prediction = clf.predict(exmp_measures)
print('SciKit Learn:', accuracy)

#-----------------------------------------#
#		BUILD FROM SCRATCH				  #
#-----------------------------------------#

df = pd.read_csv('breastcancer.data.webarchive')
df.replace('?', -99999, inplace=True)
df.drop(['sample_code_number'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

def k_nearest_neighbors(data, predict, k=3):

	# IF K IS AN EVEN NUMEBER THE VOTE RESULTS COULD BE A TIE (INCONCLUSIVE)
	# IF K <= THE NUMBER OF CLASSES IN THE DATA SET THERE WILL NOT BE ENOUGH VOTES 
	if len(data) >= k:
		warnings.warn('K is too low!')
	if k % 2 == 0:
		warnings.warn('K should not be even')

	# BREAKS DATASET (DICTIONARY) DOWN TO ITS INDIVIDUAL COORDINATES
	distances = []
	for group in data:
		for features in data[group]:

			# TESTS FOR DISTANCE BETWEEN EACH POINT AND THE POINT-TO-PREDICT
			# BELOW IS THE NUMPY VERSION OF CALCULATING EUCLIDIAN DISTANCE
			euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))

			# POPULATES 'distances' W/ EACH DISTANCES AND ITS CORRESPONDING 'GROUP'
			distances.append([euclidian_distance, group])

	# SORTS 'distances' IN ASCENDING ORDER AND PICKS THE FIRST 'k' LISTS
	# POPULATES VOTES WITH THE 'GROUP' LABELS OF THE FIRST 'k' LISTS
	votes = [i[1] for i in sorted(distances)[:k]]

	# FINDS THE MOST COMMON 'GROUP' LABEL IN 'votes'
	# THE CLASSIFICATION -- TO WHICH 'GROUP' THE DATAPOINT BELONGS 
	vote_results = Counter(votes).most_common(1)[0][0]

	# DEVIDES THE NUMBER OF MOST-COMMON 'GROUP' LABEL BY 'k' TO DETERMINE 'CONFIDENCE'
	# HOW MANY WERE PREDICTED CORRECTLY OUT OF HOW MANY WERE TESTED
	confidence = Counter(votes).most_common(1)[0][1] / k

	return vote_results, confidence

#-----------------------------------------#
#		TESTING W/ RANDOM DATA			  #
#-----------------------------------------#

random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

[train_set[i[-1]].append(i[:-1]) for i in train_data]
[test_set[i[-1]].append(i[:-1]) for i in test_data]

correct = 0
total = 0 


for group in test_set:
	for data in test_set[group]:
		vote, confidence = k_nearest_neighbors(train_set, data, k=5)
		if group == vote:
			correct += 1
		total += 1
