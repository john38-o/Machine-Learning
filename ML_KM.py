# K-Means

import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn import preprocessing

DataFrame = pd.read_excel('titanic.xls')
DataFrame.drop(['body', 'name'], 1, inplace=True)
DataFrame.apply(pd.to_numeric, errors='ignore')
DataFrame.fillna(-99999, inplace=True)

#																	 #
# DATA SET CONTAINS NON-NUMERICAL DATA -- ALL DATA MUST BE NUMERICAL #
#																	 #

def handle_non_numerical_data(df):
	# POPULATES LIST OF COLUMN 'INDECIES'
	columns = df.columns.values

	for column in columns:

		# FORMAT: {'TEXT': NUMBER}
		text_digit_vals = {}

		def convert_to_int(val):
			return text_digit_vals[val]

		# CHECKS WHAT TYPE OF DATA IS CONTAINED IN EACH COLUMN
		if df[column]. dtype != np.int64 and df[column].dtype != np.float64:

			# POPULATES LIST WITH COLUMN VALUES
			column_contents = df[column].values.tolist()
			# POPULATES LIST WITH EACH UNIQUE ELEMENT IN THE COLUMN
			unique_elements = set(column_contents)

			# ASSIGNS EACH UNIQUE ELEMENT A NUMBER VIA DICTIONARY
			x = 0 
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1

			# RE-SETS THE COLUMN W/ ITS CORRESPONDING NUMBER IN THE DICTIONARY 'text_digit_vals'
			df[column] = list(map(convert_to_int, df[column]))

	return df

DataFrame = handle_non_numerical_data(DataFrame)

#																	 #
# 					   TRAIN / TEST MODEL							 #
#																	 #

X  = np.array(DataFrame.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X) 
y = np.array(DataFrame['survived'])

clf = KMeans(n_clusters = 2)
clf.fit(X)

def KMeans_test(features, labels):
	X = features
	y = labels
	correct = 0 

	# FOR EACH ROW IN THE DATAFRAME -- WHICH IS SAVED AS AN NUMPY ARRAY 
	for i in range(len(X)):
		# POPULATES NUMPY ARRAY W/ VALUES FROM SPECIFIC ROW 'X[i]'
		test_row = np.array(X[i].astype(float))
		# PLACES 'test_row' INSIDE ANOTHER NUMPY ARRAY SO IT CAN BE RUN THROUGH CLASSIFIER
		# WANT THE CLASSIFIER TO CONSIDER ALL THE DATA IN THE ROW AT ONCE -- NOT EACH VALUE INDIVIDUALY
		test_row = test_row.reshape(-1, len(test_row))

		prediction = clf.predict(test_row)

		# IF THE PREDICTION--WHICH IS INCASED IN AN LIST-- IS EQUAL TO THE SURVIVE COLUMN VALUE FOR
		# THIS ROW 'y[i]' THEN THE PREDICTION WAS CORRECT
		if prediction[0] == y[i]:
			correct += 1

	# PRINTS OUT THE PERCENTAGE CORRECTLY CLASSIFIED
	return (correct/len(X))


#																	 #
# 					   BUILD FROM SCRATCH							 #
#																	 #

X = np.array([[1, 2],
			 [1.5, 1.8],
			 [5, 8],
			 [8, 8],
			 [1, 0.6],
			 [9, 11]])


class K_Means:

	def __init__(self, k=2, tol=0.001, max_iter=300):
		# NUMBER OF CLUSTERS TO FIND
		self.k = k
		# ONCE CENTROIDS' % CHANGE IS LESS THAN 'tol' -- STOP THE SEARCH
		self.tol = tol
		# MAXIMUM AMOUNT OF TIMES MODEL WILL CHECK FOR MORE ACCURATE CENTROIDS
		self.max_iter = max_iter

	def fit(self, data):
		
		# CENTROID DICTIONARY == { centroid #: [cent_x1, cent_x2] }\
		# EMPTYS DICTIONARY FOR EACH NEW INTERATIOIN
		self.centroids = {}

		for i in range(self.k):
			# ASSIGNS CENTROID STATUS TO THE FIRST TWO DATAPOINTS IN THE TRAINING SET
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			# CLASSIFICATIONS DICTIONARY == { classification: feature_set }
			# EMPTYS DICTIONARY FOR EACH NEW INTERATIOIN
			self.classifications = {}

			for j in range(self.k):
				# ASSIGNS 'k' CLASSIFICATIONS TO 'self.classifications' AND AN EMPTY LIST TO EACH
				self.classifications[j] = []

			for feature_set in data:
				# POPULATES A LIST WITH THE DISTANCES FROM THE SPEC DATAPOINT 'feature_set' TO EACH CENTROID
				# ONLY HOLDS TWO VALUES
				distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]

				# W/ ONLY TWO VALUES IN LIST -- ONE FOR K0 AND ONE FOR K1 -- THE INDEX OF THAT VALUE WILL
				# CORRESPOND W/ THE CENTROID CLASSIFICATION -- 'classification == 0 or 1'
				# SELECTS THE INDEX WITH THE SMALLEST DISTANCE VALUE -- CLOSEST TO CENTROID
				classification = distances.index(min(distances))

				# POPULATES CLASSIFICATIONS DICTIONARY WITH KEY: CLASSIFICATION AND VALUE: VECTOR-DIRECTION
				self.classifications[classification].append(feature_set)

			# BEFORE CENTROIDS ARE CHANGED, THE CURRENT CENTROIDS ARE ASSIGNED TO 'prev_centroids' FOR
			# LATER COMPARISON
			prev_centroids = dict(self.centroids)

			# FOR EACH DICTIONARY KEY IN 'self.classifications' 
			for classification in self.classifications:
				# FINDS THE AVERAGE VECTOR-DIRECTION OF EACH FEATURE-SET PER CLASS -- ASSIGNS NEW 
				# CENTROIDS TO THAT LOCATION
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)

			optimized = True


			for c in self.centroids:
				original_cenroid = prev_centroids[c]
				current_centroid = self.centroids[c]

				# CALCULATES THE % CHANGE IN DISTANCE FROM OLD CENTROID LOCATION TO NEW CENTROID LOCATION
				if np.sum((current_centroid - original_cenroid) / original_cenroid * 100.0) > self.tol:
					optimized = False

			if optimized:
				break

	def predict(self, features):

		# POPULATES A LIST WITH THE DISTANCES FROM THE SPEC DATAPOINT 'feature_set' TO EACH CENTROID
		# ONLY HOLDS TWO VALUES
		distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]

		# W/ ONLY TWO VALUES IN LIST -- ONE FOR K0 AND ONE FOR K1 -- THE INDEX OF THAT VALUE WILL
		# CORRESPOND W/ THE CENTROID CLASSIFICATION -- 'classification == 0 or 1'
		# SELECTS THE INDEX WITH THE SMALLEST DISTANCE VALUE -- CLOSEST TO CENTROID
		classification = distances.index(min(distances))

		return classification


unknowns = np.array([[1,3],
					[8,9],
					[0,3],
					[5,4],
					[6,4]])
clf = K_Means()
clf.fit(X)

# for unknown in unknowns:
# 	classification = clf.predict(unknown)
# 	print(classification)



