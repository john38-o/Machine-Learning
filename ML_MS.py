# Mean Shift

import numpy as np 
import pandas as pd 
from sklearn.cluster import MeanShift 
from sklearn import preprocessing

DataFrame = pd.read_excel('titanic.xls')
Origional_DF = pd.DataFrame.copy(DataFrame)
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
# 					   		TRAIN MODEL								 #
#																	 #

X  = np.array(DataFrame.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X) 
y = np.array(DataFrame['survived'])

# clf = MeanShift()
# clf.fit(X)

		 ######## FINDING SURVIVAL RATES PER CLUSTER ########

def survival_rates_per_cluster(clf, Origional_DF):

	# GETS ALL LABELS -- ALL CLASSIFICATIONS/CLUSTER GROUP LABELS
	labels = clf.labels_
	# GETS CENTROID VECTOR-DIRECTIONS
	cluster_centers = clf.cluster_centers_

	Origional_DF['cluster_group']=np.nan

	for i in range(len(X)):
		# POPULATES COLUMN 'cluster_group' W/ EACH ROW'S CORRESPONDING CLUSTER GROUP
		Origional_DF['cluster_group'].iloc[i] = labels[i]

	# DETERMINES HOW MANY CLUSTERS MODEL FOUND
	n_clusters_ = len(np.unique(labels))
	# FORMAT: {cluster: survival rate}
	survival_rates = {}

	# FOR EACH CLUSTER GROUP
	for i in range(n_clusters_):

		# POPULATES TEMPORARY DATAFRAME W/ ONLY THE ROWS FROM THE ORIGINAL DATAFRAME
		# WHICH SHARE THE COMMON 'i' CLUSTER GROUP
	    temp_df = Origional_DF[ (Origional_DF['cluster_group']==float(i)) ]

	    # POPULATES DATAFRAME W/ ONLY THE ROWS FROM THE TEMPORARY DATAFRAME
		# WHICH SHARE THE COMMON 'survived' STATUS
	    survival_cluster = temp_df[  (temp_df['survived'] == 1) ]

	    # CALCULATES SURVIVAL RATE BY DIVIDING THE LEGNTH OF THE FRAME CONTAINING ALL THE 
	    # SURVIVERS FROM A SPEC CLUSTER BY THE LEGNTH OF THE ENTIRE CLUSTER GROUP
	    survival_rate = len(survival_cluster) / len(temp_df)

	    # FORMAT: {cluster: survival rate}
	    survival_rates[i] = survival_rate

	return survival_rates

    
#																	 #
# 					   BUILD FROM SCRATCH							 #
#																	 #

X = np.array([[1, 2],
			 [1.5, 1.8],
			 [5, 8],
			 [8, 8],
			 [1, 0.6],
			 [9, 11],
			 [8,2],
			 [10, 2], 
			 [9,2]])

class Mean_Shift:

	def __init__(self, bandwidth=None, radius_norm_step=100):
		# BANDWIDTH == THE AREA AROUND EACH CENTROID TO SEARCH
		self.bandwidth = bandwidth
		# DEFINES THE NUMBER OF RINGS AROUND EACH CENTROID THAT WILL BE CAST
		self.radius_norm_step = radius_norm_step 

	def fit (self, data):

		if self.bandwidth == None:
			# FIND THE CENTROID FOR THE ENTIRE DATA SET
			all_data_centroid = np.average(data, axis=0)
			# FINDS THE MAGNITUDE OF THE CENTROID
			all_data_norm = np.linalg.norm(all_data_centroid)
			# A RING W/ RADIUS 'all_data_norm' SHOULD CREATE A CIRCLE THAT CONTAINS ALL POINTS
			# THE BANDWIDTH USED SHOULD BE THE CIRCLE THAT ENCOMPASSES ALL POINTS DIVIDED 
			# BY THE RADIUS 'all_data_norm' OF THAT CIRCLE
			self.bandwidth = all_data_norm / self.radius_norm_step

		# FORMAT: {centroid #, vector-direction}
		centroids = {}

		for i in range(len(data)):
			# MAKES EACH DATA POINT A CENTROID
			centroids[i] = data[i]
		# POPULATES A LIST WITH 'WEIGHTS' IN DESCENDING ORDER FROM (STEP AMOUNT - 1) TO ZERO
		weights = [i for i in range(self.radius_norm_step)][::-1]

		while True:

			new_centroids = []

			for i in centroids:
				# A LIST TO CONTAIN THE VECTOR-DIRECTIONS OF OTHER POINTS WITHIN THIS SPEC
				# CENTROID'S BANDWIDTH
				in_bandwidth = [] 
				# GETS THE SPEC VECTOR-DIRECTION FOR THE CENTROID IN QUESTION
				centroid = centroids[i]

				for feature_set in data:
					# CALCULATES THE DISTANCE FROM EACH DATA POINT TO THE SPEC CENTROID
					distance = np.linalg.norm(feature_set - centroid)
					# IF THE DATA POINT AND CENTROID HAVE THE SAME VECTOR DISTANCE (i.e.) THEY'RE 
					# THE SAME POINT -- ASSIGN A REALLY SMALL VALUE TO 'distance' SO ITS 
					# 'weight_index' VALUE = 0
					if distance == 0:
						distance = 0.00000000001

					# CALCULATES HOW MANY RINGS OUT THE POOINT IS FROM THE CENTROID
					weight_index = int(distance/self.bandwidth)

					# IF THE DATA POINT LIES OUTSIDE THE OUTER-MOST RING, ASSIGN THE LARGEST 
					# INDEX TO ITS 'weight_index'
					if weight_index > self.radius_norm_step-1:
						weight_index = self.radius_norm_step-1

					# POPULATES A MASSIVE LIST W/ EACH DATA POINT 'weights[weight_index]**2' TIMES
					# THEN ADDS THEM TO THE POINTS IN THE CENTROID'S BANDWIDTH
					to_add = (weights[weight_index]**2)*[feature_set]
					in_bandwidth +=to_add

				# CALCULATES THE AVERAGE VECTOR-DIRECTION OF ALL WEIGHTED POINTS IN 'in_bandwidth' 
				# BECAUSE OF THE WEIGHTS THE AVERAGE SHOULD APPROACH THE TRUE CENTROID
				new_centroid = np.average(in_bandwidth, axis=0)
				new_centroids.append(tuple(new_centroid))

			# 'set()' RETURNS UNIQUE TUPLES, 'list()' PLACES THEM IN A LIST, 'sort()' SORTS THEM IN ASCENDING ORDER
			uniques = sorted(list(set(new_centroids)))

			# LIST TO REMOVE ALL BUT ONE DATAPOINT FROM A GROUP THAT ARE EXTREMELY CLOSE
			to_pop = []

			for i in uniques:
				for ii in uniques:
					if i == ii:
						pass
					# IF 'ii' CENTROID IS WITHIN THE BANDWIDTH (A RELATIVELY SMALL VALUE) OF 'i' CENTROID -- IF THE 
					# POINTS ARE A NEGLIGABLE DISTANCE FROM ONE ANOTHER, REMOVE ALL BUT ONE
					elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.bandwidth:
						to_pop.append(ii)
						break
			for i in to_pop:
				try:
					uniques.remove(i)
				except:
					pass

			# BEFORE CENTROIDS ARE CHANGED, THE CURRENT CENTROIDS ARE ASSIGNED TO 'prev_centroids' FOR
			# LATER COMPARISON
			prev_centroids = dict(centroids)

			# EMPTYS CENTROIDS DICTIONARY
			centroids = {}

			# ASSIGNS THE REMAINING UNIQUE CENTROIDS TO 'centroids' FOR THE NEXT ITERATION
			for i in range(len(uniques)):
				centroids[i] = np.array(uniques[i])

			optimized = True


			for i in centroids:
				# IF THE CENTROIDS HAVE MOVED BETWEEN ITERATIONS
				if not np.array_equal(centroids[i], prev_centroids[i]):
					optimized = False
				# IF EVEN JUST ONE CENTROID MOVED, THE MODEL IS NOT OPTIMIZED
				if not optimized:
					break

			if optimized:
				break

		# ASSIGNS THE FINAL GROUP OF CENTROIDS TO THE GLOBAL VARIABLE
		self.centroids = centroids

		# FORMAT: {centroid class: vector-distance}
		self.classifications = {}

		# FOR EACH CENTROID CLASS, CREATE A LIST FOR ITS CORRESPONDING DATA POINTS
		for i in range(len(self.centroids)):
			self.classifications[i] = []

		for feature_set in data:
			# POPULATES A LIST WITH THE DISTANCES FROM THE SPEC DATAPOINT 'feature_set' TO EACH CENTROID
			# HOLDS AS MANY VALUES AS THERE ARE REMAINING CENTROIDS
			distances = [np.linalg.norm(feature_set - c) for c in self.centroids]

			# W/ AS MANY VALUES AS THERE ARE REMAINING CENTROIDS IN LIST -- ONE FOR K0 AND ONE 
			# FOR K1... AND SO ON -- THE INDEX OF THAT VALUE WILL CORRESPOND W/ THE CENTROID CLASSIFICATION 
			# SELECTS THE INDEX WITH THE SMALLEST DISTANCE VALUE -- CLOSEST TO CENTROID
			classification = distances.index(min(distances))
			# POPULATES CLASSIFICATIONS DICTIONARY WITH KEY: CLASSIFICATION AND VALUE: VECTOR-DIRECTION 
			self.classifications[classification].append(feature_set)


	def predict(self, features):
		# POPULATES A LIST WITH THE DISTANCES FROM THE SPEC DATAPOINT 'feature_set' TO EACH CENTROID
		# HOLDS AS MANY VALUES AS THERE ARE REMAINING CENTROIDS
		distances = [np.linalg.norm(features - c) for c in self.centroids]

		# W/ AS MANY VALUES AS THERE ARE REMAINING CENTROIDS IN LIST -- ONE FOR K0 AND ONE 
		# FOR K1... AND SO ON -- THE INDEX OF THAT VALUE WILL CORRESPOND W/ THE CENTROID CLASSIFICATION 
		# SELECTS THE INDEX WITH THE SMALLEST DISTANCE VALUE -- CLOSEST TO CENTROID
		classification = distances.index(min(distances))

		return classification

clf = Mean_Shift()
clf.fit(X)
