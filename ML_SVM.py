#Support-Vector Machine

import numpy as np 
from sklearn import preprocessing, cross_validation, svm 
import pandas as pd

df = pd.read_csv('breastcancer.data.webarchive')
df.replace('?', -99999, inplace=True)
df.drop(['sample_code_number'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

exmp_measures = np.array([4,2,1,1,1,2,3,2,1])

prediction = clf.predict(exmp_measures)


#-----------------------------------------#
#		BUILD FROM SCRATCH				  #
#-----------------------------------------#

# THE WHOLE IDEA IS TO FIND VALUES FOR 'w' AND 'b' SO THAT WHEN THE 
# EQUATION 'xi . w + b' IS RUN, CLASSIFICATIONS ARE GIVEN BASED 
# ON WHETHER THE RUSULT IS ABOVE OR BELOW 0

#------------------------------------------------------------#

# SMART TO BUILD A SVM OBJECT SO YOU CAN FIND 'w' AND 'b' -- TRAIN THE MODEL
# AFTER 'w' AND 'b' ARE SAVED IN THE SVM OBEJECT, PREDICTIONS CAN BE VERY EFFICIENT
class SVM:
	def __init__(self):
		pass

	# TRAINS THE DATA -- FINDS 'w' AND 'b'
	def fit(self, data):
		self.data = data

		# Optimization Dictionary == {||w||: [w,b]}
		opt_dict = {}

		# MUST RUN CALC ON W IN EACH 'QUADRANT'
		transforms = [[ 1, 1],
					  [-1 ,1],
					  [1, -1],
					  [-1,-1]]

		# POPULATES LIST 'all_data' W/ EACH INDIVIDUAL DATA POINT
		# 'yi' == THE CLASS OF THE DATA -- EITHER 1 / -1
		# EACH 'yi' CONTAINS A LIST OF VECTORS -- BELOW 'feature_set'
		all_data = []
		for yi in self.data:
			for feature_set in self.data[yi]:
				for feature in feature_set:
					all_data.append(feature)

		# GENERATES MAX/MIN VALUES FROM THE LIST
		self.max_feature_value = max(all_data)
		self.min_feature_value = min(all_data)

		# CLEAR 'all_data' AS TO NOT WASTE MEMORY
		all_data = None


		step_sizes = [self.max_feature_value * 0.1,
					  self.max_feature_value * 0.01,
					  #'0.001' STEPS EXPEND A LOT OF PROCESSING
					  self.max_feature_value * 0.001]

		# 'b-steps' VERY EXPENSIVE		  
		b_range_multiple = 5

		#DON'T NEED TO TAKE AS SMALL OF STEPS WHEN LOOKING FOR 'b'
		b_multiple = 5

		# CREATES A STARTING POINT FOR THE STEPPER -- VERY HIGH AS TO NOT MISS ANYTHING
		latest_optimum = self.max_feature_value*10

		# STEPS DOWN THE 'U' LOOKING FOR THE MINIMUM POINT
		for step in step_sizes:

			# VECTOR TO CHECK
			w = np.array([latest_optimum,latest_optimum])

			#OPTIMIZED == CHECKED ALL VALUES BETWEEN 'w' AND 0
			optimized = False

			while not optimized:

				# PROVIDES 'b' FOR THE FOLLOWING CODE -- ITERATES OVER A SMALLER RANGE THAN 'w' 
				# ALSO TAKES CONSIDERABLALY LARGER STEPS
				for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
									  (self.max_feature_value*b_range_multiple), 
									  step*b_multiple):

					# TAKES 'w' FROM ABOVE AND RUNS THE FOLLOWING FOR EACH OF ITS 'TRANSOFRMATIONS'
					for transformations in transforms:
						w_t = w * transformations

						#'found-option' = FOUND A 'w'/'b' COMBO THAT NETS >= 1
						found_option = True

						# RUNS 'yi(xi . w + b) >= 1' FOR EVERYDATA 
						# POINT W/ ITS CORRESPONDING SIGN (1 / -1)
						for yi in self.data:
							for xi in self.data[yi]:

								# IF 'yi(xi . w + b)' < 1 THEN THIS 'w/'b' 
								# COMBO IS NOT A VIABLE OPTION
								if not yi * (np.dot(w_t,xi) + b) >= 1:
									found_option = False

						# IF 'yi(xi . w + b)' >= 1 THEN THIS 'w/'b' 
						# COMBO IS A VIABLE OPTION AND SHOULD BE PUT IN THE 
						# OPTIMIZATION DICTIONARY 'opt_dict' IN THE FORMAT 
						# { ||w||: [w,b] }
						if found_option:
							opt_dict[np.linalg.norm(w_t)] = [w_t,b]

				# IF ALL VALUES BETWEEN 'w' AND 0 ARE CHECKED
				if w[0] < 0:
					optimized = True
					print('Optimized a step.')
				else:
					w = w - step

			# POPULATES A LIST 'norms' WITH THE KEYS FROM 'opt_dict' IN ASCENDING ORDER
			norms = sorted([n for n in opt_dict])

			# norms[0] CONTAINS THE SMALLEST ||w|| VALUE 
			opt_choice = opt_dict[norms[0]]

			# ASSIGNS 'w' AND 'b' TO THEIR GLOBAL VARIABLES -- CAN BE OVERWRITTEN 
			# IF MORE ACCURATE VALUES ARE FOUND
			self.w = opt_choice[0]
			self.b = opt_choice[1]

			# opt_choice[0][0] == w's x1 VALUE -- STEPPING THIS DOWN NARROWS 
			# THE SEARCH ON THE NEXT RUN
			latest_optimum = opt_choice[0][0]+(step*10)

	def predict(self, predictor):

		# THE FOLLOWING IS NUMPY FOR FINDING THE 'SIGN' OF A VARIABLE
		# IF x . w + b > 0 -- RETURN 1 | | IF x . w + b < 0 -- RETURN -1
		classification = np.sign(np.dot(np.array(predictor), self.w) + self.b)

		return classification



data_dict = {-1: np.array([[1,7],
							[2,8],
							[3,8]]),

			  1: np.array([ [5,1],
			  				[6,-1],
			  				[7,3]])}

svm = SVM()
svm.fit(data=data_dict)
prediction = svm.predict([1,3])

