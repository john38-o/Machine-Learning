#ML_Regression

import pandas as pd 
import numpy as np
import quandl, math, datetime, random, pickle
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


# df = quandl.get('WIKI/GOOGL')

# df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
# df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# forecast_col = 'Adj. Close'
# df.fillna(-99999, inplace=True)

# forecast_out = int(math.ceil(0.01*len(df)))

# df['label'] = df[forecast_col].shift(-forecast_out)

# X = np.array(df.drop(['label'], 1))
# X = preprocessing.scale(X)
# X_lately = X[-forecast_out:]
# X = X[:-forecast_out]

# df.dropna(inplace=True)

# y = np.array(df['label'])

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)

# with open('linreg.pickle', 'wb') as file:
# 	pickle.dump(clf, file)

# pickle_in = open('linreg.pickle', 'rb')
# clf = pickle.load(pickle_in)

# accuracy = clf.score(X_test, y_test)
# forecast_set = clf.predict(X_lately)

#-----------------------------------------#
#		BUILD FROM SCRATCH				  #
#-----------------------------------------#


from statistics import mean

xs = [1,2,3,4,5,6]
ys = [5,4,6,5,6,7]

xs = np.array(xs, dtype=np.float64)
ys = np.array(ys, dtype=np.float64)

def BF_SLOPE(xs, ys):

	m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs)**2) - mean(xs**2))

	return m

m = BF_SLOPE(xs, ys)

def BF_YINT(xs, ys, m):

	b = mean(ys) - (m * mean(xs))

	return b

b = BF_YINT(xs, ys, m)

regression_line = [m*x + b for x in xs]

def sq_err(ys_orig, reg_line):
	return sum ((reg_line - ys_orig)**2)

def r_squared(ys, reg_line):
	y_mean_ln = [mean(ys) for y in ys]
	sq_er_reg = sq_err(ys, reg_line)
	sq_err_ym = sq_err(ys, y_mean_ln)
	return 1 - (sq_er_reg/sq_err_ym)

r_squared = r_squared(ys, regression_line)

#-----------------------------------------#
#				TESTING					  #
#-----------------------------------------#

def create_testing_dataset(how_much, variance, y_increase=2, correlation=False):
	val = 1 
	ys = []

	for i in range(how_much):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += y_increase
		elif correlation and correlation == 'neg':
			val -= y_increase
	xs = [i for i in range(how_much)]

	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

test_xs, test_ys = create_testing_dataset(40, 40, 2, correlation='pos')

test_m = BF_SLOPE(test_xs, test_ys)
test_b = BF_YINT(test_xs, test_ys, test_m)

test_reg_ln = [test_m * x + test_b for x in test_xs]

test_r_sq = r_squared(test_ys, test_reg_ln)








