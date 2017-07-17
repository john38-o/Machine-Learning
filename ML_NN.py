# Neural Networks

import tensorflow as tf 
# PIXEL VALUE DATASET FROM INT'S 0-9 #
from tensorflow.examples.tutorials.mnist import input_data

INPUT = input('\n\n\nWhich model would you like to run?\n\n')

#																	 #
#							 BASIC CONCEPTS 			  			 #
#																	 #

x1 = tf.constant(5)
x2 = tf.constant(6)

# RETURNS AN ABSTRACT TENSOR-- NOT '30'
# PROCESSES RUN 'OUT-OF SESSION' DON'T GENERATE VALUES -- THEY CREATE MODELS
result = tf.multiply(x1,x2)

# CREATES SESSION OBJECT
sesh = tf.Session()
# RETRIEVES ACTUAL 'result' VALUE
with tf.Session() as sesh:
	sesh.run(result)

### BASIC MODEL (FEED FORWARD NN) ###
'''
INPUT LAYER > WEIGHT > HIDDEN LAYER 1 (ACTIVATION FUNCTION) > WEIGHTS > HIDDEN LAYER 2
(ACTIVATION FUNCTION) > WEIGHTS > OUTPUT LAYER

COMPARE OUTPUT TO INTENDED OUTPUT VIA COST FUNCTION (i.e. CROSS ENTROPY)
RUN OPTIMIZATION FUNCTION (OPTIMIZER) TO MINIMIZE COST (i.e. ADAMOPTIMIZER, SGD, ADAGRAD)

PERFORMS 'BACKPROPOGATION' BY GOING BACK THROUGH THE NETWORK AND MANIPULATES WEIGHTS

FEED FORWARD + BACKPROP = 'EPOCH'

'''

#																	 #
#							 MNIST MODEL  			  			 	 #
#																	 #

class MNIST_Network:

	# ESTABLISHES THE COMPUTATION GRAPH FOR THE MODEL
	def __init__(self):
		self.mnist = input_data.read_data_sets('/tmp/data/', one_hot=True) 
		'''
		one_hot' MEANS CLASSIFICATIONS WILL BE A LIST WITH ONE DEFINING '1' AND THE REST '0'

		FOR THIS EXAMPLE -- POSSIBLE CLASSIFICATIONS 0-9
		CLASSIFICATION '0' == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		'''

		# DEFINE THE AMOUNT OF NODES PER 'HIDDEN LAYER'
		self.n_nodes_hl1 = 500
		self.n_nodes_hl2 = 500
		self.n_nodes_hl3 = 500

		# DEFINE NUMBER OF CLASSES 
		self.n_classes = 10 # NUMBERS 0-9

		# FOR MEMORY AND PROCESSING PURPOSES, DEFINE 'batch_size' WHEN DEALING WITH V LARGE DATASETS
		# RUNS BATCH THROUGH THE MODEL, MANIPULATES THE WEIGHTS, THEN DOES THE NEXT BATCH
		self.batch_size = 100

		# PLACEHOLDERS ARE GOOD TO ENSURE INPUTS THROWN THROUGH THE MODEL ARE IN THE APPROPRIATE FORMAT
		# DEFINES A TYPE AND SHAPE FOR INPUT DATA TO BE
		self.x = tf.placeholder('float', [None, 784]) # [None, 784] = [matrix_height, matrix_width]
		self.y = tf.placeholder('float')

	def neural_network_model(self, data):

		# NEURON EQUATION: ''(INPUT DATA * WEIGHTS) + BIAS'' -- BIAS IS ADDED INCASE ALL INPUTS ARE EMPTY  

		# ASSIGNS RANDOM 'TENSORFLOW VARIABLES' AS WEIGHTS		  [MATRIX SIZE, NODES]
		hidden_layer_1 = { 'weights': tf.Variable(tf.random_normal([784, self.n_nodes_hl1])),
						   'biases': tf.Variable(tf.random_normal([self.n_nodes_hl1])) }
		
		hidden_layer_2 = { 'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
						   'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2])) }
		
		hidden_layer_3 = { 'weights': tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl3])),
						   'biases': tf.Variable(tf.random_normal([self.n_nodes_hl3])) }
		
		output_layer = { 'weights': tf.Variable(tf.random_normal([self.n_nodes_hl3, self.n_classes])),
						   'biases': tf.Variable(tf.random_normal([self.n_classes])) }


		# CREATES THE FIRST LAYER OF COMPUTATIONS IN THE NETWORK
		layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
		# RUNS ACTIVATION FUNCTION 'RECTIFIED LINEAR'
		layer_1 = tf.nn.relu(layer_1) 

		layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
		layer_2 = tf.nn.relu(layer_2) 

		layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['biases'])
		layer_3 = tf.nn.relu(layer_3) 

		output = tf.add(tf.matmul(layer_3, output_layer['weights']), output_layer['biases'])

		return output

	def train_neural_network(self):
		input_data = self.x
		# GENERATES A PREDICTION FROM 'input_data'
		prediction = self.neural_network_model(input_data)

		# 'reduce_mean' RETURNS THE AVERAGE OF AN ARRAY
		# RUNS THE 'COST' FUNCTION 'softmax_cross_entropy_with_logits' TO CALCULATE DISTANCE
		# BETWEEN PREDICTION 'prediction' AND KNOWN LABEL 'y'
		cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=prediction) )

		# 'AdamOptimizer()' RUNS TO MINIMIZE THE COST (DISTANCE BETWEEN PREDICTION AND ACTUAL)
		optimizer = tf.train.AdamOptimizer().minimize(cost) # 'AdamOptimizer()' DEFAULTS WITH 'learning_rate' == 0.001

		# NUMBER OF 'FEED-FORWARD + BACKPROPOGATION' CYCLES
		n_epochs = 10

		with tf.Session() as sesh:
			# BOILER PLATE
			sesh.run(tf.global_variables_initializer())

			#### TRAINS DATA ####

			for epoch in range(n_epochs):
				epoch_loss = 0
				# TOTAL NUMBER OF SAMPLES DIVIDED BY BATCH SIZE YEILD HOW MANY CYCLES ARE NEEDED
				# FOR EACH CYCLE
				for _ in range(int(self.mnist.train.num_examples / self.batch_size)):
					# TRAINS MODEL IN BATCHES
					epoch_x, epoch_y = self.mnist.train.next_batch(self.batch_size)
					# OPTIMIZES AND RETURNS THE COST
					_, c = sesh.run([optimizer, cost], feed_dict = {self.x: epoch_x, self.y: epoch_y})
					# CALCULATES TOTAL ACCUMULATED COST OVER ALL THE EPOCHS
					epoch_loss += c
				print('Epoch', epoch, 'complete out of', n_epochs, 'w/ total loss:', epoch_loss)

			#### RUNS THROUGH MODEL ####

			# DETERMINES IF THE PREDICTION IS EQUAL TO THE ACTUAL ANSWER
			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))

			# GRABS THE AVERAGE NUMBER OF CORRECT
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			acc_eval = accuracy.eval({self.x: self.mnist.test.images, self.y: self.mnist.test.labels})
			print('Accuracy:', acc_eval)

if INPUT == 'mnist':
	clf = MNIST_Network()
	clf.train_neural_network()

#																	 #
#						  POS/NEG SENTIMENT (10k)	 			 	 #
#			   		   (PREPARING DATA FOR NETWORK)					 #

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy, random, pickle
from collections import Counter

class Sentiment_Setup:

	def __init__(self, lines=100000):
		# LEMMATIZING BRINGS WORDS INTO THEIR ROOT FORM -- 'CATS' TO CAT
		self.lemmatizer = WordNetLemmatizer()
		# DEFINES THE NUMBER OF MAX NUMBER OF LINES TO READ PER FILE
		self.n_lines = lines

	def create_lexicon(self, files):
		# LIST OF ALL UNIQUE RELAVENT WORDS FOUND IN EACH FILE
		lexicon = []
		for file in files:
			with open(file, 'r') as f:
				contents = f.readlines()
				for line in contents[:self.n_lines]:
					# BREAKS THE LINE UP INTO A LIST OF THE INDIVIDUAL WORDS, 
					# THEN ADDS THEM TO THE LEXICON
					all_words = word_tokenize(line.lower())
					lexicon += list(all_words)

		# LEMMATIZES EACH WORD IN THE LEXICON
		lexicon = [self.lemmatizer.lemmatize(i) for i in lexicon]

		# RETURNS A DICTIONARY WITH EACH WORD AND ITS FREQUENCY IN THE LIST
		w_counts = Counter(lexicon)
		lexicon_2 = []

		for w in w_counts:
			# FILTERS OUT WORDS THAT ARE TOO COMMON AND TOO RARE
			if 1000 > w_counts[w] > 50:
				lexicon_2.append(w)

		print('Lexicon is {} words'.format(len(lexicon_2)))
		return lexicon_2


	def sample_handling(self, sample, lexicon, classification):
		feature_set = []

		# OPENS AND READS EACH DATASET -- TXT FILE
		with open(sample, 'r') as file:
			contents = file.readlines()

			# BREAKS DOWN THE FILE INTO ITS INDIVIDUAL WORDS PER LINE
			for line in contents[:self.n_lines]:
				current_words = word_tokenize(line.lower())
				current_words = [self.lemmatizer.lemmatize(i) for i in current_words]

				# POPULATES AN ARRAY FILLED W/ 0's TO BE USED IN THE BINARY-HOT SYSTEM
				# [0,0,0,0,0,0,0,0,0,0,0,0]
				features = numpy.zeros(len(lexicon))

				# FOR EACH WORD -- IF ITS IN THE LEXICON -- TURN 'HOT' ITS INDEX IN THE 'features' LIST
				for word in current_words:
					if word.lower() in lexicon:
						index_value = lexicon.index(word.lower())
						features[index_value] += 1

				# TURNS 'features' FROM A NUMPY ARRAY TO A LIST		
				features = list(features)

				# POPULATES 'feature_set' W/ ALL THE INDIVIDUAL FEATURE LISTS AND CLASSIFICATIONS
				feature_set.append([features, classification])

		return feature_set

	def create_featuresets_and_labels(self, files, test_size=0.1):
		# GENERATES LEXICON
		lexicon = self.create_lexicon(files)
		features = []
		# POPULATES 'features' W/ BOTH FEATURE SETS
		features += self.sample_handling(files[0], lexicon, [1,0])
		features += self.sample_handling(files[1], lexicon, [0,1])

		# ALWAYS WANT SHUFFLE THE DATA BEFORE RUNNING IT THROUGH NETWORK
		random.shuffle(features)

		features = numpy.array(features)
		testing_size = int(test_size * len(features))

		# GENERATES TRAINING AND TESTING SETS
		# 'features' IN FORMAT [[features, label], [features, label], ... ] 
		train_x = list(features[:,0][:-testing_size]) # '[:,0]' -- numpy notation that grabs all the 0-ith elements
									  # 			 				from each list in a list of lists
									  # 				'[:-testing_size]' -- numpy notation that grabs all but the 
									  # 									last 10% of values from array
		train_y = list(features[:,1][:-testing_size])

		test_x = list(features[:,0][-testing_size:])
		test_y = list(features[:,1][-testing_size:])

		return train_x, train_y, test_x, test_y


if INPUT == 'sentprep 10k':
	train_x, train_y, test_x, test_y = Sentiment_Setup().create_featuresets_and_labels(['pos.txt', 'neg.txt'])
	with open('sentiment_set.pickle', 'wb') as file:
		pickle.dump([train_x, train_y, test_x, test_y] , file)
	print('Completed Preparations')


#																	 #
#						  POS/NEG SENTIMENT (1.6m)	 			 	 #
#			   		   (PREPARING DATA FOR NETWORK)					 #

class Sent_Prep_2:

	def __init__(self, train_file, test_file):
		self.train_file = train_file 
		self.train_set = 'train_set.csv'
		self.test_file = test_file
		self.test_set = 'test_set.csv'

		for _file_ in [train_file, test_file]:

			if file == train_file:
				outfile = open(self.train_set, 'a')
			elif file == test_file:
				outfile = open(self.test_set, 'a')

		    with open(_file_, buffering=200000, encoding='latin-1') as file:

		        try:
		            for line in file:
		                line = line.replace('"','')
		                initial_polarity = line.split(',')[0]
		                if initial_polarity == '0':
		                    initial_polarity = [1,0]
		                elif initial_polarity == '4':
		                    initial_polarity = [0,1]

		                tweet = line.split(',')[-1]
		                outline = str(initial_polarity) + ':::' + tweet
		                outfile.write(outline)

		        except Exception as e:
		            print(str(e))

		    outfile.close()

	def create_lexicon(self):
		lexicon = []

		with open(self.train_set, 'r', buffering=100000, encoding='latin-1') as file:

			try:
				counter = 1
				content = ''

				for line in file:
					counter += 1

					if (counter/2500.0).is_integer():
						tweet = line.split(':::')[1]
						content += ' '+tweet
						words = word_tokenize(content)
						words = [lemmatizer.lemmatize(i) for i in words]
						lexicon += list(set(words))
						print(counter, len(lexicon))

			except Exception as e:
				print(str(e))

		self.lexicon = lexicon

	def convert_to_vec(self):

		self.processed_test_file = 'processed_test_file.csv'

		outfile = open(self.processed_test_file, 'a')

		with open(self.test_set, buffering=20000, encoding='latin-1') as file:
			counter = 0

			for line in file:
				counter +=1
				label = line.split(':::')[0]
				tweet = line.split(':::')[1]
				current_words = word_tokenize(tweet.lower())
				current_words = [lemmatizer.lemmatize(i) for i in current_words]

				features = np.zeros(len(self.lexicon))

				for word in current_words:
					if word.lower() in self.lexicon:
						index_value = self.lexicon.index(word.lower())
						# OR DO +=1, test both
						features[index_value] += 1

				features = list(features)
				outline = str(features)+ '::' + str(label) + '\n'
				outfile.write(outline)

			print('Converted {} lines to vectors.'.format(counter))

	def shuffle_data(self):
		df = pd.read_csv(self.train_set, error_bad_lines=False)
		df = df.iloc[np.random.permutation(len(df))]
		df.to_csv('train_set_shuffled.csv', index=False)

	def create_test_data_pickle(self):

		feature_sets = []
		labels = []
		counter = 0
		with open(self.processed_test_file, buffering=20000) as f:
			for line in f:
				try:
					features = list(eval(line.split('::')[0]))
					label = list(eval(line.split('::')[1]))

					feature_sets.append(features)
					labels.append(label)
					counter += 1
				except:
					pass
		print(counter)
		feature_sets = np.array(feature_sets)
		labels = np.array(labels)

	def get_House_in_Order(self):
		self.create_lexicon()
		self.convert_to_vec()
		self.shuffle_data()
		self.create_test_data_pickle()


#																	 #
#						   SENTIMENT MODEL 			  			 	 #
#																	 #

class Sentiment_Network:

	# ESTABLISHES THE COMPUTATION GRAPH FOR THE MODEL
	def __init__(self, files):

		self.train_x, self.train_y, self.test_x, self.test_y = Sentiment_Setup().create_featuresets_and_labels(files)

		# DEFINE THE AMOUNT OF NODES PER 'HIDDEN LAYER'
		self.n_nodes_hl1 = 750
		self.n_nodes_hl2 = 750

		# DEFINE NUMBER OF CLASSES 
		self.n_classes = 2 # NUMBERS POSITIVE OR NEGATIVE 

		# FOR MEMORY AND PROCESSING PURPOSES, DEFINE 'batch_size' WHEN DEALING WITH V LARGE DATASETS
		# RUNS BATCH THROUGH THE MODEL, MANIPULATES THE WEIGHTS, THEN DOES THE NEXT BATCH
		self.batch_size = 32
		self.total_batches = int(1600000 / batch_size)

		# PLACEHOLDERS ARE GOOD TO ENSURE INPUTS THROWN THROUGH THE MODEL ARE IN THE APPROPRIATE FORMAT
		# DEFINES A TYPE AND SHAPE FOR INPUT DATA TO BE
		self.x = tf.placeholder('float')
		self.y = tf.placeholder('float')


		self.saver = tf.train.Saver()

	def neural_network_model(self, data):

		# NEURON EQUATION: ''(INPUT DATA * WEIGHTS) + BIAS'' -- BIAS IS ADDED INCASE ALL INPUTS ARE EMPTY  

		# ASSIGNS RANDOM 'TENSORFLOW VARIABLES' AS WEIGHTS		  [MATRIX SIZE, NODES]
		hidden_layer_1 = { 'weights': tf.Variable(tf.random_normal([len(self.train_x[0]), self.n_nodes_hl1])),
						   'biases': tf.Variable(tf.random_normal([self.n_nodes_hl1])) }
		
		hidden_layer_2 = { 'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
						   'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2])) }
		
		output_layer = { 'weights': tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_classes])),
						   'biases': tf.Variable(tf.random_normal([self.n_classes])) }


		# CREATES THE FIRST LAYER OF COMPUTATIONS IN THE NETWORK
		layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
		# RUNS ACTIVATION FUNCTION 'RECTIFIED LINEAR'
		layer_1 = tf.nn.relu(layer_1) 

		layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
		layer_2 = tf.nn.relu(layer_2) 

		output = tf.add(tf.matmul(layer_2, output_layer['weights']), output_layer['biases'])

		return output

	def train_neural_network(self):
		# GENERATES A PREDICTION FROM input_data
		prediction = self.neural_network_model(self.x)

		# 'reduce_mean' RETURNS THE AVERAGE OF AN ARRAY
		# RUNS THE 'COST' FUNCTION 'softmax_cross_entropy_with_logits' TO CALCULATE DISTANCE
		# BETWEEN PREDICTION 'prediction' AND KNOWN LABEL 'y'
		cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=prediction) )

		# 'AdamOptimizer()' RUNS TO MINIMIZE THE COST (DISTANCE BETWEEN PREDICTION AND ACTUAL)
		optimizer = tf.train.AdamOptimizer().minimize(cost) # 'AdamOptimizer()' DEFAULTS WITH 'learning_rate' == 0.001

		# NUMBER OF 'FEED-FORWARD + BACKPROPOGATION' CYCLES
		n_epochs = 10

		with tf.Session() as sesh:
			# BOILER PLATE
			sesh.run(tf.global_variables_initializer())

			#### TRAINS DATA ####

			for epoch in range(n_epochs):
				epoch_loss = 0
				
				# CREATES CUSTOM BATCH TRAINING SYSTEM
				i = 0
				while i < len(self.train_x):
					start = i
					end  = i + self.batch_size
					# GENERATES BATCH X/Y OF SPEC BATCH SIZES
					batch_x = numpy.array(self.train_x[start:end])
					batch_y = numpy.array(self.train_y[start:end])

					# OPTIMIZES AND RETURNS THE COST
					_, c = sesh.run([optimizer, cost], feed_dict = {self.x: batch_x, self.y: batch_y})
					# CALCULATES TOTAL ACCUMULATED COST OVER ALL THE EPOCHS
					epoch_loss += c

					i += self.batch_size

				print('Epoch', epoch + 1, 'complete out of', n_epochs, 'w/ total loss:', epoch_loss)

			#### RUNS THROUGH MODEL ####

			# DETERMINES IF THE PREDICTION IS EQUAL TO THE ACTUAL ANSWER
			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))

			# GRABS THE AVERAGE NUMBER OF CORRECT
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			acc_eval = accuracy.eval({self.x: self.test_x, self.y: self.test_y})
			print('Accuracy:', acc_eval)

	def use_neural_network(self, input_data):
	    prediction = neural_network_model(self.x)
	    with open('lexicon.pickle','rb') as f:
	        lexicon = pickle.load(f)
	        
	    with tf.Session() as sess:
	        sess.run(tf.initialize_all_variables())
	        saver.restore(sess,"model.ckpt")
	        current_words = word_tokenize(input_data.lower())
	        current_words = [lemmatizer.lemmatize(i) for i in current_words]
	        features = np.zeros(len(lexicon))

	        for word in current_words:
	            if word.lower() in lexicon:
	                index_value = lexicon.index(word.lower())
	                # OR DO +=1, test both
	                features[index_value] += 1

	        features = np.array(list(features))
	        # pos: [1,0] , argmax: 0
	        # neg: [0,1] , argmax: 1sav
	        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
	        if result[0] == 0:
	            print('Positive:',input_data)
	        elif result[0] == 1:
	            print('Negative:',input_data)
if INPUT == 'sentiment':
	clf = Sentiment_Network(['pos.txt', 'neg.txt'])
	clf.train_neural_network()


