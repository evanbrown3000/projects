#EVAN	BROWN	JUNE	15
#Manually implement softmax batch grad desc with early stopping for educational purposes
#then compare to sklearn's optimized implementation

from random import sample
import copy
import numpy as np
import pandas as pd
from pandas import DataFrame
import numpy
from math import exp, log
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

#Preprocess and import data
digits = load_digits()
X, y = digits['data'], digits['target'] #for tablet only, remove secondary indexing for colab

print('converting X y to DataFrame, series, from type: ',type(X),type(y))
X, y = pd.DataFrame(X), pd.Series(y)
X, X_test, y, y_test = train_test_split(X,y, train_size = 0.8)
X_train, X_val, y_train, y_val = train_test_split(X,y, train_size = 0.8)

print('Train size:',X_train.shape, y_train.shape)
print('Validation size:',X_val.shape, y_val.shape)
print('Test size:',X_test.shape, y_test.shape)

#class manually implementing softmax batch grad desc with early stop
class LogisticRegression():
	theta_list = None
	eta, min_error, n_epochs, batch_sample_ratio = [np.nan] *4

#min_error used for early stopping with log loss fxn	
#max_error used for highest log_loss error val 
	def __init__(self, theta_list= None, eta=0.1,min_error = 0.000001, n_epochs = 50, batch_sample_ratio = 0.1,max_error = 100):
		self.theta_list = theta_list
		self.eta = eta
		self.min_error = min_error
		self.n_epochs = n_epochs
		self.batch_sample_ratio = batch_sample_ratio
		self.max_error = max_error
	def sum_logs(vec):
		sum = 0
		for i in vec:
			sum+= log(i)
		return sum		
					
	def sigma(t):
		try:
			return 1/ (1+exp(-t))
		except OverflowError: #exp(-t) too small (too many decimal places), simga approaches 1
			return 1

	#inplace
	def apply_sigma(vec):
		if(type(vec)== pd.Series):
			vec = vec.values
		i = 0
		while i < len(vec-1):
			vec[i] = LogisticRegression.sigma(vec[i])
			i+=1
		return vec
		
	def learning_schedule(t , t0 =5, t1 = 50):
		return t0 / (t + t1)

	def log_loss(self, p_hat, y):
		max_error = self.max_error
		min_z = 10**(-max_error) #max_error = -log(min_z)
		sum_logs = LogisticRegression.sum_logs
		error_total = 0
		n_obs = len(y)
		
		for i in range(n_obs):
			if y.iloc[i]==1:
				z = p_hat[i]
				if z < min_z: #prevent log 0
					error_total += max_error
				else:		
					error_total += -log(z) 
			else:				
				z = 1 - p_hat[i]
				if z < min_z:
					error_total += max_error
				else:
					error_total += -log(z)
		return -error_total / n_obs
	
#if X_val not given, must give iteration num to stop at, otherwise use a default
	def fit(self, X_train, X_val, y_train, y_val, class_list=None):
		THETA_INIT_VAL = 0
		
		if(class_list == None):
			class_list = y_val.unique() #gives categories		
		
		n_rows, n_cols = X_train.shape    
		n_features = n_cols
		self.theta_list = {class_name:[THETA_INIT_VAL]*n_features for class_name in class_list} #zero initialization,
		p_hat_list = copy.deepcopy(self.theta_list)
		
		min_error = self.min_error
		n_epochs = self.n_epochs
		
		batch_size = int(n_rows*self.batch_sample_ratio)
		apply_sigma = LogisticRegression.apply_sigma
		log_loss = self.log_loss
		learning_schedule = LogisticRegression.learning_schedule
	
		for epoch in range(n_epochs):
			minibatch_indices = sample(range(n_rows) , batch_size) 
			X_train_minibatch = X_train.values[minibatch_indices, :] #sample is without replace, choices is w/
			y_train_minibatch = y_train.values[minibatch_indices] 
			
			n_rows, n_cols = X_train_minibatch.shape
			for class_name, theta in self.theta_list.items():
				#cross entropy gradient vector for class k:
				# (1/m) * elementwise_mult( (p_hat_vec -  y_vec) , X_column_k )
				y_train_minibatch_k = y_train_minibatch == class_name #0 or 1 for whether y is of class k
				#calc gradients
				scores = X_train_minibatch.dot( theta)
				p_hats = apply_sigma(scores)
				errors = p_hats - y_train_minibatch_k
				
				sum_errors_times_x_i = np.zeros(n_cols)
				for i in range(n_rows-1):
					row_i = X_train_minibatch[i,:]
					errors_times_x_i = errors[i] * row_i
					sum_errors_times_x_i += errors_times_x_i
				avg_error_times_x_i = (1/n_rows)*(sum_errors_times_x_i)
				gradients = avg_error_times_x_i
				eta = learning_schedule(epoch) #reduce learning rate eta with learning_schedule fxn
				self.theta_list[class_name] = theta - eta * gradients #adjust theta 
	
				#early stopping:
				p_hat_list[class_name] = apply_sigma(X_val.dot(theta)) #calc new probability prediction vector for class k, using validation set
				if(-log_loss(p_hat_list[class_name],y_val) < min_error):
					return
		#done looping			
		return
			
	def predict_proba(self, X_test, class_list=None):
		n_rows = X_test.shape[0]
		score_list = {class_name: np.empty((n_rows,),dtype=float) for class_name in self.theta_list.keys() } #one score for each obs
		prob_list = copy.deepcopy(score_list)	
		for class_name in class_list.keys():
			theta_k = self.theta_list[class_name]
			scores = X_test.dot(theta_k)
			score_list[class_name] = scores
			
		for i in range(n_rows):
			sum_scores_i = 0
			score_list_i = {name : np.nan for name in score_list.keys()}	
			for class_name in score_list.keys():
				score_i_k = score_list[class_name].iloc[i] 
				sum_scores_i +=  score_i_k
			for class_name in score_list.keys():
				divisor = exp(sum_scores_i)
				if(divisor == 0):
					divisor =0.000001 
				prob_list[class_name][i] = exp(score_list[class_name].iloc[i])/divisor 
		return prob_list
	def predict(self, X_test, class_list=None):
		if(class_list ==None):
			class_list = self.theta_list
		
		n_rows=X_test.shape[0]
		prob_list = self.predict_proba(X_test, class_list) #get predictions for each class as dict of arrays
		
		#find highest predicted class for each obs
		predicted_class_vec = np.empty((n_rows,)) #one prediction per obs		
		i=0
		while i < (n_rows):
			
			max_class = None
			max_prob = -1						
			for k in class_list:
				
				prob_k_i =prob_list[k][i]				
				if(prob_k_i > max_prob):
					max_prob = prob_k_i
					max_class = k 
			predicted_class_vec[i] = max_class 
			i+=1	
		return predicted_class_vec


#TESTING	VS	SKLEARN~~~~~~~~~~~~~~
lr_model = LogisticRegression()

from timeit import default_timer as timer
start = timer()
lr_model.fit(X_train, X_val, y_train, y_val)
stop = timer()
print('fit time, manual implementation: ', stop-start)
#print('learned coefficients:', lr_model.theta_list)
predictions = lr_model.predict(X_test)

from sklearn.metrics import accuracy_score
#print('predictions:', predictions)
print('accuracy:', accuracy_score(predictions, y_test))

from sklearn.linear_model import LogisticRegression as SKLR
sklr = SKLR()
start = timer()
sklr.fit(X_train, y_train)
stop = timer()
print('fit time, sklearns optimized implementation: ', stop-start)
#print('learned coefficients:', sklr.coef_)
predictions_sklr = sklr.predict(X_test)
print('percent manual predictions agree with sklearn:', accuracy_score(predictions_sklr,predictions))
print('accuracy sklearn:', accuracy_score(predictions_sklr, y_test))