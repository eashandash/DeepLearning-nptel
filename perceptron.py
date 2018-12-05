import numpy as np
import csv

X = []

with open('foo.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	# next(readCSV)  # Skip header line
	for row in readCSV:
		a = list(map(float, row))
		X.append(a)


class Perceptron():
	"""docstring for Perceptron"""
	def __init__(self,num_inputs):
		super(Perceptron, self).__init__()
		
		self.num_inputs = num_inputs
		self.weights = np.random.rand(self.num_inputs)
		self.bias = np.random.rand(1)[0]
		
		self.batch_grad_w = np.zeros(self.num_inputs)
		self.batch_grad_b = 0.0
		self.batch_count = 0
		self.batch_size = 1

		self.learning_rate = 0.5

	def feedforward(self,inputs):

		output = np.sum(self.weights*inputs)+self.bias
		output = self.sigmoid(output)

		return output

	def gradient_wrt_w(self,p_grad,inputs):

		return p_grad*inputs;

	def gradient_wrt_b(self,p_grad):

		return p_grad;

	def backprop(self,p_grad,inputs):

		if self.batch_count%self.batch_size == 0 and self.batch_size != 1:
			
			self.weights = self.weights - self.learning_rate*self.batch_grad_w
			self.bias = self.bias -  self.learning_rate*self.batch_grad_b
			
			self.batch_grad_w = np.zeros(self.num_inputs)
			self.batch_grad_b = 0.0

		elif self.batch_size == 1:
			self.weights = self.weights - self.learning_rate*np.asarray(self.gradient_wrt_w(p_grad,inputs))
			self.bias = self.bias -  self.learning_rate*self.gradient_wrt_b(p_grad)

		else:
			self.batch_grad_w += np.asarray(self.gradient_wrt_w(p_grad,inputs))
			self.batch_grad_b += self.gradient_wrt_b(p_grad)

		self.batch_count += 1


	def sigmoid(self,inputs):

		return 1.0/(1.0 + np.exp(-inputs))

	def sigmoid_backprop(self,inputs,p_grad):
		
		pre_output = self.sigmoid((np.sum(self.weights*inputs)+self.bias))

		self.backprop( p_grad*pre_output*(1-pre_output) , inputs )

	
	def error_backprop(self,inputs,outputs,mode="quiet"):

		predicted = self.feedforward(inputs)


		error_b = (predicted-outputs)

		self.sigmoid_backprop(inputs,error_b)

		if mode=="verbose":
			print(error_b)

	def set_learning_rate(self,l_rate):

		self.learning_rate = l_rate

	def print_params(self):
		print("Printing Params:")
		print(self.weights)
		print(self.bias)

	def error(self,inputs,outputs):

		predicted = self.feedforward(inputs)

		return 0.5*((predicted - outputs)**2)




if __name__ == "__main__":

	def trail():

		p1 = Perceptron(2)

		epochs = 200000

		p1.set_learning_rate(0.1)

		for e in range(epochs):
			for [x,y,o] in X:
				p1.error_backprop(np.asarray([x,y]),o,mode="quiet")


		# epochs = 10000
		# p1.set_learning_rate(0.01)

		# for e in range(epochs):
		# 	for [x,y,o] in X:
		# 		p1.error_backprop(np.asarray([x,y]),o,mode="quiet")



		print("Total Error:")
		err = 0
		for [x,y,o] in X:
			err += p1.error(np.asarray([x,y]),o)
		print(err)

		print("test")

		for [x,y,o] in X:
			print(p1.feedforward(np.asarray([x,y])),o)


	trail()


