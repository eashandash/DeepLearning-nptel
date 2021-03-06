import numpy as np
import csv
import timeit
# from multiprocessing.pool import ThreadPool
# pool = ThreadPool(processes=1)



X = []



with open('foo.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	# next(readCSV)  # Skip header line
	for row in readCSV:
		a = list(map(float, row))
		X.append(a)


class Perceptron():
	"""docstring for Perceptron"""
	def __init__(self,num_inputs,optimizer):
		super(Perceptron, self).__init__()
		
		self.num_inputs = num_inputs
		self.weights = np.random.rand(self.num_inputs)
		self.bias = np.random.rand(1)[0]

		if optimizer == "GradientDescent":
			self.optimizer = 0 # gradient descent
			self.learning_rate = 0.5
		
		elif optimizer == "Adam":
			self.optimizer = 1 # Adam
			self.beta1 = 0.9
			self.beta2 = 0.999
			self.epsilon = 1e-5
			self.m = np.asarray([0.0]*self.num_inputs)
			self.v = np.asarray([0.0]*self.num_inputs)
			self.learning_rate = 0.001
			self.m_b = 0.0
			self.v_b = 0.0
		
		self.timestep = 1
		self.batch_grad_w = np.zeros(self.num_inputs)
		self.batch_grad_b = 0.0
		self.batch_count = 0
		self.batch_size = 50

		

	def feedforward(self,inputs):

		output = np.sum(self.weights*inputs)+self.bias
		output = self.sigmoid(output)

		return output

	def gradient_wrt_w(self,p_grad,inputs):

		return p_grad*inputs;

	def gradient_wrt_b(self,p_grad):

		return p_grad;

	def backprop(self,inputs,p_grad):


		
		inner_work = self.sigmoid((np.sum(self.weights*np.asarray(inputs))+self.bias))
		
		grad_b = p_grad*inner_work*(1-inner_work)
		grad_w = grad_b*np.asarray(inputs)

		if self.optimizer == 1:
			self.m = (self.beta1 * self.m) + ((1-self.beta1)*grad_w)
			self.v = (self.beta2 * self.v) + ((1-self.beta2)*(grad_w**2))
			bc_m = (self.m)/(1-self.beta1**self.timestep)
			bc_v = (self.v)/(1-self.beta2**self.timestep)

			update_value_w = (bc_m)/(np.sqrt(bc_v)+self.epsilon)

			self.m_b = (self.beta1 * self.m_b) + ((1-self.beta1)*grad_b)
			self.v_b = (self.beta2 * self.v_b) + ((1-self.beta2)*(grad_b**2))
			bc_m = (self.m_b)/(1-self.beta1**self.timestep)
			bc_v = (self.v_b)/(1-self.beta2**self.timestep)
			update_value_b = (bc_m)/(np.sqrt(bc_v)+self.epsilon)

			

			if self.batch_count%self.batch_size == 0 and self.batch_size != 1:
			
				self.weights = self.weights - self.learning_rate*self.batch_grad_w
				self.bias = self.bias -  self.learning_rate*self.batch_grad_b
				
				self.batch_grad_w = np.zeros(self.num_inputs)
				self.batch_grad_b = 0.0

			elif self.batch_size == 1:
				self.weights = self.weights - self.learning_rate*update_value_w
				self.bias = self.bias -  self.learning_rate*update_value_b

			else:
				self.batch_grad_w += update_value_w
				self.batch_grad_b += update_value_b

		else:
			if self.batch_count%self.batch_size == 0 and self.batch_size != 1:
				
				self.weights = self.weights - self.learning_rate*self.batch_grad_w
				self.bias = self.bias -  self.learning_rate*self.batch_grad_b
				
				self.batch_grad_w = np.zeros(self.num_inputs)
				self.batch_grad_b = 0.0

			elif self.batch_size == 1:
				self.weights = self.weights - self.learning_rate*grad_w
				self.bias = self.bias -  self.learning_rate*grad_b

			else:
				self.batch_grad_w += grad_w
				self.batch_grad_b += grad_b

		self.batch_count += 1
		self.timestep+=1
		
		return p_grad*(self.weights)


	def sigmoid(self,inputs):

		return 1.0/(1.0 + np.exp(-inputs))


	def set_learning_rate(self,l_rate):

		self.learning_rate = l_rate

	def print_params(self):
		print("\tPrinting Unit Params:")
		print("\t",self.weights)
		print("\t",self.bias)

	def error(self,inputs,outputs):

		predicted = self.feedforward(inputs)

		return 0.5*((predicted - outputs)**2)





class Graph():

	def __init__(self,num_input,layer_sizes,optimizer):

		self.graph = [[Perceptron(num_input,optimizer) for _ in range(layer_sizes[0])]]

		for x in range(1,len(layer_sizes)):
			self.graph.append([Perceptron(layer_sizes[x-1],optimizer) for _ in range(layer_sizes[x])])
		

	def feedforward(self,inputs):

		input_holder = inputs
		layer_outputs = [input_holder]
		#  layer_outputs need to have input as index 0

		for layer in self.graph:
			layer_output = []
			for unit in layer:
				layer_output.append(unit.feedforward(input_holder))
			input_holder = np.asarray(layer_output)
			layer_outputs.append(layer_output)
			

		final_output= input_holder
		return final_output,layer_outputs


	def backprop(self,layer_outputs,expected):

		p_grad = [(layer_outputs[-1][0] - expected)]*len(self.graph[-1])
		for i in range(len(self.graph)):
			k = 0
			temp = np.asarray([0.0]*len(layer_outputs[len(self.graph)-i-1]))
			
			for unit in self.graph[len(self.graph)-i-1]:
				# async_result = pool.apply_async(unit.backprop,[layer_outputs[len(self.graph)-i-1],p_grad[k]])
				temp += unit.backprop(layer_outputs[len(self.graph)-i-1],p_grad[k])#async_result.get()
				k+=1
			p_grad = temp



	def print_params(self):

		print("Printing Graph Params")
		for layer in self.graph:
			for unit in layer:
				unit.print_params()


	def error(self,inputs,outputs):

		final_pred,l = self.feedforward(inputs)
		return 0.5*((final_pred - outputs)**2)

if __name__ == "__main__":

	start = timeit.default_timer()

	#  num_inputs , layer_sizes
	g = Graph(2,[8,16,16,16,8,8,4,1],"Adam")

	epochs = 4000

	# x = X[0][0]
	# y = X[0][1]
	# o = X[0][2]

	for e in range(epochs):
		for [x,y,o] in X:

			f,l = g.feedforward([x,y])
			g.backprop(l,o)

	stop = timeit.default_timer()

	print('Time: ', stop - start) 

	for [x,y,o] in X:
		f,l = g.feedforward([x,y])
		print(f,o)

	print("Total Error:")
	err = 0
	for [x,y,o] in X:
		err += g.error([x,y],o)
	print(err)
