import tensorflow as tf
import numpy as np
import time, datetime
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = ConfigProto()
config.allow_soft_placement=True
#config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True


start_time = time.time()

name = 'AllenCahn'

# setting of the problem
d = 400
T = 0.3
Xi = np.zeros([1,d])

# setup of algorithm and implementation
N = 20
h = T/N
sqrth = np.sqrt(h)
n_maxstep = 10000
batch_size = 1
gamma = 0.001

# neural net architectures
# n_neuronForGamma = [d, d, d, d**2]
# n_neuronForA = [d, d, d, d]

nn1_ForGamma = [d, d, d, d**2]
nn1_ForA = [d, d, d, d]
nn2_ForGamma = [d, 30, 30, d**2]
nn2_ForA = [d, 30, 30, d]
nn3_ForGamma = [d, 40, 40, d**2]
nn3_ForA = [d, 40, 40, d]
nn4_ForGamma = [d, 50, 50, d**2]
nn4_ForA = [d, 50, 50, d]
nn_allGamma = [nn1_ForGamma, nn2_ForGamma, nn3_ForGamma, nn4_ForGamma]
nn_allA = [nn1_ForA, nn2_ForA, nn3_ForA, nn4_ForA]
cnnForA = [32, 32, 1, d]
cnnForGamma = [32, 32, 1, d**2]

# ( adapted ) rhs of the pde
def f0(t,X,Y,Z,Gamma):
	return -Y + tf.pow(Y,3)

# terminal condition
def g(X):
	return 1/(1 + 0.2* tf.reduce_sum(tf.square(X),
							1,keepdims=True))*0.5

# helper functions for constructing the neural net ( s )


def _lnn_time_net(x,name,isgamma=False):
	with tf.compat.v1.variable_scope(name):
		layer1 = _nn_time_net(x,isgamma,name='layer1',number=1)
		layer2 = _nn_time_net(x,isgamma,name='layer2',number=2)
		layer3 = _nn_time_net(x,isgamma,name='layer3',number=3)
		layer4 = _nn_time_net(x,isgamma,name='layer4',number=4)
		z = 0.25 * (layer1 + layer2 + layer3 + layer4)

	return z

def _nn_time_net(x,isgamma,name,number):
	n_neuronForGamma = nn_allGamma[number-1]
	n_neuronForA = nn_allA[number-1]
	with tf.compat.v1.variable_scope(name):
		layer1 = _one_layer(x,(1-isgamma )*n_neuronForA[1]+isgamma*n_neuronForGamma[1],name='nn_layer1')
		layer2 = _one_layer(layer1,(1-isgamma)*n_neuronForA[2]+isgamma*n_neuronForGamma[2],name='nn_layer2')
		z = _one_layer(layer2,(1-isgamma)*n_neuronForA[3]+isgamma*n_neuronForGamma[3],activation_fn=None,
																						name='nn_final')
	return z

def _one_layer(input_,output_size,activation_fn=tf.nn.relu,stddev=5.0,name='linear'):
	with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
		shape = input_.get_shape().as_list()
		w = tf.compat.v1.get_variable('Matrix',[shape[1],output_size],tf.float64,
							tf.random_normal_initializer(
								stddev=stddev/np.sqrt(shape[1]+output_size)))
		b = tf.compat.v1.get_variable('Bias',[1,output_size],tf.float64,
							tf.constant_initializer(0.0))
		hidden = tf.matmul(input_,w) + b
		if activation_fn:
			return activation_fn(hidden)
		else:
			return hidden

def _cnn_time_net(x,name,isgamma=False):
	with tf.compat.v1.variable_scope(name):
		input_x = tf.reshape(x,[1,16,16,1])
		layer1 = _cnn_layer(input_x,(1-isgamma)*cnnForA[0]+isgamma*cnnForGamma[0], name='layer1')
		layer2 = _cnn_layer(layer1,(1-isgamma)*cnnForA[1]+isgamma*cnnForGamma[1], name='layer2')
		layer3 = _cnn_layer(layer2,(1-isgamma)*cnnForA[2]+isgamma*cnnForGamma[2], name='layer3')
		z = _nn_layer(layer3,(1-isgamma)*cnnForA[3]+isgamma*cnnForGamma[3], activation_fn=None, name='final')

	return z

def _cnn_layer(input_, output_size, activation_fn=tf.nn.relu, name='conv'):
	with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
		shape = input_.get_shape().as_list()
		w_conv = tf.compat.v1.get_variable('Matrix', [3,3,shape[3],output_size], tf.float64, 
											tf.random_normal_initializer())
		b_conv = tf.compat.v1.get_variable('Bias', [output_size], tf.float64,
											tf.constant_initializer(0.0))
		hidden = tf.nn.conv2d(input_, w_conv, strides=[1,1,1,1], padding='SAME') + b_conv
		if activation_fn:
			return activation_fn(hidden)
		else:
			return hidden

def _nn_layer(input_, output_size, activation_fn=tf.nn.relu, stddev=5.0, name='linear'):
	with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
		input_x = tf.reshape(input_, [1,-1])
		w = tf.compat.v1.get_variable('Matrix',[d,output_size],tf.float64,
							tf.random_normal_initializer(
								stddev=stddev/np.sqrt(d+output_size)))
		b = tf.compat.v1.get_variable('Bias',[1,output_size],tf.float64,
							tf.constant_initializer(0.0))
		hidden = tf.matmul(input_x,w) + b
		if activation_fn:
			return activation_fn(hidden)
		else:
			return hidden

# with tf.compat.v1.Session() as sess:
# 	dW = tf.random.normal(shape=[batch_size,d],stddev=sqrth,dtype=tf.float64)
# 	y = _cnn_time_net(dW, name='dW')
# 	print(y.shape)


with tf.compat.v1.Session(config=config) as sess:
	# background dynamics
	dW = tf.random.normal(shape=[batch_size,d],stddev=sqrth,dtype=tf.float64)
	# initial values of the stochastic processes
	X = tf.Variable(np.ones([batch_size,d]) * Xi,dtype=tf.float64,name='X',trainable = False)
	Y0 = tf.Variable(tf.random.uniform([1],minval=-1,maxval=1,dtype=tf.float64),name='Y0')
	Z0 = tf.Variable(tf.random.uniform([1,d],minval=-.1,maxval=.1,dtype=tf.float64),name='Z0')
	Gamma0 = tf.Variable(tf.random.uniform([d,d],minval=-.1,maxval=.1,dtype=tf.float64),name='Gamma0')
	A0 = tf.Variable(tf.random.uniform([1,d],minval=-.1,maxval=.1,dtype=tf.float64),name='A0')
	allones = tf.ones(shape=[batch_size,1],dtype=tf.float64,name='MatrixOfOnes')
	Y = allones * Y0
	Z = tf.matmul(allones,Z0)
	Gamma = tf.multiply(tf.ones([batch_size, d, d],dtype=tf.float64),Gamma0)
	A = tf.matmul(allones, A0)

	# forward discretization
	with tf.compat.v1.variable_scope('forward'):
		for i in range(N-1):
			Y = Y + f0(i*h,X,Y,Z,Gamma)*h + tf.reduce_sum(dW*Z,1,keepdims=True)
			Gamma = tf.reshape(_lnn_time_net(X,'Gamma',isgamma=True)/d**2,[batch_size,d,d])
			if i != N-1:
				A = _lnn_time_net(X, 'A')/d
			X = X + dW
			dW = tf.random.normal(shape=[batch_size,d],stddev=sqrth,dtype=tf.float64)

		Y = Y + f0((N-1)*h,X,Y,Z,Gamma)*h + tf.reduce_sum(dW*Z,1,keepdims=True)
		X = X + dW
		loss_function = tf.reduce_mean(tf.square(Y-g(X)))

	# specifying the optimizer
	global_step = tf.compat.v1.get_variable('global_step',[],initializer=tf.constant_initializer(0),
								trainable=False,dtype=tf.int32)

	learning_rate = tf.compat.v1.train.exponential_decay(gamma,global_step,decay_steps=20000, 
								decay_rate=0.5,staircase=True)

	trainable_variables = tf.compat.v1.trainable_variables()
	grads = tf.gradients(loss_function,trainable_variables)
	optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
	apply_op = optimizer.apply_gradients(zip(grads,trainable_variables),
										global_step=global_step,name='train_step')

	with tf.control_dependencies([apply_op]):
		train_op_2 = tf.identity(loss_function,name='train_op2')

	# to save history
	learning_rates = []
	y0_values = []
	losses = []
	running_time = []
	steps = []
	L1_error = []
	sess.run(tf.compat.v1.global_variables_initializer())

	try :
		# the actual training loop
		for _ in range(n_maxstep + 1):
			y0_value, step = sess.run([Y0, global_step])
			currentLoss, currentLearningRate = sess.run([train_op_2,learning_rate])
			error = np.absolute(y0_value - 0.02711)/0.02711
			learning_rates.append(currentLearningRate)
			losses.append(currentLoss)
			y0_values.append(y0_value)
			running_time.append(time.time()-start_time)
			steps.append(step)
			L1_error.append(error)

			if step % 1000 == 0:
				# error = np.absolute(y0_value - 0.027106)/0.027106
				print("step: ", step, "loss: ", currentLoss, "Y0: ", y0_value,
					"L1 error: ", error, "learning rate: ", currentLearningRate)
				# learning_rates.append(currentLearningRate)
				# losses.append(currentLoss)
				# y0_values.append(y0_value)
				# running_time.append(time.time()-start_time)
				# steps.append(step)
				# L1_error.append(error)
				

		end_time = time.time()
		print("running time: ", end_time-start_time)

	except KeyboardInterrupt:
		print("\nmanually disengaged")

# writing results to a csv file
output = np.zeros((len(y0_values),6))
output[: ,0] = steps
output[: ,1] = losses
output[: ,2] = y0_values
output[:, 3] = L1_error
output[: ,4] = learning_rates
output[: ,5] = running_time


np.savetxt("plot/"+str(name)+"_d"+str(d)+"_"+datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+".csv",
			output, delimiter=",", header="step, loss function, Y0, L1 error, learning rate, running time")