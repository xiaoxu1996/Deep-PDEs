import tensorflow as tf
import numpy as np
import time, datetime
import os

from tensorflow.python.training import moving_averages
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth = True


start_time = time.time()
tf.compat.v1.reset_default_graph()

name = 'BSB'

# setting of the problem
d = 100 # or d=256 or 400
T = 1.0

# setup of algorithm and implementation
N = 20
h = T/N
sqrth = np.sqrt(h)
# if d = 100
n_maxstep = 400
# if d = 256 or 400
# n_maxstep = 1000
n_dispalystep = 100
batch_size = 64
Xinit = np.array([1.0,0.5]*50)
mu = 0
sigma = 0.4
sigma_min = 0.1
sigma_max = 0.4
r = 0.05
_extra_train_ops = []


# neural net architectures
# Multiscale fusion net parameters
nn1_ForGamma = [d, d, d, d**2]
nn1_ForA = [d, d, d, d]
nn2_ForGamma = [d, 75, 75, d**2]
nn2_ForA = [d, 75, 75, d]
nn3_ForGamma = [d, 125, 125, d**2]
nn3_ForA = [d, 125, 125, d]
nn4_ForGamma = [d, 50, 50, d**2]
nn4_ForA = [d, 50, 50, d]
nn_allGamma = [nn1_ForGamma, nn2_ForGamma, nn3_ForGamma, nn4_ForGamma]
nn_allA = [nn1_ForA, nn2_ForA, nn3_ForA, nn4_ForA]
# Convolutional nerual network
cnnForA = [32, 32, 1, d]
cnnForGamma = [32, 32, 1, d**2]

def sigma_value(W):
	return sigma_max*tf.cast(tf.greater_equal(W,tf.cast(0,tf.float64)),tf.float64)+sigma_min*tf.cast(
							tf.greater_equal(tf.cast(0,tf.float64),W),tf.float64)

def f_tf(t, X, Y, Z, Gamma):
	return -0.5*tf.expand_dims(tf.linalg.trace(tf.square(tf.expand_dims(X,-1))*(tf.square(sigma_value(Gamma))
								-sigma**2)*Gamma),-1)+r*(Y-tf.reduce_sum(X*Z,1,keepdims=True))

def g_tf(X):
	return tf.reduce_sum(tf.square(X),1,keepdims=True)

def sigma_function(X):
	return sigma * tf.matrix_diag(X)

def mu_function(X):
	return mu * X

# helper functions for constructing the neural net ( s )
# Multiscale fusion
def _lnn_time_net(x,name,isgamma=False):
	with tf.compat.v1.variable_scope(name):
		x_norm = _batch_norm(x,name='layer0_normalization')
		layer1 = _nn_time_net(x_norm,isgamma,name='layer1',number=1)
		layer2 = _nn_time_net(x_norm,isgamma,name='layer2',number=2)
		layer3 = _nn_time_net(x_norm,isgamma,name='layer3',number=3)
		layer4 = _nn_time_net(x_norm,isgamma,name='layer4',number=4)
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
	with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
		shape = input_.get_shape().as_list()
		w = tf.compat.v1.get_variable('Matrix',[shape[1],output_size],tf.float64,
							tf.random_normal_initializer(
								stddev=stddev/np.sqrt(shape[1]+output_size)))
		hidden = tf.matmul(input_,w)
		hidden_bn = _batch_norm(hidden, name='normalization')
		if activation_fn:
			return activation_fn(hidden)
		else:
			return hidden

def _batch_norm(x, name):
	with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
		params_shape = [x.get_shape()[-1]]
		beta = tf.compat.v1.get_variable('beta',params_shape,tf.float64,
								initializer=tf.random_normal_initializer(0.0,stddev=0.1))
		gamma = tf.compat.v1.get_variable('gamma',params_shape,tf.float64,
								initializer=tf.random_uniform_initializer(0.1, 0.5))
		moving_mean = tf.compat.v1.get_variable('moving_mean',params_shape, tf.float64,
									initializer=tf.constant_initializer(0.0),trainable=False)
		moving_variance = tf.compat.v1.get_variable('moving_variance', params_shape, tf.float64,
									initializer=tf.constant_initializer(1.0),trainable=False)
		mean, variance = tf.nn.moments(x,[0],name='moments')
		_extra_train_ops.append(moving_averages.assign_moving_average(moving_mean,mean,0.99))
		y = tf.nn.batch_normalization(x,mean,variance,beta,gamma,1e-6)
		y.set_shape(x.get_shape())
		return y

def _cnn_time_net(x,name,isgamma=False):
	with tf.compat.v1.variable_scope(name):
		input_x = tf.reshape(x,[64,20,20,1])
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
		input_x = tf.reshape(input_, [64,-1])
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

with tf.compat.v1.Session(config=config) as sess:
	# background dynamics
	dW = tf.random.normal(shape=[batch_size,d],stddev=1,dtype=tf.float64)
	# initial values of the stochastic processes
	X = tf.Variable(np.ones([batch_size,d]) * Xinit,dtype=tf.float64,name='X',trainable = False)
	Y0 = tf.Variable(tf.random.uniform([1],minval=0,maxval=1,dtype=tf.float64),name='Y0')
	Z0 = tf.Variable(tf.random.uniform([1,d],minval=-.1,maxval=.1,dtype=tf.float64),name='Z0')
	Gamma0 = tf.Variable(tf.random.uniform([d,d],minval=-1,maxval=1,dtype=tf.float64),name='Gamma0')
	A0 = tf.Variable(tf.random.uniform([1,d],minval=-.1,maxval=.1,dtype=tf.float64),name='A0')
	allones = tf.ones(shape=[batch_size,1],dtype=tf.float64,name='MatrixOfOnes')
	Y = allones * Y0
	Z = tf.matmul(allones,Z0)
	Gamma = tf.multiply(tf.ones([batch_size, d, d],dtype=tf.float64),Gamma0)
	A = tf.matmul(allones, A0)

	# forward discretization
	with tf.compat.v1.variable_scope('forward'):
		for t in range(0, N-1):
			dX = mu * X * h + sqrth * sigma * X * dW
			Y = Y + f_tf(t*h,X,Y,Z,Gamma)*h + tf.reduce_sum(Z*dX,1,keepdims=True)
			X = X + dX
			# if d=100 
			A = _lnn_time_net(X, "A")/d
			Gamma = _lnn_time_net(X, "Gamma",isgamma=True)/d**2
			# if d=256 or 400
			# A = _cnn_time_net(X, 'A')/d
			# Gamma = _cnn_time_net(X,'Gamma',isgamma=True)/d**2
			Gamma = tf.reshape(Gamma, [batch_size, d, d])
			dW = tf.random.normal(shape=[batch_size,d],stddev=1,dtype=tf.float64)
		dX = mu * X * h + sqrth * sigma * X * dW
		Y = Y + f_tf((N-1)*h,X,Y,Z,Gamma)*h + tf.reduce_sum(Z*dX,1,keepdims=True)
		X = X + dX
		loss_function = tf.reduce_mean(tf.square(Y-g_tf(X)))

	# specifying the optimizer
	global_step = tf.compat.v1.get_variable('global_step',[],initializer=tf.constant_initializer(0),
								trainable=False,dtype=tf.int32)

	# if d=256
	learning_rate = tf.compat.v1.train.exponential_decay(2.0,global_step,decay_steps=500, 
								decay_rate=0.5,staircase=True)
	# if d=100
	learning_rate = tf.compat.v1.train.exponential_decay(1.0,global_step,decay_steps=200, 
								decay_rate=0.5,staircase=True)

	trainable_variables = tf.compat.v1.trainable_variables()
	grads = tf.gradients(loss_function,trainable_variables)
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
	apply_op = optimizer.apply_gradients(zip(grads,trainable_variables),
										global_step=global_step,name='train_step')
	train_ops = [apply_op] + _extra_train_ops
	train_op = tf.group(*train_ops)

	with tf.control_dependencies([train_op]):
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
			error = np.absolute(y0_value - 308.4195)/308.4195
			learning_rates.append(currentLearningRate)
			losses.append(currentLoss)
			y0_values.append(y0_value)
			running_time.append(time.time()-start_time)
			steps.append(step)
			L1_error.append(error)

			if step % n_dispalystep == 0:
				print("step: ", step, "loss: ", currentLoss, "Y0: ", y0_value,
					"L1 error: ", error, "learning rate: ", currentLearningRate)


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

np.savetxt("./"+str(name)+"_d"+str(d)+"_"+datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+".csv",
			output, delimiter=",", header="step, loss function, Y0, L1 error, learning rate, running time")