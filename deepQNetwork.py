import tensorflow as tf
import numpy as np
import math

# ------------------ CNN ----------------------------------------- #

# Parameters
IMG_SIZE = 84
INP_FRAMES = 4

ACTIONS = 6

CONV1_FILTER = 8
CONV1_FEATURES = 16
CONV1_STRIDE = 4

CONV2_FILTER = 4
CONV2_FEATURES = 32
CONV2_STRIDE = 2

FC1_NODES = 256

GAMMA = 0.9

# -------------Used Tutorial code : https://www.tensorflow.org/get_started/mnist/pros ---------#

def get_weights(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def get_bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def convolution(x, W, s=1):  #   s -> strides
  return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

# ---------------------------------------------------------------------------------------- #

# ---- Followed CNN TF tutorial : https://www.tensorflow.org/get_started/mnist/pros --------------- #
# This is the same Deep network architecture described in Playing Atari with Deep Reinforcement Learning
# By Mnih et al., Deepmind, 2013
def dqn(sess):

	# Placeholders for variable (4 variables required for Deep Q network)
	inputImage = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, INP_FRAMES])
	action = tf.placeholder(tf.float32, shape=[None, ACTIONS])
	reward = tf.placeholder(tf.float32, shape=[None])
	Q_ = tf.placeholder(tf.float32, shape=[None, 1, ACTIONS])

	# convolutional layer 1
	conv1_weights = get_weights([CONV1_FILTER, CONV1_FILTER, INP_FRAMES, CONV1_FEATURES])
	conv1_bias = get_bias([CONV1_FEATURES])

	convLayer1 = convolution(inputImage, conv1_weights, CONV1_STRIDE) + conv1_bias
	hiddenLayer1 = tf.nn.relu(convLayer1)

	# convolutional layer 2
	conv2_weights = get_weights([CONV2_FILTER, CONV2_FILTER, CONV1_FEATURES, CONV2_FEATURES])
	conv2_bias = get_bias([CONV2_FEATURES])

	convLayer2 = convolution(hiddenLayer1, conv2_weights, CONV2_STRIDE) + conv2_bias
	hiddenLayer2 = tf.nn.relu(convLayer2)

	# Fully connected layer
	conv_layer_dimension = hiddenLayer2.shape

	fullyConnectedLayer1_weights = get_weights(
		[conv_layer_dimension[1].value * conv_layer_dimension[2].value * conv_layer_dimension[3].value, FC1_NODES])
	fullyConnectedLayer1_bias = get_bias([FC1_NODES])

	fullyConnectedLayer1_reshape = tf.reshape(hiddenLayer2, 
		[-1, conv_layer_dimension[1].value * conv_layer_dimension[2].value * conv_layer_dimension[3].value])
	fullyConnectedLayer1 = tf.matmul(fullyConnectedLayer1_reshape, fullyConnectedLayer1_weights) + fullyConnectedLayer1_bias
	hiddenLayer3 = tf.nn.relu(fullyConnectedLayer1)

	# Final output layer (Output is Q values)
	fc_layer_dimension = hiddenLayer3.shape

	Q_weights = get_weights([fc_layer_dimension[1].value, ACTIONS])
	Q_bias = get_bias([ACTIONS])

	Q = tf.matmul(hiddenLayer3, Q_weights) + Q_bias

	# Cost function
	# C = [Action_in_Q - {reward + gamma*Greedy(Q_)}]^2
	greedyQ_ = tf.reduce_max(Q_)
	discountQ_ = tf.multiply(greedyQ_, GAMMA)
	y_ = tf.add(discountQ_, reward)

	Q_action = tf.multiply(Q, action) # Only the Q-value for sampled action will have non-zero value
	Q_final = tf.reduce_sum(Q_action)

	cost = tf.square(Q_final - y_)

	# Backpropagate cost and adjust weights
	costTerm = tf.reduce_mean(cost)
	train_step = tf.train.AdamOptimizer(1e-6).minimize(costTerm)

	return inputImage, action, reward, Q_, Q, train_step
