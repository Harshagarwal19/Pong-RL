import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import cPickle as pickle
import scipy.misc
from collections import deque
import random
import deepQNetwork as DQN
from PIL import Image

# PARAMETERS
IMG_SIZE = 84
INP_FRAMES = 4

ACTIONS = 6

REPLAY_MEMORY = deque()	# store (4 Images, action, reward, Q_)

REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 10000
MINIBATCH_SIZE = 32

EPSILON = 1

COUNT = 0
GAME_COUNT = 1

# Statistics
TOTAL_REWARD = 0
AVG_REWARD = 0

GAMMA = 0.99

# ----- Followed documentation here : https://gym.openai.com/docs ------------------- #

# Helper functions
def getMultipleImageFrames(env, action):
	totalReward = 0

	env.render()
	inp, reward, done, info = env.step(action)

	inputImages = changeImage(inp)
	totalReward += reward

	for i in range(INP_FRAMES-1):
		env.render()
		inp, reward, done, info = env.step(action)

		# get changed image
		changedImage = changeImage(inp)
		totalReward += reward

		inputImages = np.dstack((inputImages, changedImage))

	# inputImages = inputImages.reshape((1, IMG_SIZE, IMG_SIZE, INP_FRAMES))	

	return inputImages, totalReward, done

# ------------------------------------------------------------------------------- #

def changeImage(img):	# resize the image
	img = grayScaleImage(img)
	# img = scipy.misc.imresize(img, (IMG_SIZE,IMG_SIZE))
	img = scipy.misc.imresize(img, (110,84))
	img = img[20:104,:]
	return img

# ----- Used this answer : http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python ------------------- #
def grayScaleImage(rgbImage):
	# return np.dot(rgbImage[...,:3], [0.299, 0.587, 0.114])
	img = Image.fromarray(rgbImage)
	img = img.convert('L')
	return img
# ------------------------------------------------------------------------------- #

# ----- Followed this answer : http://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
def showFrames(inputImages):
	for i in range(INP_FRAMES):
		data = inputImages[:,:,i]
		img = Image.fromarray(data)
		img.show()
# ------------------------------------------------------------------------------- #

def getQ(sess, inputImage, inputImages, Q):
	inputImages = inputImages.reshape((1, IMG_SIZE, IMG_SIZE, INP_FRAMES))
	return Q.eval(feed_dict={inputImage:inputImages})

def backPropagate(sess, inputImage, action, y_, train_step, batch):
	# append in list to fill placeholder
	inpImg = []
	actions = []
	y_s = [] 
	for i in batch:
		inpImg.append(i[0])
		actions.append(i[1])
		y_s.append(i[2])
		
	train_step.run(feed_dict={inputImage:inpImg, action:actions, y_:y_s})	
	
# Declare session
# ----- Followed documentation here : https://www.tensorflow.org/get_started/mnist/pros ---------- #
# ----- Followed documentation here : https://www.tensorflow.org/programmers_guide/variables --- #
sess = tf.InteractiveSession()

# Get DQN
inputImagePlaceholder, actionPlaceholder, y_, Q, train_step = DQN.dqn(sess)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# ------------------------------------------------------------------------------- #

# Load old network
# ----- Followed documentation here : https://www.tensorflow.org/programmers_guide/variables --- #
checkpoint = tf.train.get_checkpoint_state("./networks")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")
# ------------------------------------------------------------------------------- #

# Declare game environment
# ----- Followed documentation here : https://gym.openai.com/docs ------------------- #
env = gym.make('Pong-v0')
observation = env.reset()
# ------------------------------------------------------------------------------- #

# ----- Followed documentation here : https://gym.openai.com/docs ------------------- #
action = env.action_space.sample()
inputImages_t, reward_t, done = getMultipleImageFrames(env, action)	
Q_val= getQ(sess, inputImagePlaceholder, inputImages_t, Q)	# Q_t

TOTAL_REWARD = reward_t

while True:

	COUNT += 1
	
	# decide action based on EPSILON
	if random.random()>EPSILON:
		# Exploit
		# action = tf.argmax(Q_val,1).eval()[0]
		action = np.argmax(Q_val)
	else:
		# Explore
		action = env.action_space.sample()

	# take action
	inputImages_t1, reward_t1, done = getMultipleImageFrames(env, action)	
	Q_val1 = getQ(sess, inputImagePlaceholder, inputImages_t1, Q)	# Q_t+1
	# if COUNT==20:
	# 	showFrames(inputImages_t1)
	# 	break
	# Store in replay memory
	actionMatrix = np.zeros(ACTIONS)
	actionMatrix[action] = 1
	if done:	# for last state, no gamma*greedy(Q_t+1)
		v_s_t1 = reward_t1
	else:
		v_s_t1 = reward_t1 + GAMMA * np.max(Q_val1)
	
	REPLAY_MEMORY.append((inputImages_t, actionMatrix, v_s_t1))

	# use inputImages_t1 as inputimages_t2
	inputImages_t = inputImages_t1
	Q_val = Q_val1

	# Backpropagate
	if len(REPLAY_MEMORY)>=BATCH_SIZE:
		batch = random.sample(REPLAY_MEMORY, MINIBATCH_SIZE)
		backPropagate(sess, inputImagePlaceholder, actionPlaceholder, y_, train_step, batch)

	# Keep track of total reward for this game
	TOTAL_REWARD += reward_t1	

	if EPSILON>0.1 and COUNT%10000==0:	# reduce epsilon to have more exploitation over time
		EPSILON -= 0.01

	# check REPLAY_MEMORY size (shouldn't be more than batch_size)
	if len(REPLAY_MEMORY)>REPLAY_MEMORY_SIZE:
		REPLAY_MEMORY.popleft()	

	# save the network
	# ----- Followed documentation here : https://www.tensorflow.org/programmers_guide/variables --- #
	if COUNT%10000==0: 
		save_path = saver.save(sess, "./networks/dqn.ckpt")
		print("Model saved in file: %s" % save_path)
	# ------------------------------------------------------------------------------- #

	if done:
		AVG_REWARD = (((GAME_COUNT-1)*AVG_REWARD) + TOTAL_REWARD)/GAME_COUNT
		print "Game : ", GAME_COUNT , " COUNT = ", COUNT ," Total reward = ", TOTAL_REWARD, " Average reward = ", AVG_REWARD
		TOTAL_REWARD = 0
		GAME_COUNT += 1
		env.reset()	

		



















