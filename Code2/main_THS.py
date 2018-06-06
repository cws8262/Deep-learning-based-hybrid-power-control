import tensorflow as tf
import numpy as np
import THS

import DDPG
import matplotlib
matplotlib.rcParams['backend'] = "Tkagg"
import matplotlib.pyplot as plt
import scipy.io as sio

n = 2
np.random.seed(n)
tf.set_random_seed(n)

MAX_EPISODE = 100
MAX_STEP = 200
BATCH_SIZE = 512
MEMORY_SIZE = 10000
VAR_INIT = 100

S_DIM = 7
A_DIM = 2


sess = tf.Session()
myDDPG = DDPG.DDPG(sess = sess, state_dimm = S_DIM, action_dimm = A_DIM, memory_size = MEMORY_SIZE, tau = 0.01, gamma = 0.9, \
	learning_rate_Q = 0.0005, learning_rate_P = 0.001)
sess.run(tf.global_variables_initializer())

sess.run(myDDPG.hard_replace_P)
sess.run(myDDPG.hard_replace_Q)

var = VAR_INIT

myTHS = THS.THS()

train_now = False


for episode in range(MAX_EPISODE):
	state, tend = myTHS.init(soc_init = 0.55, cyc_num = 0)
	state = np.array(state)
	ep_reward = 0
	state_saver = np.zeros([myTHS.tend.astype(int),S_DIM])
	action_saver = np.zeros([myTHS.tend.astype(int),2])
	action_saver_ = np.zeros([myTHS.tend.astype(int),2])
	for step in range(2*tend.astype(int)):
		
		[cyc_d, end_flag]=myTHS.feed_cyc()
		err_flag = True
		err_flag_chk = 0
		while err_flag:
			action_ = myDDPG.get_action(myTHS.SN(state).reshape([1,S_DIM]))

			action_Te = (action_[0,0] + 1) * 102.0000/2
			action_we = (action_[0,1] + 1) * 418.9000/2

			action = np.copy(action_)

			Te_min, Te_max = myTHS.calc_bound_Te()
			noiseTe = np.random.normal(0,var)
			action[0,0] = np.clip(action_Te + noiseTe, Te_min, Te_max) # Te

			we_min, we_max = myTHS.calc_bound_we(cyc_d, action[0,0])
			noise_we = np.random.normal(0,var)
			action[0,1] = np.clip(action_we + noise_we, we_min, we_max) # we
			# action[0,1] = np.clip(action[0,1], 0, 418.9000) # we
			# print(action)
			# print(cyc_d,action[0,0],action[0,1])
			[s_t_r, action, err_flag]=myTHS.calc_transition2(cyc_d,action[0,0],action[0,1])
			if err_flag:
				err_flag_chk += 1
				if err_flag_chk > 3:
					err_flag = True
					break

		state, action, reward, state_next, state_extra, state_extra_next = myTHS.transition(s_t_r, action, end_flag)

		action_Te_norm = ((action[0] / (102.0000/2)) - 1)
		action_we_norm = ((action[1] / (418.9000/2)) - 1)
		action_norm = np.array([[action_Te_norm, action_we_norm]])

		# print(state)
		if err_flag:
			reward = - 50 * myTHS.tend/myTHS.t

		mem_val = np.hstack([myTHS.SN(state.flatten()),action_norm.flatten(),reward.flatten(),myTHS.SN(state_next.flatten())])
		state_saver[step,:] = state
		action_saver[step,:] = action

		myDDPG.store(mem_val)
			
		ep_reward += reward
		state = state_next


		if end_flag:# or err_flag:
			break

		if episode * MAX_STEP + step + 1 > MEMORY_SIZE:
			batch = myDDPG.get_batch(BATCH_SIZE)
			myDDPG.train_model(batch)
			train_now = True
		

		if(train_now):
			var = var * 0.99999

	print(episode, ep_reward, var, step, train_now)
	# if train_now:
	if True:
		fig1 = plt.figure(1)
		plt.clf()
		plt.subplot(7,1,1)
		plt.plot(state_saver[:,0])
		plt.subplot(7,1,2)
		plt.plot(state_saver[:,1])
		plt.subplot(7,1,3)
		plt.plot(state_saver[:,2])
		plt.subplot(7,1,4)
		plt.plot(state_saver[:,3])
		plt.subplot(7,1,5)
		plt.plot(state_saver[:,4])
		plt.subplot(7,1,6)
		plt.plot(action_saver[:,0])
		plt.subplot(7,1,7)
		plt.plot(action_saver[:,1])
	


		plt.pause(0.001)
		plt.show(block = False)

sio.savemat('result.mat', {'state':state_saver, 'action':action_saver})