import tensorflow as tf
import numpy as np
import gym

import DDPG

MAX_EPISODE = 200
MAX_STEP = 200
BATCH_SIZE = 32
MEMORY_SIZE = 10000
VAR_INIT = 3
RENDER = False

sess = tf.Session()
myDDPG = DDPG.DDPG(sess = sess, state_dimm = 3, action_dimm = 1, memory_size = MEMORY_SIZE, tau = 0.01, gamma = 0.9, \
	learning_rate_Q = 0.0005, learning_rate_P = 0.0005)
sess.run(tf.global_variables_initializer())

sess.run(myDDPG.hard_replace_P)
sess.run(myDDPG.hard_replace_Q)

env = gym.make('Pendulum-v0')
var = VAR_INIT
for episode in range(MAX_EPISODE):
	state = np.array(env.reset())
	ep_reward = 0
	for step in range(MAX_STEP):
		state2batch = np.hstack([state.flatten(), np.zeros(myDDPG.state_dim + \
			myDDPG.action_dim + 1)]).reshape(1, 2 * myDDPG.state_dim + myDDPG.action_dim + 1)

		action_ = myDDPG.get_action(state.reshape([1,3]))

		action = np.clip(np.random.normal(2*action_, var), -2, 2)


		new_state, reward, done, _ = env.step(action)

		mem_val = np.hstack([state.flatten(),action.flatten(),reward.flatten(),new_state.flatten()])

		myDDPG.store(mem_val)

		if episode * MAX_STEP + step + 1 > MEMORY_SIZE:
			var = var * 0.9995
			batch = myDDPG.get_batch(BATCH_SIZE)
			myDDPG.train_model(batch)
		if done:
			break
		ep_reward += reward
		state = new_state
		if RENDER:
			env.render()
	if (ep_reward > - 300):
		RENDER = True

	print(episode, ep_reward, var)