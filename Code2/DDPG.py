import tensorflow as tf
import numpy as np

class DDPG:
	def __init__(self,sess ,state_dimm, action_dimm, memory_size, tau, gamma, learning_rate_Q, learning_rate_P):


		self.sess = sess

		self.state_dim = state_dimm
		self.action_dim = action_dimm
		self.mem_size = memory_size

		self.P_scope_name = "ActorNet" #Policy or Action
		self.P_target_scope_name = "TargetActorNet" #Policy or Action
		self.Q_scope_name = "CriticNet"
		self.Q_target_scope_name = "TargetCriticNet"
 
		self.State_tensor = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='State')
		self.Action_tensor = tf.placeholder(tf.float32, shape=[None, self.action_dim], name='Action')
		self.Reward_tensor = tf.placeholder(tf.float32, shape=[None, 1], name='Reward')
		self.State1_tensor = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='State1')

		self.P = self.make_net_P(S = self.State_tensor, trainable = True, scope_name = self.P_scope_name)
		self.P_ = self.make_net_P(S = self.State1_tensor, trainable = False, scope_name = self.P_target_scope_name)
		self.Q = self.make_net_Q(S = self.State_tensor, P = self.Action_tensor, trainable = True, scope_name = self.Q_scope_name)
		self.Q_ = self.make_net_Q(S = self.State1_tensor, P =self.P_, trainable = False, scope_name = self.Q_target_scope_name)

		self.memory = np.zeros([self.mem_size,2*self.state_dim+self.action_dim+1])#states, action, reward, states_next
		self.cursor = 0;

		self.P_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.P_scope_name)
		self.P_target_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.P_target_scope_name)
		self.Q_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Q_scope_name)
		self.Q_target_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Q_target_scope_name)

		self.tau = tau

		self.hard_replace_P = [tf.assign(tp, p) for tp, p in zip(self.P_target_param, self.P_param)]
		self.hard_replace_Q = [tf.assign(tp, p) for tp, p in zip(self.Q_target_param, self.Q_param)]
		self.soft_replace_P = [tf.assign(tp, (1-self.tau)*tp+self.tau*p) for tp, p in zip(self.P_target_param, self.P_param)]
		self.soft_replace_Q = [tf.assign(tp, (1-self.tau)*tp+self.tau*p) for tp, p in zip(self.Q_target_param, self.Q_param)]

		self.gamma = gamma

		self.y = self.Reward_tensor + self.gamma * self.Q_

		self.QLoss = tf.reduce_mean(tf.square(self.y - self.Q))
		self.lr_Q = learning_rate_Q
		self.trainQ = tf.train.AdamOptimizer(self.lr_Q).minimize(self.QLoss)

		self.gradQ = tf.gradients(ys = self.Q, xs = self.Action_tensor)[0]
		self.lr_P = learning_rate_P
		self.gradP = tf.gradients(ys = self.P, xs = self.P_param, grad_ys = self.gradQ)
		self.opt_P = tf.train.AdamOptimizer(-self.lr_P)
		self.trainP = self.opt_P.apply_gradients(zip(self.gradP, self.P_param))

	def get_action(self,S):
		return self.sess.run(self.P, feed_dict={self.State_tensor: S})

	def train_model(self, batch):
		S = batch[:, 0 : self.state_dim]
		A = batch[:, self.state_dim : self.state_dim + self.action_dim]
		R = batch[:, (self.state_dim + self.action_dim):(self.state_dim + self.action_dim)+1]
		S_ = batch[:, self.state_dim + self.action_dim + 1 : ]

		self.sess.run(self.trainQ, feed_dict = {	self.State_tensor: S,
													self.Action_tensor: A, 
													self.Reward_tensor: R,
													self.State1_tensor: S_})
		self.sess.run(self.trainP, feed_dict = {	self.State_tensor: S,
													self.Action_tensor: A, 
													self.Reward_tensor: R,
													self.State1_tensor: S_})
		self.sess.run(self.soft_replace_P)
		self.sess.run(self.soft_replace_Q)


	def make_net_Q(self, S, P, trainable, scope_name):
		""" state size is the dimensions of states and scope name is
		a unique name of the each network"""

		with tf.variable_scope(scope_name):

			l1_num = 40
			l2_num = 40
			l3_num = 40
			l4_num = 40
			l5_num = 40
			l6_num = 1

			W1 = tf.get_variable("W1", shape=[self.state_dim + self.action_dim,l1_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W2 = tf.get_variable("W2", shape=[l1_num,l2_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W3 = tf.get_variable("W3", shape=[l2_num,l3_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W4 = tf.get_variable("W4", shape=[l3_num,l4_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W5 = tf.get_variable("W5", shape=[l4_num,l5_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W6 = tf.get_variable("W6", shape=[l5_num,l6_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)

			b1 = tf.get_variable(name="b1", shape=[l1_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b2 = tf.get_variable(name="b2", shape=[l2_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b3 = tf.get_variable(name="b3", shape=[l3_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b4 = tf.get_variable(name="b4", shape=[l4_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b5 = tf.get_variable(name="b5", shape=[l5_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b6 = tf.get_variable(name="b6", shape=[l6_num], initializer=tf.zeros_initializer(), trainable = trainable)

			L1=tf.nn.leaky_relu(tf.matmul(tf.concat([S, P], 1),W1)+b1)
			L2=tf.nn.leaky_relu(tf.matmul(L1,W2)+b2)
			L3=tf.nn.leaky_relu(tf.matmul(L2,W3)+b3)
			L4=tf.nn.leaky_relu(tf.matmul(L3,W4)+b4)
			L5=tf.nn.leaky_relu(tf.matmul(L4,W5)+b5)

			return (tf.matmul(L5,W6)+b6)
			# return (tf.matmul(L1,W2)+b2)


	def make_net_P(self, S, trainable, scope_name):
		""" state size is the dimensions of states and scope name is
		a unique name of the each network"""
		with tf.variable_scope(scope_name):

			l1_num = 30
			l2_num = 30
			l3_num = 30
			l4_num = 30
			l5_num = 30
			l6_num = self.action_dim

			W1 = tf.get_variable("W1", shape=[self.state_dim,l1_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W2 = tf.get_variable("W2", shape=[l1_num,l2_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W3 = tf.get_variable("W3", shape=[l2_num,l3_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W4 = tf.get_variable("W4", shape=[l3_num,l4_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W5 = tf.get_variable("W5", shape=[l4_num,l5_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)
			W6 = tf.get_variable("W6", shape=[l5_num,l6_num], \
				initializer=tf.contrib.layers.xavier_initializer(), trainable = trainable)

			b1 = tf.get_variable(name="b1", shape=[l1_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b2 = tf.get_variable(name="b2", shape=[l2_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b3 = tf.get_variable(name="b3", shape=[l3_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b4 = tf.get_variable(name="b4", shape=[l4_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b5 = tf.get_variable(name="b5", shape=[l5_num], initializer=tf.zeros_initializer(), trainable = trainable)
			b6 = tf.get_variable(name="b6", shape=[l6_num], initializer=tf.zeros_initializer(), trainable = trainable)

			L1=tf.nn.leaky_relu(tf.matmul(S,W1)+b1)
			L2=tf.nn.leaky_relu(tf.matmul(L1,W2)+b2)
			L3=tf.nn.leaky_relu(tf.matmul(L2,W3)+b3)
			L4=tf.nn.leaky_relu(tf.matmul(L3,W4)+b4)
			L5=tf.nn.leaky_relu(tf.matmul(L4,W5)+b5)

			return (tf.nn.tanh(tf.matmul(L5,W6)+b6))
			# return 2 * tf.tanh((tf.matmul(L1,W2)+b2))

	def store(self, data): # states, action, reward, states_next
		self.memory[self.cursor,:] = data
		self.cursor = (self.cursor + 1) % self.mem_size

	def get_batch(self,batch_size):
		r = np.random.choice(self.mem_size, size=batch_size)
		return self.memory[r,:]