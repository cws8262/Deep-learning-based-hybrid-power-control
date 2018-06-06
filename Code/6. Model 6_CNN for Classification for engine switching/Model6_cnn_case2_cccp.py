import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as sio
import tensorflow as tf
import time
from sklearn.cross_validation import train_test_split


input_=sio.loadmat('train_X.mat',squeeze_me=False)
input_=np.array(input_['X'])

output_=sio.loadmat('train_Y.mat',squeeze_me=False)
output_=np.array(output_['y'])

X_train=input_
y_train=output_


input_=sio.loadmat('test_X.mat',squeeze_me=False)
input_=np.array(input_['X'])

output_=sio.loadmat('test_Y.mat',squeeze_me=False)
output_=np.array(output_['y'])

X_test=input_
y_test=output_

Xmean=np.mean(X_train,axis=0)
Xsig=np.sqrt(np.var(X_train,axis=0))
X_test=(X_test-Xmean)/Xsig
X_train=(X_train-Xmean)/Xsig

#//////////////////////////////////////////////////////////////////////////////////////////////////////

X=tf.placeholder(tf.float32,[None,X_train.shape[1],X_train.shape[2], 1])#feature x sample x window x num
Y=tf.placeholder(tf.float32,[None,1])#feature x sample
drop_out=tf.placeholder(tf.float32)

with tf.variable_scope("CPL") as cpl_v_scope:
	l1_conv_size = 3
	l1_filter_num = 3
	l1_pool_size = 2
	W1_cnn = tf.Variable(tf.random_normal([l1_conv_size, l1_conv_size, 1, l1_filter_num], stddev=0.01))
	b1_cnn = tf.Variable(tf.zeros([l1_filter_num]))
	L1_cnn = tf.nn.conv2d(X, W1_cnn, strides=[1, 1, 1, 1], padding='SAME')
	L1_cnn = tf.nn.relu(L1_cnn)
	L1_cnn = tf.nn.dropout(L1_cnn, keep_prob=drop_out)

	l2_conv_size = 3
	l2_filter_num = 6
	l2_pool_size = 2
	W2_cnn = tf.Variable(tf.random_normal([l2_conv_size, l2_conv_size, l1_filter_num, l2_filter_num], stddev=0.01))
	b2_cnn = tf.Variable(tf.zeros([l2_filter_num]))
	L2_cnn = tf.nn.conv2d(L1_cnn, W2_cnn, strides=[1, 1, 1, 1], padding='SAME')
	L2_cnn = tf.nn.relu(L2_cnn)
	L2_cnn = tf.nn.dropout(L2_cnn, keep_prob=drop_out)

	l3_conv_size = 3
	l3_filter_num = 9
	l3_pool_size = 2
	W3_cnn = tf.Variable(tf.random_normal([l3_conv_size, l3_conv_size, l2_filter_num, l3_filter_num], stddev=0.01))
	b3_cnn = tf.Variable(tf.zeros([l3_filter_num]))
	L3_cnn = tf.nn.conv2d(L2_cnn, W3_cnn, strides=[1, 1, 1, 1], padding='SAME')
	L3_cnn = tf.nn.relu(L3_cnn)
	L3_cnn = tf.nn.max_pool(L3_cnn, ksize=[1, 1, l3_pool_size, 1],strides=[1, 1, l3_pool_size, 1], padding='SAME')
	L3_cnn = tf.nn.dropout(L3_cnn, keep_prob=drop_out)

	L_cnn = tf.reshape(L3_cnn, [-1, l3_filter_num * int(L3_cnn.shape[1]) * int(L3_cnn.shape[2])])

with tf.variable_scope("FCL") as fcl_v_scope:
   l1_num=20
   W1 = tf.get_variable("W1", shape=[L_cnn.shape[1],l1_num], initializer=tf.contrib.layers.xavier_initializer())
   b1 = tf.Variable(tf.zeros([l1_num]))
   L1 = tf.nn.relu(tf.matmul(L_cnn,W1)+b1)
   L1 = tf.nn.dropout(L1, keep_prob=drop_out)

   l2_num=20
   W2 = tf.get_variable("W2", shape=[l1_num,l2_num], initializer=tf.contrib.layers.xavier_initializer())
   b2 = tf.Variable(tf.zeros([l2_num]))
   L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
   L2 = tf.nn.dropout(L2, keep_prob=drop_out)

   l3_num=20
   W3 = tf.get_variable("W3", shape=[l2_num,l3_num], initializer=tf.contrib.layers.xavier_initializer())
   b3 = tf.Variable(tf.zeros([l3_num]))
   L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)
   L3 = tf.nn.dropout(L3, keep_prob=drop_out)

   l4_num=20
   W4 = tf.get_variable("W4", shape=[l3_num,l4_num], initializer=tf.contrib.layers.xavier_initializer())
   b4 = tf.Variable(tf.zeros([l4_num]))
   L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
   L4 = tf.nn.dropout(L4, keep_prob=drop_out)

   l5_num=20
   W5 = tf.get_variable("W5", shape=[l4_num,l5_num], initializer=tf.contrib.layers.xavier_initializer())
   b5= tf.Variable(tf.zeros([l5_num]))
   L5=tf.nn.relu(tf.matmul(L4,W5)+b5)
   L5=tf.nn.dropout(L5, keep_prob=drop_out)

   l6_num=20
   W6 = tf.get_variable("W6", shape=[l5_num,l6_num], initializer=tf.contrib.layers.xavier_initializer())
   b6= tf.Variable(tf.zeros([l6_num]))
   L6=tf.nn.relu(tf.matmul(L5,W6)+b6)
   L6=tf.nn.dropout(L6, keep_prob=drop_out)

   l7_num=20
   W7 = tf.get_variable("W7", shape=[l6_num,l7_num], initializer=tf.contrib.layers.xavier_initializer())
   b7= tf.Variable(tf.zeros([l7_num]))
   L7=tf.nn.relu(tf.matmul(L6,W7)+b7)
   L7=tf.nn.dropout(L7, keep_prob=drop_out)

   l8_num=1
   W8 = tf.get_variable("W8", shape=[l7_num,l8_num], initializer=tf.contrib.layers.xavier_initializer())
   b8 = tf.Variable(tf.zeros([l8_num]))

   hypothesis=tf.sigmoid((tf.matmul(L7, W8) + b8))

cost = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(hypothesis, 1e-15, 1.0)) + (1 - Y)*tf.log(tf.clip_by_value(1-hypothesis, 1e-15, 1.0)))

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

saver = tf.train.Saver()

init=tf.global_variables_initializer()

def stochastic(X,y,batch):
   r = np.random.permutation(X.shape[0])
   size=int(X.shape[0]/batch)
   X=X[:size*batch,...]
   y=y[:size*batch,...]
   return np.split(X,size,axis=0), np.split(y,size,axis=0), size


with tf.Session() as sess:
   sess.run(init)
   saver.restore(sess, "Weighting_factor_case2.ckpt")
   batch = 512

   X_temp = np.reshape(X_train, [-1, X_train.shape[1], X_train.shape[2], 1])
   y_temp = y_train
   X_test_temp = np.reshape(X_test, [-1, X_test.shape[1], X_test.shape[2], 1])
   y_test_temp = y_test

   for step in range(0):
    
      sX,sY,ssize=stochastic(X_temp,y_temp,batch)
      for i in range(ssize):
         sess.run(optimizer, feed_dict={X:sX[i], Y:sY[i],drop_out:1.0})
      costval_tst=sess.run(accuracy, feed_dict={X:X_test_temp, Y:y_test_temp,drop_out:1.0})
 
      print(step+1,costval_tst)
      if (step+1) % 10 == 0:
         save_path = saver.save(sess, "Weighting_factor_case2.ckpt")

   variables_names = [v.name for v in tf.trainable_variables() if v.name.startswith(cpl_v_scope.name)]
   values = sess.run(variables_names)
   i=0
   cpl_wb = np.zeros((6,), dtype=np.object)
   for k, v in zip(variables_names, values):
       cpl_wb[i]=v;
       i=i+1
       # print(i)

   variables_names = [v.name for v in tf.trainable_variables() if v.name.startswith(fcl_v_scope.name)]
   values = sess.run(variables_names)
   i=0
   fcl_wb = np.zeros((16,), dtype=np.object)
   for k, v in zip(variables_names, values):
       fcl_wb[i]=v;
       i=i+1

   sio.savemat('cpl_wb_case2.mat', {'cpl_wb' :cpl_wb})
   sio.savemat('fcl_wb_case2.mat', {'fcl_wb' :fcl_wb})
   sio.savemat('xmean_case2.mat',{'xmean' :Xmean})
   sio.savemat('xsig_case2.mat',{'xsig' :Xsig})

