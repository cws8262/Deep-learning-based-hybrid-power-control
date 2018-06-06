import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as sio
import tensorflow as tf
import time
from sklearn.cross_validation import train_test_split


input_=sio.loadmat('train_X_3.mat',squeeze_me=False)
input_=np.array(input_['X'])

output_=sio.loadmat('train_Y_3.mat',squeeze_me=False)
output_=np.array(output_['y'])

X_train=input_
y_train=output_


input_=sio.loadmat('test_X_3.mat',squeeze_me=False)
input_=np.array(input_['X'])

output_=sio.loadmat('test_Y_3.mat',squeeze_me=False)
output_=np.array(output_['y'])

X_test=input_
y_test=output_

newX=np.reshape(X_train,(X_train.shape[0]*X_train.shape[1],X_train.shape[2]))

Xmean_=np.mean(newX,axis=0).reshape(1,newX.shape[1])
Xmean=np.repeat(Xmean_,X_train.shape[1],axis=0)
Xsig_=np.sqrt(np.var(newX,axis=0)).reshape(1,newX.shape[1])
Xsig=np.repeat(Xsig_,X_train.shape[1],axis=0)

X_test=(X_test-Xmean)/Xsig
X_train=(X_train-Xmean)/Xsig

Ymean=np.mean(y_train,axis=0)
Ysig=np.sqrt(np.var(y_train,axis=0))

y_test=(y_test-Ymean)/Ysig
y_train=(y_train-Ymean)/Ysig
#//////////////////////////////////////////////////////////////////////////////////////////////////////

X=tf.placeholder(tf.float32,[None,X_train.shape[1],X_train.shape[2]])# batch size, seq_len, data_dim
Y=tf.placeholder(tf.float32,[None,2])#feature x sample
drop_out=tf.placeholder(tf.float32)

with tf.variable_scope("LSTM") as lstm_v_scope:
  Mcell= tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=60, activation=tf.tanh) for _ in range(1)])
  Mcell_DO = tf.contrib.rnn.DropoutWrapper(Mcell, output_keep_prob=drop_out)
  outputs, _states = tf.nn.dynamic_rnn(Mcell_DO, X, dtype=tf.float32)

with tf.variable_scope("FCL") as fcl_v_scope:
   l1_num=10
   W1 = tf.get_variable("W1", shape=[outputs[:,-1].shape[1],l1_num], initializer=tf.contrib.layers.xavier_initializer())
   b1 = tf.Variable(tf.zeros([l1_num]))
   L1 = tf.nn.relu(tf.matmul(outputs[:,-1],W1)+b1)
   L1 = tf.nn.dropout(L1, keep_prob=drop_out)

   l2_num=10
   W2 = tf.get_variable("W2", shape=[l1_num,l2_num], initializer=tf.contrib.layers.xavier_initializer())
   b2= tf.Variable(tf.zeros([l2_num]))
   L2=tf.nn.relu(tf.matmul(L1,W2)+b2)
   L2=tf.nn.dropout(L2, keep_prob=drop_out)

   l3_num=10
   W3 = tf.get_variable("W3", shape=[l2_num,l3_num], initializer=tf.contrib.layers.xavier_initializer())
   b3= tf.Variable(tf.zeros([l3_num]))
   L3=tf.nn.relu(tf.matmul(L2,W3)+b3)
   L3=tf.nn.dropout(L3, keep_prob=drop_out)

   l4_num=10
   W4 = tf.get_variable("W4", shape=[l3_num,l4_num], initializer=tf.contrib.layers.xavier_initializer())
   b4= tf.Variable(tf.zeros([l4_num]))
   L4=tf.nn.relu(tf.matmul(L3,W4)+b4)
   L4=tf.nn.dropout(L4, keep_prob=drop_out)

   l5_num=2
   W5 = tf.get_variable("W5", shape=[l4_num,l5_num], initializer=tf.contrib.layers.xavier_initializer())
   b5 = tf.Variable(tf.zeros([l5_num]))

   hypothesis=(tf.matmul(L4, W5) + b5)

regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5)

beta = 0
cost = tf.reduce_mean(tf.square(hypothesis - Y) + beta * regularizers)

learning_rate=0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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
   # saver.restore(sess, "Weighting_factor_3.ckpt")
   batch=512

   X_temp=X_train
   y_temp=y_train

   for step in range(300):
    
      sX,sY,ssize=stochastic(X_temp,y_temp,batch)
      cost_train = np.zeros([ssize,1])
      for i in range(ssize):
         sess.run(optimizer, feed_dict={X:sX[i], Y:sY[i],drop_out:1.0})
         cost_trin=sess.run(cost, feed_dict={X:sX[i], Y:sY[i],drop_out:1.0})
         cost_train[i, 0] = cost_trin

      sX_t,sY_t,ssize_t=stochastic(X_test,y_test,batch)
      cost_test = np.zeros([ssize_t,1])
      for i in range(ssize_t):
         cost_tst=sess.run(cost, feed_dict={X:sX_t[i], Y:sY_t[i],drop_out:1.0})
         cost_test[i, 0] = cost_tst

      print(step+1,np.mean(cost_train), np.mean(cost_test))

      if (step+1) % 10 == 0:
         save_path = saver.save(sess, "Weighting_factor_3.ckpt")


   variables_names = [v.name for v in tf.trainable_variables() if v.name.startswith(lstm_v_scope.name)]
   values = sess.run(variables_names)
   i=0
   lstm_wb = np.zeros((4,), dtype=np.object)
   for k, v in zip(variables_names, values):
       lstm_wb[i]=v;
       i=i+1
   sio.savemat('lstm_wb_3.mat', {'lstm_wb':lstm_wb})


   variables_names = [v.name for v in tf.trainable_variables() if v.name.startswith(fcl_v_scope.name)]
   values = sess.run(variables_names)
   i=0
   fcl_wb = np.zeros((10,), dtype=np.object)
   for k, v in zip(variables_names, values):
       fcl_wb[i]=v;
       i=i+1

   sio.savemat('fcl_wb_3.mat', {'fcl_wb' :fcl_wb})
   sio.savemat('xmean_3.mat',{'xmean' :Xmean_})
   sio.savemat('xsig_3.mat',{'xsig' :Xsig_})
   sio.savemat('ymean_3.mat',{'ymean' :Ymean})
   sio.savemat('ysig_3.mat',{'ysig' :Ysig})