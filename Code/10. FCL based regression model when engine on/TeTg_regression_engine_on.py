import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as sio
import tensorflow as tf
import time
from sklearn.cross_validation import train_test_split


input_=sio.loadmat('engon_train_X.mat',squeeze_me=False)
input_=np.array(input_['X_for_engon'])

output_=sio.loadmat('engon_train_Y.mat',squeeze_me=False)
output_=np.array(output_['y_for_engon'])

X_train=input_
y_train=output_


input_=sio.loadmat('engon_test_X.mat',squeeze_me=False)
input_=np.array(input_['X_for_engon'])

output_=sio.loadmat('engon_test_y.mat',squeeze_me=False)
output_=np.array(output_['y_for_engon'])

X_test=input_
y_test=output_

Xmean=np.mean(X_train,axis=0)
Xsig=np.sqrt(np.var(X_train,axis=0))

X_test=(X_test-Xmean)/Xsig
X_train=(X_train-Xmean)/Xsig

Ymean=np.mean(y_train,axis=0)
Ysig=np.sqrt(np.var(y_train,axis=0))

y_test=(y_test-Ymean)/Ysig
y_train=(y_train-Ymean)/Ysig

#//////////////////////////////////////////////////////////////////////////////////////////////////////

X=tf.placeholder(tf.float32,[None,X_train.shape[1]])#feature x sample 
Y=tf.placeholder(tf.float32,[None,2])#feature x sample
drop_out=tf.placeholder(tf.float32)

with tf.variable_scope("FCL") as fcl_v_scope:
   l1_num=40
   W1 = tf.get_variable("W1", shape=[X_train.shape[1],l1_num], initializer=tf.contrib.layers.xavier_initializer())
   b1 = tf.Variable(tf.zeros([l1_num]))
   L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
   L1 = tf.nn.dropout(L1, keep_prob=drop_out)

   l2_num=40
   W2 = tf.get_variable("W2", shape=[l1_num,l2_num], initializer=tf.contrib.layers.xavier_initializer())
   b2= tf.Variable(tf.zeros([l2_num]))
   L2=tf.nn.relu(tf.matmul(L1,W2)+b2)
   L2=tf.nn.dropout(L2, keep_prob=drop_out)

   l3_num=40
   W3 = tf.get_variable("W3", shape=[l2_num,l3_num], initializer=tf.contrib.layers.xavier_initializer())
   b3= tf.Variable(tf.zeros([l3_num]))
   L3=tf.nn.relu(tf.matmul(L2,W3)+b3)
   L3=tf.nn.dropout(L3, keep_prob=drop_out)

   l4_num=40
   W4 = tf.get_variable("W4", shape=[l3_num,l4_num], initializer=tf.contrib.layers.xavier_initializer())
   b4= tf.Variable(tf.zeros([l4_num]))
   L4=tf.nn.relu(tf.matmul(L3,W4)+b4)
   L4=tf.nn.dropout(L4, keep_prob=drop_out)

   l5_num=40
   W5 = tf.get_variable("W5", shape=[l4_num,l5_num], initializer=tf.contrib.layers.xavier_initializer())
   b5= tf.Variable(tf.zeros([l5_num]))
   L5=tf.nn.relu(tf.matmul(L4,W5)+b5)
   L5=tf.nn.dropout(L5, keep_prob=drop_out)

   l6_num=40
   W6 = tf.get_variable("W6", shape=[l5_num,l6_num], initializer=tf.contrib.layers.xavier_initializer())
   b6= tf.Variable(tf.zeros([l6_num]))
   L6=tf.nn.relu(tf.matmul(L5,W6)+b6)
   L6=tf.nn.dropout(L6, keep_prob=drop_out)

   l7_num=40
   W7 = tf.get_variable("W7", shape=[l6_num,l7_num], initializer=tf.contrib.layers.xavier_initializer())
   b7= tf.Variable(tf.zeros([l7_num]))
   L7=tf.nn.relu(tf.matmul(L6,W7)+b7)
   L7=tf.nn.dropout(L7, keep_prob=drop_out)

   l8_num=2
   W8 = tf.get_variable("W8", shape=[l7_num,l8_num], initializer=tf.contrib.layers.xavier_initializer())
   b8= tf.Variable(tf.zeros([l8_num]))

   hypothesis=(tf.matmul(L7, W8) + b8)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate=0.0001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()

init=tf.global_variables_initializer()

def stochastic(X,y,batch):
   r = np.random.permutation(X.shape[0])
   X=np.copy(X[r,...])
   y=np.copy(y[r,...])
   size=int(X.shape[0]/batch)
   X=X[:size*batch,...]
   y=y[:size*batch,...]
   return np.split(X,size,axis=0), np.split(y,size,axis=0), size


with tf.Session() as sess:
   sess.run(init)
   # saver.restore(sess, "Weighting_factor_engineon.ckpt")
   batch=512

   X_temp=X_train
   y_temp=y_train
   costval=sess.run(cost, feed_dict={X:X_train, Y:y_train,drop_out:1.0})
   costval_tst=sess.run(cost, feed_dict={X:X_test, Y:y_test,drop_out:1.0})

   print(0,costval,costval_tst)

   for step in range(300):
  
      sX,sY,ssize=stochastic(X_temp,y_temp,batch)
      for i in range(ssize):
         sess.run(optimizer, feed_dict={X:sX[i], Y:sY[i],drop_out:1.0})
      costval=sess.run(cost, feed_dict={X:X_train, Y:y_train,drop_out:1.0})
      costval_tst=sess.run(cost, feed_dict={X:X_test, Y:y_test,drop_out:1.0})
      print(step+1,costval,costval_tst)
      if (step+1) % 10 == 0:
         save_path = saver.save(sess, "Weighting_factor_engineon.ckpt")

   variables_names = [v.name for v in tf.trainable_variables() if v.name.startswith(fcl_v_scope.name)]
   values = sess.run(variables_names)
   i=0
   fcl_wb = np.zeros((16,), dtype=np.object)
   for k, v in zip(variables_names, values):
       fcl_wb[i]=v;
       i=i+1
   sio.savemat('fcl_wb_engon.mat', {'fcl_wb' :fcl_wb})

   sio.savemat('xmean_engon.mat',{'xmean' :Xmean})
   sio.savemat('xsig_engon.mat',{'xsig' :Xsig})
   sio.savemat('ymean_engon.mat',{'ymean' :Ymean})
   sio.savemat('ysig_engon.mat',{'ysig' :Ysig})