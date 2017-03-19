'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Surjodoy Ghosh Dastider
Roll No.: 16CS60R75

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf

weight_path = "weights/dnn"


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W_fc1 = weight_variable([784, 1000])
b_fc1 = bias_variable([1000])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
W_fc2 = weight_variable([1000, 500])
b_fc2 = bias_variable([500])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
W_fc3 = weight_variable([500, 300])
b_fc3 = bias_variable([300])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
W_fc4 = weight_variable([300, 10])
b_fc4 = bias_variable([10])
y_dense = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)

epochs = 10
batch_size = 100

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_dense))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_dense,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	

def train(trainX, trainY):
	'''
	Complete this function.
	'''
	trainX_ = trainX.reshape(-1,784)
	trainY_ = np.zeros((len(trainY), 10))
	for i in range(len(trainY)):
		trainY_[i][trainY[i]] = 1

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	for i in range(epochs):
		batch_start = 0
		while batch_start<len(trainX_):
			batch_end = batch_start+batch_size
			batch_x = trainX_[batch_start:batch_end]
			batch_y = trainY_[batch_start:batch_end]
			batch_start = batch_end
			sess.run(train_step, feed_dict={x:batch_x, y_:batch_y})
			train_accuracy = sess.run(accuracy, feed_dict={x:batch_x, y_:batch_y})
		print("epoch "+str(i)+", training accuracy "+str(train_accuracy*100)+"%")

	print("train accuracy "+str(sess.run(accuracy, feed_dict={x: trainX_, y_: trainY_})*100)+"%")
	save_path = saver.save(sess, weight_path)


def test(testX):
	'''
	Complete this function.
	This function must read the weight files and
	return the predicted labels.
	The returned object must be a 1-dimensional numpy array of
	length equal to the number of examples. The i-th element
	of the array should contain the label of the i-th test
	example.
	'''

	testX_ = testX.reshape(-1,784)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, weight_path)

	predictions = sess.run(tf.argmax(y_dense,1), feed_dict={x: testX_})

	return predictions
