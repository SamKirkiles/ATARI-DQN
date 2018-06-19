import tensorflow as tf
from ops import graves_rmsprop_optimizer

class QApproximator:

	def __init__(self,actions,scope):

		with tf.device("/device:GPU:0"), tf.variable_scope(scope):

			self.nA = actions
			self.weight_scope=scope

			# Takes in arbitrary number of states
			self.states = tf.placeholder(tf.uint8,shape=(None,84,84,4),name="states")
			self.targets = tf.placeholder(tf.float32, shape=(None),name="targets")
			self.actions = tf.placeholder(tf.int32, shape=(None),name="actions")


			cast = tf.cast(self.states,tf.float32)

			normalized = tf.divide(cast,255.0)

			conv1 = tf.layers.conv2d(inputs=normalized,filters=32,kernel_size=8,strides=4,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
			conv2 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=4,strides=2,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
			conv3 = tf.layers.conv2d(inputs=conv2,filters=64,kernel_size=3,strides=1,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
			flattened = tf.contrib.layers.flatten(conv3)
			dense3 = tf.layers.dense(inputs=flattened,units=512,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
			self.dense4 = tf.layers.dense(inputs=dense3,units=self.nA,kernel_initializer=tf.contrib.layers.xavier_initializer())

			selected_actions = tf.reduce_sum(self.dense4 * tf.one_hot(self.actions,self.nA),axis=1)

			td_error = selected_actions - self.targets
			loss = tf.reduce_mean(tf.square(td_error))

			self.step,grads = graves_rmsprop_optimizer(loss,0.00025,0.95, 0.01, 1)

	def sgd_step(self,sess,states,actions,targets):
		sess.run(self.step,feed_dict={self.states:states,self.actions:actions,self.targets:targets})

	def predict(self,sess,states):
		return sess.run(self.dense4,feed_dict={self.states:states})
