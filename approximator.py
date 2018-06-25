import tensorflow as tf
from ops import graves_rmsprop_optimizer

class QApproximator:

	def __init__(self,actions,scope,momentum):

		with tf.device("/device:GPU:0"), tf.variable_scope(scope):

			self.nA = actions
			self.weight_scope=scope

			# Takes in arbitrary number of states
			self.states = tf.placeholder(tf.uint8,shape=(None,84,84,4),name="states")
			self.targets = tf.placeholder(tf.float32, shape=(None),name="targets")
			self.actions = tf.placeholder(tf.int32, shape=(None),name="actions")


			cast = tf.cast(self.states,tf.float32)

			normalized = tf.divide(cast,255.0)

			# Three convolutional layers
			conv1 = tf.contrib.layers.conv2d(normalized, 32, 8, 4, activation_fn=tf.nn.relu)
			conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
			conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

			# Fully connected layers
			flattened = tf.contrib.layers.flatten(conv3)
			fc1 = tf.contrib.layers.fully_connected(flattened, 512)
			self.dense4 = tf.contrib.layers.fully_connected(fc1, 4,activation_fn=None)

			selected_actions = tf.reduce_sum(self.dense4 * tf.one_hot(self.actions,self.nA),axis=1)
			
			self.losses = tf.losses.huber_loss(labels=self.targets,predictions=selected_actions,delta=1.0)
			self.loss = tf.reduce_mean(self.losses)

			# Optimizer Parameters from original paper
			self.optimizer= tf.train.RMSPropOptimizer(0.00025,momentum=momentum,epsilon=0.01)

			self.step = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

	def sgd_step(self,sess,states,actions,targets):
		_,loss = sess.run([self.step,self.loss],feed_dict={self.states:states,self.actions:actions,self.targets:targets})
		return loss

	def predict(self,sess,states):
		return sess.run(self.dense4,feed_dict={self.states:states})

	
