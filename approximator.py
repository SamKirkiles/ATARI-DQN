import tensorflow as tf

class QApproximator:

	def __init__(self,actions,scope):

		with tf.device("/device:GPU:0"), tf.variable_scope(scope):

			self.nA = actions
			self.weight_scope=scope

			# Takes in arbitrary number of states
			self.states = tf.placeholder(tf.uint8,shape=(None,84,84,4),name="states")
			self.targets = tf.placeholder(tf.uint8, shape=(None),name="targets")
			self.actions = tf.placeholder(tf.int32, shape=(None),name="actions")


			cast = tf.cast(self.states,tf.float32)

			normalized = tf.divide(cast,255.0)

			conv1 = tf.layers.conv2d(inputs=normalized,filters=16,kernel_size=8,strides=4,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
			conv2 = tf.layers.conv2d(inputs=conv1,filters=32,kernel_size=4,strides=2,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
			flattened = tf.contrib.layers.flatten(conv2)
			dense3 = tf.layers.dense(inputs=flattened,units=256,kernel_initializer=tf.contrib.layers.xavier_initializer())
			self.dense4 = tf.layers.dense(inputs=dense3,units=self.nA,kernel_initializer=tf.contrib.layers.xavier_initializer())
			# This will be 32 x 4 we need to make it 32 x 1 but with the correct value


			gather_index = tf.range(tf.shape(self.actions)[0]) * self.nA + self.actions 
			action_values = tf.gather(tf.reshape(self.dense4,[-1]),gather_index)

			error = tf.losses.mean_squared_error(labels=self.targets,predictions=action_values)

			self.step = tf.train.RMSPropOptimizer(learning_rate=0.00025,decay=0.99,momentum=0.95).minimize(error)

	def sgd_step(self,sess,states,actions,targets):
		return sess.run(self.step,feed_dict={self.states:states,self.actions:actions,self.targets:targets})

	def predict(self,sess,states):
		return sess.run(self.dense4,feed_dict={self.states:states})
