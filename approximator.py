import tensorflow as tf
from ops import graves_rmsprop_optimizer

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

			conv1 = tf.layers.conv2d(inputs=normalized,filters=32,kernel_size=8,strides=4,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
			conv2 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=4,strides=2,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
			conv3 = tf.layers.conv2d(inputs=conv2,filters=64,kernel_size=3,strides=1,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
			flattened = tf.contrib.layers.flatten(conv3)
			dense3 = tf.layers.dense(inputs=flattened,units=512,kernel_initializer=tf.contrib.layers.xavier_initializer())
			self.dense4 = tf.layers.dense(inputs=dense3,units=self.nA,kernel_initializer=tf.contrib.layers.xavier_initializer())
			# This will be 32 x 4 we need to make it 32 x 1 but with the correct value


			gather_index = tf.range(tf.shape(self.actions)[0]) * self.nA + self.actions 
			action_values = tf.gather(tf.reshape(self.dense4,[-1]),gather_index)

			error = tf.losses.mean_squared_error(labels=self.targets,predictions=action_values)

			self.step,grads = graves_rmsprop_optimizer(error,0.00025,0.95, 0.01, 1)

	def sgd_step(self,sess,states,actions,targets):
		sess.run(self.step,feed_dict={self.states:states,self.actions:actions,self.targets:targets})

	def predict(self,sess,states):
		return sess.run(self.dense4,feed_dict={self.states:states})

	# The rmsprop optimizer used in original deep mind paper
	def graves_rmsprop_optimizer(loss, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip):
		with tf.name_scope('rmsprop'):
			optimizer = None
			optimizer = tf.train.GradientDescentOptimizer(learning_rate)

			grads_and_vars = optimizer.compute_gradients(loss)

			grads = []
			params = []
			for p in grads_and_vars:
				if p[0] == None:
					continue
				grads.append(p[0])
				params.append(p[1])
			#grads = [gv[0] for gv in grads_and_vars]
			#params = [gv[1] for gv in grads_and_vars]
			if gradient_clip > 0:
				grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

			square_grads = [tf.square(grad) for grad in grads]

			avg_grads = [tf.Variable(tf.zeros(var.get_shape()))
						 for var in params]
			avg_square_grads = [tf.Variable(
				tf.zeros(var.get_shape())) for var in params]

			update_avg_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + tf.scalar_mul((1 - rmsprop_decay), grad_pair[1]))
								for grad_pair in zip(avg_grads, grads)]
			update_avg_square_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1])))
									   for grad_pair in zip(avg_square_grads, grads)]
			avg_grad_updates = update_avg_grads + update_avg_square_grads

			rms = [tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
				   for avg_grad_pair in zip(avg_grads, avg_square_grads)]

			rms_updates = [grad_rms_pair[0] / grad_rms_pair[1]
						   for grad_rms_pair in zip(grads, rms)]
			train = optimizer.apply_gradients(zip(rms_updates, params))

			return tf.group(train, tf.group(*avg_grad_updates)), grads_and_vars
