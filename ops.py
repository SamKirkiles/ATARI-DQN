import tensorflow as tf
import numpy as np

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
