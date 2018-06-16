import tensorflow as tf

class WeightCopier:

	def __init__(self,from_approx,to_approx):

		params1 = tf.trainable_variables(scope=from_approx.weight_scope)
		params2 = tf.trainable_variables(scope=to_approx.weight_scope)

		self.updates = []
		for v1,v2 in zip(params1,params2):
			self.updates.append(v2.assign(v1))

	def copy(self,sess):
		sess.run(self.updates)