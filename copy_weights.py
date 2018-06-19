import tensorflow as tf

class WeightCopier:

	def __init__(self,q,target_q):

		params_q = tf.trainable_variables(scope=q.weight_scope)
		params_target = tf.trainable_variables(scope=target_q.weight_scope)

		updates = []

		for v1,v2 in zip(sorted(params_q, key=lambda v: v.name),sorted(params_target, key=lambda v: v.name)):
			updates.append(v2.assign(v1))

        self.update_target = tf.group(*update)

	def copy(self,sess):
		sess.run(self.update_target)