
import tensorflow as tf

class Preprocessor:

	def __init__(self):

		def build_graph():
			# We first preprocess our images to grayscale.
			with tf.device("/device:GPU:0"):
				self.input_state = tf.placeholder(tf.uint8,shape=(None,None,3),name="state")
				grayscale = tf.image.rgb_to_grayscale(self.input_state)
				self.output = tf.image.crop_to_bounding_box(grayscale, 34, 0, 160, 160)
				self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
				self.output = tf.squeeze(self.output)

		build_graph()

	def process(self,sess,state):
		return sess.run(self.output,feed_dict={self.input_state:state})