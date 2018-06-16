
import tensorflow as tf

class Preprocessor:

	def __init__(self):

		def build_graph():
			# We first preprocess our images to grayscale.
			with tf.device("/device:GPU:0"):
				self.input_state = tf.placeholder(tf.uint8,shape=(None,None,3),name="state")
				grayscale = tf.image.rgb_to_grayscale(self.input_state)
				resize = tf.image.resize_images(grayscale,(110,84),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,align_corners=True)
				crop = resize[19:103,:]
				self.result = tf.squeeze(crop)

		build_graph()

	def process(self,sess,state):
		return sess.run(self.result,feed_dict={self.input_state:state})