import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessor import Preprocessor
from approximator import QApproximator
from copy_weights import WeightCopier
import numpy as np
import random

class DQN:

	def __init__(self,env_name="BreakoutDeterministic-v4"):
		self.env = gym.make(env_name)
		self.nA = self.env.action_space.n
		
		self.Q =  QApproximator(self.nA,"Q")
		self.Q_ =  QApproximator(self.nA,"Q_")

		self.counter = 0
		self.epsilons = np.linspace(1.0, 0.1, 500000)

		self.runid = np.random.randint(10000)

		self.max_replay = 500000
			
		self.copier = WeightCopier(self.Q,self.Q_)

	def train(self,num_episodes=500000,discount=0.99,restore=True):

		saver = tf.train.Saver()

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
			
			filewriter = tf.summary.FileWriter(logdir="logs/" + str(self.runid), graph=sess.graph)
			merge = tf.summary.merge_all()

			processor = Preprocessor()


			if restore:
				saver.restore(sess, tf.train.latest_checkpoint('./saves'))
			else:
				sess.run(tf.global_variables_initializer())

			D = []

			# Initialize replay memory
			state = self.env.reset()
			state = processor.process(sess,state)
			state = [state] * 4
			state = np.stack(state,axis=2)

			for _ in range(50000):
				action = np.random.choice(self.nA,p=self._policy(sess,state,epsilon=self.epsilons[min(len(self.epsilons)-1,self.counter)]))
				next_state,reward,done,_ = self.env.step(action)
				reward = np.clip(reward, 0, 1)
				next_state = processor.process(sess,next_state)
				next_state = np.append(state[:,:,1:],next_state[...,None],axis=2)
				transition_tuple = (state,action,reward,next_state,done)
				D.append(transition_tuple)
				state = next_state

			episode_reward = 0
			avg_reward = 0
			beta = 0.01
			
			while True:
			
				state = self.env.reset()
				state = processor.process(sess,state)
				
				state = [state] * 4
				state = np.stack(state,axis=2)

				reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Reward',simple_value=episode_reward)])
				filewriter.add_summary(reward_summary,self.counter)

				episode_reward = 0

				
				while True:

					self.env.render()
					self.counter += 1
					e = self.epsilons[min(len(self.epsilons)-1,self.counter)]


					action = np.random.choice(self.nA,p=self._policy(sess,state,e))
					next_state,reward,done,_ = self.env.step(action)
					reward = np.clip(reward, 0, 1)
					next_state = processor.process(sess,next_state)
					next_state = np.append(state[:,:,1:],next_state[...,None],axis=2)

					transition_tuple = (state,action,reward,next_state,done)

					D.append(transition_tuple)

					episode_reward += reward
					avg_reward =  (1-beta) * episode_reward + beta * reward

					print("Iteration: " + str(self.counter) + " Epsilon: " + str(e) + " Avg Reward: " + str(round(avg_reward,4)), end="\r", flush=True)


					if len(D) > self.max_replay:
						del D[0]
					if self.counter%10000 == 0:
						self.copier.copy(sess)
						# Save model
						print("Periodically saving model...")
						save_path = saver.save(sess, "./saves/model.ckpt")

					if self.counter % 20 == 0:
						epsilon_summary = tf.Summary(value=[tf.Summary.Value(tag='Epsilon',simple_value=e)])
						filewriter.add_summary(epsilon_summary,self.counter)
					if done:
						break

					batch = random.sample(D,32)

					b_states,b_actions,b_rewards,b_next_states,b_done= map(np.array,zip(*batch))

					targets = b_rewards + discount  * np.invert(b_done).astype(np.float32) * np.amax(self.Q_.predict(sess,b_next_states),axis=1)

					self.Q.sgd_step(sess,b_states,b_actions,targets)

					
					state = next_state

		
	def _policy(self,sess,state,epsilon=0.1):

		# Epsilon Greedy Policy

		A = np.zeros(self.nA)
		A += (epsilon/self.nA)
		best_action = np.argmax(self.Q.predict(sess,state[None,...]))
		A[best_action] += 1-epsilon
		return A


	def demo(self,steps=10000):

		state = self.env.reset()


		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
			processor = Preprocessor()

			for i in range(1):
				state,_,_,_ = self.env.step(self.env.action_space.sample())
				image = processor.process(sess,state)
				plt.imshow(image,cmap='gray')
				plt.figure()
			plt.show()
		self.env.close()

