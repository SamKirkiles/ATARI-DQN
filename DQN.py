import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessor import Preprocessor
from approximator import QApproximator
from copy_weights import WeightCopier
import numpy as np
import random
from gym.wrappers import Monitor
import os
from replay_mem import Memory

class DQN:

	def __init__(self,env_name="BreakoutDeterministic-v0"):
		self.env = gym.make(env_name)
		self.nA = self.env.action_space.n
		
		self.Q =  QApproximator(self.nA,"Q")
		self.Q_ =  QApproximator(self.nA,"Q_")

		self.counter = 0
		self.episode = 0
		self.epsilons = np.linspace(1.0, 0.1, 500000)

		self.runid = np.random.randint(10000)

		self.max_replay = 1000000
			
		self.copier = WeightCopier(self.Q,self.Q_)

		self.D = Memory()

	def train(self,num_episodes=500000,discount=0.99,restore=True):

		saver = tf.train.Saver(max_to_keep=5,keep_checkpoint_every_n_hours=1)

		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			
			filewriter = tf.summary.FileWriter(logdir="logs/" + str(self.runid), graph=sess.graph)
			merge = tf.summary.merge_all()

			processor = Preprocessor()

			monitor_path = "./monitor"
			if not os.path.exists(monitor_path):
				os.makedirs(monitor_path)


			if restore:
				saver.restore(sess, tf.train.latest_checkpoint('./saves'))
			else:
				sess.run(tf.global_variables_initializer())


			# Initialize replay memory
			state = self.env.reset()
			state = processor.process(sess,state)
			temp_state = [state] * 4
			temp_state = np.stack(temp_state,axis=2)

			for _ in range(50000):
				action = np.random.choice(self.nA,p=self._policy(sess,temp_state,epsilon=self.epsilons[min(len(self.epsilons)-1,self.counter)]))
				next_state,reward,done,_ = self.env.step(action)
				reward = np.clip(reward, 0, 1)
				next_state = processor.process(sess,next_state)
				temp_next_state = np.append(temp_state[:,:,1:],next_state[...,None],axis=2)
				self.D.add(state,action,reward,done,next_state)
				state = next_state
				temp_state = temp_next_state

			episode_reward = 0
			avg_reward = 0
			beta = 0.01
			e=1.0
			
			self.env = Monitor(self.env,directory=monitor_path, video_callable=lambda count: count % 250 == 0, resume=True)

			while True:
			
				state = self.env.reset()
				state = processor.process(sess,state)
				temp_state = [state] * 4
				temp_state = np.stack(temp_state,axis=2)

				reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Reward',simple_value=episode_reward)])
				filewriter.add_summary(reward_summary,self.counter)

				print("Iteration: " + str(self.counter) + " Episode: " + str(self.episode) + " Epsilon: " + str(e) + " Episode Reward: " + str(episode_reward) + " Replay Size: " + str(self.D.replay_length()), end="\r", flush=True)

				episode_reward = 0
				self.episode += 1
				
				while True:

					self.counter += 1
					e = self.epsilons[min(len(self.epsilons)-1,self.counter)]


					action = np.random.choice(self.nA,p=self._policy(sess,temp_state,e))
					next_state,reward,done,_ = self.env.step(action)
					reward = np.clip(reward, 0, 1)
					next_state = processor.process(sess,next_state)
					temp_next_state = np.append(temp_state[:,:,1:],next_state[...,None],axis=2)

					self.D.add(state,action,reward,done,next_state)

					episode_reward += reward

					if self.counter%10000 == 0:
						self.copier.copy(sess)
						# Save model
						print("Copying weights and saving model...")
						save_path = saver.save(sess, "./saves/model.ckpt")

					if self.counter % 20 == 0:
						epsilon_summary = tf.Summary(value=[tf.Summary.Value(tag='Epsilon',simple_value=e)])
						filewriter.add_summary(epsilon_summary,self.counter)
					if done:
						break

					self._sgd_step(sess,discount)
						
					state = next_state
					temp_state = temp_next_state

	
	def _sgd_step(self,sess,discount):

		b_states, b_actions, b_rewards, b_next_states, b_done = self.D.sample()

		targets = b_rewards + discount  * np.invert(b_done).astype(np.float32) * np.amax(self.Q_.predict(sess,b_next_states),axis=1)

		self.Q.sgd_step(sess,b_states,b_actions,targets)

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
