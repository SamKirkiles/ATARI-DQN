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

	def __init__(self,env_name="Breakout-v0"):

		self.env = gym.make(env_name)
		self.nA = self.env.action_space.n
		
		self.Q =  QApproximator(self.nA,"Q")
		self.Q_target =  QApproximator(self.nA,"Q_target")
		self.copier = WeightCopier(self.Q,self.Q_target)
		self.processor = Preprocessor()

		self.runid = np.random.randint(10000)
		
		self.actions = np.zeros(4)

	def train(self,discount=0.99,replay_memory_init_size=50000,epsilon_start=1.0,epsilon_end=0.1,epsilon_decay_steps=500000,monitor_record_steps=250,max_replay=1000000,restore=True):

		self._counter = 0
		self._episode = 0
		self.D = Memory(replay_size=max_replay)


		saver = tf.train.Saver(max_to_keep=5,keep_checkpoint_every_n_hours=1)

		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			
			filewriter = tf.summary.FileWriter(logdir="logs/" + str(self.runid), graph=sess.graph)


			monitor_path = "./monitor"
			if not os.path.exists(monitor_path):
				os.makedirs(monitor_path)


			if restore:
				saver.restore(sess, tf.train.latest_checkpoint('./saves'))
			else:
				sess.run(tf.global_variables_initializer())


			# Initialize replay memory
			state = self.env.reset()
			state = self.processor.process(sess,state)
			temp_state = [state] * 4
			temp_state = np.stack(temp_state,axis=2)

			for _ in range(replay_memory_init_size):
				action = np.random.choice(self.nA)
				self.actions[action] += 1	
				next_state,reward,done,_ = self.env.step(action)
				next_state = self.processor.process(sess,next_state)
				temp_next_state = np.append(temp_state[:,:,1:],next_state[...,None],axis=2)
				self.D.add(state,action,reward,done,next_state)
				if done:
					state = self.env.reset()
					state = self.processor.process(sess,state)
					temp_state = [state] * 4
					temp_state = np.stack(temp_state,axis=2)
				else:
					state = next_state
					temp_state = temp_next_state

			episode_reward = 0
			avg_reward = 0
			beta = 0.01
			e=1.0
			loss = 0

			self.env = Monitor(self.env,directory=monitor_path, video_callable=lambda count: count % monitor_record_steps == 0, resume=True)

			while True:
			
				state = self.env.reset()
				state = self.processor.process(sess,state)
				temp_state = [state] * 4
				temp_state = np.stack(temp_state,axis=2)

				reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Reward',simple_value=episode_reward)])
				filewriter.add_summary(reward_summary,self._counter)

				print("Iteration: " + str(self._counter) + " Episode: " + str(self._episode) + " Epsilon: " + str(e) + " Episode Reward: " + str(episode_reward) + " Replay Size: " + str(self.D.replay_length()) + " Loss: " + str(loss) +  " Actions " + str(self.actions), end="\r", flush=True)

				episode_reward = 0
				self._episode += 1
				
				while True:

					self._counter += 1
					e = self._get_epsilon(epsilon_start=epsilon_start,epsilon_end=epsilon_end,epsilon_decay_steps=epsilon_decay_steps,i=self._counter)


					action = np.random.choice(self.nA,p=self._policy(sess,temp_state,e))
					self.actions[action] += 1	
					next_state,reward,done,_ = self.env.step(action)
					next_state = self.processor.process(sess,next_state) 
					temp_next_state = np.append(temp_state[:,:,1:],next_state[...,None],axis=2)

					self.D.add(state,action,reward,done,next_state)

					episode_reward += reward

					if self._counter%10000 == 0:
						self.copier.copy(sess)
						# Save model
						print("Copying weights and saving model...")
						save_path = saver.save(sess, "./saves/model.ckpt")

					if self._counter % 20 == 0:
						epsilon_summary = tf.Summary(value=[tf.Summary.Value(tag='Epsilon',simple_value=e)])
						filewriter.add_summary(epsilon_summary,self._counter)

					if done:
						break

					loss,loss_summary = self._sgd_step(sess,discount)
					filewriter.add_summary(loss_summary,self._counter)

					state = next_state
					temp_state = temp_next_state

	def _get_epsilon(self,epsilon_start=1.0,epsilon_end=0.1,epsilon_decay_steps=500000,i=None):
		if i > epsilon_decay_steps:
			return epsilon_end
		else:
			decrease = (epsilon_start-epsilon_end)/epsilon_decay_steps
			return epsilon_start - (decrease * i)
	def _sgd_step(self,sess,discount):

		b_states, b_actions, b_rewards, b_next_states, b_done = self.D.sample()

		targets = b_rewards + discount  * np.invert(b_done).astype(np.float32) * np.amax(self.Q_target.predict(sess,b_next_states),axis=1)

		return self.Q.sgd_step(sess,b_states,b_actions,targets)


	def _policy(self,sess,state,epsilon=0.1):

		# Epsilon Greedy Policy

		A = np.zeros(self.nA)
		A += (epsilon/self.nA)
		q_values = self.Q.predict(sess,state[None,...])[0]
		best_action = np.argmax(q_values)
		A[best_action] += 1-epsilon
		return A




