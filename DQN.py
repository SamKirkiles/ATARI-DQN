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

	def __init__(self,env_name="Breakout-v0",run_id=None,learning_rate=0.00025,momentum=None,epsilon=0.01):

		self.env = gym.make(env_name)
		self.nA = self.env.action_space.n
		
		self.Q =  QApproximator(self.nA,"Q",learning_rate,momentum,epsilon)
		self.Q_target =  QApproximator(self.nA,"Q_target",learning_rate,momentum,epsilon)
		self.copier = WeightCopier(self.Q,self.Q_target)
		self.processor = Preprocessor()

		if run_id is None:
			self.run_id = np.random.randint(10000)
		else:
			self.run_id = run_id

	def test(self,num_episodes=1000):

		saver = tf.train.Saver()
		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:

			saver.restore(sess, tf.train.latest_checkpoint('./saves'))

			for episode in range(num_episodes):
			
				state = self.env.reset()
				state = self.processor.process(sess,state)
				temp_state = [state] * 4
				temp_state = np.stack(temp_state,axis=2)

				while True:
					self.env.render()
					probs,max_q = self._policy(sess,temp_state,0.05)
					action = np.random.choice(self.nA,p=probs)
					next_state,reward,done,_ = self.env.step(action)
					next_state = self.processor.process(sess,next_state) 
					temp_state = np.append(temp_state[:,:,1:],next_state[...,None],axis=2)
					if done:
						break


	def train(self,discount=0.99,replay_memory_init_size=50000,epsilon_start=1.0,epsilon_end=0.1,epsilon_decay_steps=500000,monitor_record_steps=250,max_replay=1000000,num_episodes=10000,restore=True,start_counter=0):

		self._counter = start_counter
	
		self.D = Memory(replay_size=max_replay)

		saver = tf.train.Saver()

		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			
			filewriter = tf.summary.FileWriter(logdir="logs/" + str(self.run_id), graph=sess.graph)

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
				next_state,reward,done,_ = self.env.step(action)
				reward = np.sign(reward)
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
			beta = 0.01
			e=1.0
			ep_loss = 0
			avg_counter = 1
			ep_max_q = 0

			self.env = Monitor(self.env,directory=monitor_path, video_callable=lambda count: count % monitor_record_steps == 0, resume=True)

			for episode in range(num_episodes):
			
				state = self.env.reset()
				state = self.processor.process(sess,state)
				temp_state = [state] * 4
				temp_state = np.stack(temp_state,axis=2)

				print("Run: " + str(self.run_id) + " Iteration: " + str(self._counter) + " Episode: " + str(episode) + "/" + str(num_episodes) + " Epsilon: " + str(e) + " Episode Reward: " + str(episode_reward) + " Replay Size: " + str(self.D.replay_length()) + " Loss: " + str(ep_loss/avg_counter), end="\r", flush=True)

				reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Reward',simple_value=episode_reward)])
				filewriter.add_summary(reward_summary,episode)
				reward_summary = tf.Summary(value=[tf.Summary.Value(tag='AVG_Q/Episode',simple_value=ep_max_q/avg_counter)])
				filewriter.add_summary(reward_summary,episode)
				reward_summary = tf.Summary(value=[tf.Summary.Value(tag='AVG_Loss/Episode',simple_value=ep_loss/avg_counter)])
				filewriter.add_summary(reward_summary,episode)

				episode_reward = 0
				save_path = saver.save(sess, "./saves/model.ckpt")
				avg_counter = 1
				ep_max_q = 0
				ep_loss = 0

				while True:

					e = self._get_epsilon(epsilon_start=epsilon_start,epsilon_end=epsilon_end,epsilon_decay_steps=epsilon_decay_steps,i=self._counter)

					probs,max_q = self._policy(sess,temp_state,e)
					action = np.random.choice(self.nA,p=probs)
					next_state,reward,done,_ = self.env.step(action)
					reward = np.sign(reward)
					next_state = self.processor.process(sess,next_state) 
					temp_next_state = np.append(temp_state[:,:,1:],next_state[...,None],axis=2)

					self.D.add(state,action,reward,done,next_state)

					episode_reward += reward
					ep_max_q += max_q
					self._counter += 1
					avg_counter += 1

					loss = self._sgd_step(sess,discount)
					ep_loss += loss

					if self._counter%10000 == 0:
						self.copier.copy(sess)
					if done:
						break

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
		max_q = np.max(q_values)
		A[best_action] += 1-epsilon
		return A,max_q




