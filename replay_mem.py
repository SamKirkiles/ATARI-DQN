import pickle
import numpy as np
import os
import random

class Memory:

	def __init__(self,directory="./replay",prefix="memory",replay_size=1000000,batch_size=32):
		"""
		Holds data structures up to replay_size while keeping only 1/num_files in actual memory
		Pickles remaining files in persisted storage
		Access each file of batch size mem_size at random when switch_sample_memory is called
		Sample takes random sample from current loaded_mem and returns it
		"""

		self.buffer = 4

		self.states = np.empty((self.replay_size,84,84),dtype=np.uint8)
		self.reward = np.empty((self.replay_size),dtype=np.bool)
		self.terminate = np.empty((self.replay_size),dtype=np.bool)
			
		self.state_buffer = np.empty((self.buffer,84,84,4))
		self.next_state_buffer = np.empty((self.buffer,84,84,4))
		
		self.replay_size = replay_size

		self.current = 0
		self.filled = False

	def add(self,state,reward,terminate,next_state):

		self.states[self.current] = state
		self.reward[self.current] = reward
		self.terminate[self.current] = terminate

		if self.states.shape[0] >= self.replay_size:
			self.current = 0
			self.filled = True


	def _get_sequence(self,index):

		if index >= self.buffer-1:
			state = self.states[(index-self.buffer+1):(index + 1),....]
		else:
			# Clip so our index doesn't go our of bounds
			if not self.filled:
				indexes = [(index - i) % self.current for i in reversed(range(self.buffer))]
			else:
				indexes = [(index - i) % self.replay_size for i in reversed(range(self.buffer))]
			state = self.states[indexes,...]

		return np.transpose(state,(1,2,0))

	def sample(self):

		if self.filled:
			sample=self.replay_size - 2 
		else:
			sample=self.current

		index = np.random.choice(sample,self.batch_size)

		states = self.states[index]