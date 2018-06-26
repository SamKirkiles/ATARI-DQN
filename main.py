from DQN import DQN

def main():

	breakout = DQN(env_name="Breakout-v0",run_id=None,learning_rate=0.00025,momentum=0.95,epsilon=0.01)	
	"""
	breakout.train(
		discount=0.99,
		replay_memory_init_size=50000,
		epsilon_start=1.0,
		epsilon_end=0.1,
		epsilon_decay_steps=500000,
		monitor_record_steps=50,
		max_replay=800000,
		num_episodes=15000,
		restore=True,
		start_counter=0)
	"""
	breakout.test()
	print("Done")


if __name__ == "__main__":
	main()