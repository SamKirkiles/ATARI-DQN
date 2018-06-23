from DQN import DQN

def main():
	breakout = DQN(env_name="Breakout-v0")	
	breakout.train(
		discount=0.99,
		replay_memory_init_size=50000,
		epsilon_start=1.0,
		epsilon_end=0.1,
		epsilon_decay_steps=500000,
		monitor_record_steps=50,
		max_replay=500000,
		restore=False)
	#breakout.demo()

if __name__ == "__main__":
	main()